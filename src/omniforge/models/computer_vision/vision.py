"""Computer Vision Engine: HOG/Color/Texture descriptors, classification, object detection & augmentation.

Uses OpenCV (``cv2``) when available and seamlessly falls back to full Pillow + SciPy + NumPy
implementations so all computer vision workflows execute reliably in every environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from omniforge.core.exceptions import ModelError

try:
    import cv2  # type: ignore[import-not-found]

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - verified via explicit fallback tests
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False


def _otsu_threshold(gray_u8: np.ndarray) -> int:
    """Compute exact Otsu global threshold maximizing between-class variance."""
    hist, _ = np.histogram(gray_u8.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return 127
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256, dtype=np.float64))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = np.where(denom > 1e-12, ((mu_t * omega - mu) ** 2) / denom, 0.0)
    return int(np.argmax(sigma_b2))


class ComputerVisionEngine:
    """Production computer vision pipeline for image feature extraction, classification, and object detection."""

    def __init__(self, image_size: tuple[int, int] = (64, 64), random_seed: int = 42) -> None:
        self.image_size = image_size
        self.random_seed = random_seed
        self._scaler = StandardScaler()
        self._clf: RandomForestClassifier | None = None

    @staticmethod
    def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB uint8 image to float32 grayscale array."""
        if _HAS_CV2 and cv2 is not None:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return (
            0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)
        )

    def load_image(self, image_or_path: np.ndarray | str | Path) -> np.ndarray:
        """Load and resize an image to canonical RGB uint8 format."""
        if isinstance(image_or_path, (str, Path)):
            path_str = str(image_or_path)
            if _HAS_CV2 and cv2 is not None:
                bgr = cv2.imread(path_str)
                if bgr is None:
                    raise ModelError(f"Failed to read image file: {image_or_path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                try:
                    from PIL import Image

                    with Image.open(path_str) as pil_img:
                        rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
                except Exception as exc:
                    raise ModelError(f"Failed to read image file: {image_or_path}", cause=exc) from exc
        else:
            arr = np.asarray(image_or_path)
            if arr.ndim == 2:
                rgb = np.stack([arr, arr, arr], axis=-1)
            else:
                rgb = arr[:, :, :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        target_w, target_h = self.image_size
        if _HAS_CV2 and cv2 is not None:
            return cv2.resize(rgb, (target_w, target_h))

        try:
            from PIL import Image

            resized = Image.fromarray(rgb).resize((target_w, target_h), Image.Resampling.BILINEAR)
            return np.asarray(resized, dtype=np.uint8)
        except Exception:
            zoom_y = target_h / max(rgb.shape[0], 1)
            zoom_x = target_w / max(rgb.shape[1], 1)
            zoomed = ndimage.zoom(rgb, (zoom_y, zoom_x, 1.0), order=1)
            return np.clip(zoomed, 0, 255).astype(np.uint8)

    def extract_descriptor(self, image_or_path: np.ndarray | str | Path) -> np.ndarray:
        """Extract combined RGB color histogram + gradient orientation histogram (HOG-like) + edge statistics."""
        rgb = self.load_image(image_or_path)
        gray = self._rgb_to_gray(rgb)

        # 1. Normalized RGB 8-bin histograms (24 dims)
        hist_parts: list[np.ndarray] = []
        for ch in range(3):
            h, _ = np.histogram(rgb[:, :, ch], bins=8, range=(0, 256), density=True)
            hist_parts.append(h.astype(np.float32))

        # 2. Gradient magnitude & orientation histogram (8 orientation bins)
        if _HAS_CV2 and cv2 is not None:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
            edge_density = float(np.mean(edges > 0))
        else:
            gx = ndimage.sobel(gray, axis=1).astype(np.float32)
            gy = ndimage.sobel(gray, axis=0).astype(np.float32)
            mag = np.hypot(gx, gy)
            ang = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0
            lap_var = float(ndimage.laplace(gray).var())
            edge_thresh = max(float(np.percentile(mag, 85)), 20.0)
            edge_density = float(np.mean(mag >= edge_thresh))

        orient_hist, _ = np.histogram(ang, bins=8, range=(0, 360), weights=mag)
        orient_hist = orient_hist / (float(np.sum(orient_hist)) + 1e-6)

        stats_vec = np.array(
            [
                float(gray.mean()),
                float(gray.std()),
                float(mag.mean()),
                float(mag.std()),
                lap_var,
                edge_density,
            ],
            dtype=np.float32,
        )
        return np.concatenate([*hist_parts, orient_hist.astype(np.float32), stats_vec])

    def fit_classifier(self, images: list[np.ndarray | str | Path], labels: list[Any]) -> ComputerVisionEngine:
        """Train an image classifier on extracted visual descriptors."""
        X = np.vstack([self.extract_descriptor(img) for img in images])
        X_s = self._scaler.fit_transform(X)
        self._clf = RandomForestClassifier(n_estimators=120, random_state=self.random_seed, n_jobs=-1)
        self._clf.fit(X_s, labels)
        return self

    def predict_classifier(self, images: list[np.ndarray | str | Path]) -> np.ndarray:
        """Predict image class labels."""
        if self._clf is None:
            raise ModelError("Call fit_classifier() before predict_classifier().")
        X = np.vstack([self.extract_descriptor(img) for img in images])
        return np.asarray(self._clf.predict(self._scaler.transform(X)))

    def detect_objects(
        self, image_or_path: np.ndarray | str | Path, *, min_area: int = 25
    ) -> list[dict[str, Any]]:
        """Detect salient object bounding boxes via Otsu thresholding & contour/component extraction."""
        rgb = self.load_image(image_or_path)
        gray_u8 = np.clip(self._rgb_to_gray(rgb), 0, 255).astype(np.uint8)
        detections: list[dict[str, Any]] = []

        if _HAS_CV2 and cv2 is not None:
            blurred = cv2.GaussianBlur(gray_u8, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for idx, cnt in enumerate(contours):
                area = float(cv2.contourArea(cnt))
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(
                    {
                        "object_id": idx,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area_px": round(area, 2),
                        "aspect_ratio": round(w / max(h, 1), 3),
                    }
                )
        else:
            blurred = ndimage.gaussian_filter(gray_u8.astype(np.float32), sigma=1.2)
            thresh = _otsu_threshold(np.clip(blurred, 0, 255).astype(np.uint8))
            binary = blurred > thresh
            labeled, n_components = ndimage.label(binary)
            slices = ndimage.find_objects(labeled)
            for idx, sl in enumerate(slices):
                if sl is None:
                    continue
                comp_mask = labeled[sl] == (idx + 1)
                area = float(np.sum(comp_mask))
                if area < min_area:
                    continue
                y_slice, x_slice = sl[0], sl[1]
                x, y = int(x_slice.start), int(y_slice.start)
                w, h = int(x_slice.stop - x_slice.start), int(y_slice.stop - y_slice.start)
                detections.append(
                    {
                        "object_id": idx,
                        "bbox": [x, y, w, h],
                        "area_px": round(area, 2),
                        "aspect_ratio": round(w / max(h, 1), 3),
                    }
                )

        detections.sort(key=lambda d: d["area_px"], reverse=True)
        return detections

    @staticmethod
    def augment(image: np.ndarray, *, flip_horizontal: bool = True, brightness_delta: float = 0.0) -> np.ndarray:
        """Apply deterministic geometric and photometric augmentations."""
        out = image.copy()
        if flip_horizontal:
            out = cv2.flip(out, 1) if (_HAS_CV2 and cv2 is not None) else np.ascontiguousarray(out[:, ::-1, ...])
        if brightness_delta != 0.0:
            out = np.clip(out.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)
        return out
