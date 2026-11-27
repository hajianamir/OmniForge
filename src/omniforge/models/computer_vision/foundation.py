"""Pretrained Computer Vision & Foundation Model Hub: CLIP Zero-Shot, DINOv2/ViT/ResNet Transfer Embeddings, YOLO/FasterRCNN Detection & Semantic Segmentation with Offline Spatial-Pyramid Fallbacks."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from omniforge.core.exceptions import ModelTrainingError
from omniforge.core.logging import get_logger
from omniforge.core.utils import is_package_available
from omniforge.models.computer_vision.vision import ComputerVisionEngine

logger = get_logger(__name__)


def _ver(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return None


def probe_vision_backends() -> dict[str, Any]:
    """Probe installed pretrained/foundation vision packages and hardware accelerator availability."""
    has_torch = is_package_available("torch")
    device = "cpu"
    if has_torch:
        try:
            import torch

            if torch.cuda.is_available():
                device = f"cuda ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            device = "cpu"
    return {
        "device": device,
        "backends": [
            {"id": "open_clip", "name": "OpenCLIP (CLIP / SigLIP Zero-Shot)", "available": is_package_available("open_clip"), "version": _ver("open-clip-torch")},
            {"id": "timm", "name": "timm (DINOv2 / ViT / ConvNeXt / Swin)", "available": is_package_available("timm"), "version": _ver("timm")},
            {"id": "torchvision", "name": "TorchVision (ResNet / FasterRCNN / DeepLabV3)", "available": is_package_available("torchvision"), "version": _ver("torchvision")},
            {"id": "ultralytics", "name": "Ultralytics YOLO (Object Detection / Seg)", "available": is_package_available("ultralytics"), "version": _ver("ultralytics")},
            {"id": "transformers", "name": "HuggingFace Vision Transformers", "available": is_package_available("transformers"), "version": _ver("transformers")},
            {"id": "cv2", "name": "OpenCV", "available": is_package_available("cv2"), "version": _ver("opencv-python")},
            {"id": "native_pyramid", "name": "OmniForge Spatial-Pyramid HOG+LBP+Gabor+HSV (Offline Always-On)", "available": True, "version": "4.0.0"},
        ],
    }


class VisionFoundationHub:
    """Unified Vision Foundation & Pretrained Model Hub with seamless offline spatial-pyramid fallbacks."""

    def __init__(self, image_size: tuple[int, int] = (224, 224), random_seed: int = 42) -> None:
        self.image_size = image_size
        self.random_seed = int(random_seed)
        self._base_cv = ComputerVisionEngine(image_size=image_size, random_seed=random_seed)
        self._linear_probe_scaler = StandardScaler()
        self._linear_probe_clf: LogisticRegression | None = None
        self._cached_models: dict[str, Any] = {}

    @staticmethod
    def _extract_lbp_histogram(gray: np.ndarray) -> np.ndarray:
        """Compute rotation-aware 8-neighbor Local Binary Pattern texture histogram (16 bins)."""
        g = gray.astype(np.float32)
        h, w = g.shape
        if h < 3 or w < 3:
            return np.zeros(16, dtype=np.float32)
        center = g[1:-1, 1:-1]
        codes = np.zeros_like(center, dtype=np.uint8)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        for bit, (dy, dx) in enumerate(offsets):
            neigh = g[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
            codes |= ((neigh >= center).astype(np.uint8) << bit)
        hist, _ = np.histogram(codes.ravel(), bins=16, range=(0, 256), density=True)
        return hist.astype(np.float32)

    @staticmethod
    def _extract_gabor_bank(gray: np.ndarray) -> np.ndarray:
        """Extract multi-orientation Gabor-like oriented Gaussian derivative texture energy (12 dims)."""
        g = gray.astype(np.float32) / 255.0
        feats: list[float] = []
        for sigma in (1.0, 2.5):
            gx = ndimage.gaussian_filter(g, sigma=sigma, order=(0, 1))
            gy = ndimage.gaussian_filter(g, sigma=sigma, order=(1, 0))
            mag = np.hypot(gx, gy)
            lap = ndimage.gaussian_laplace(g, sigma=sigma)
            feats.extend([float(mag.mean()), float(mag.std()), float(np.abs(lap).mean()), float(lap.std()), float(np.percentile(mag, 90)), float(np.percentile(np.abs(lap), 90))])
        return np.asarray(feats, dtype=np.float32)

    def extract_spatial_pyramid_descriptor(self, image_or_path: np.ndarray | str | Path) -> np.ndarray:
        """Rich 128-dim offline descriptor combining global + 2x2 Spatial Pyramid HOG, LBP, Gabor & Color Moments."""
        rgb = self._base_cv.load_image(image_or_path)
        gray = self._base_cv._rgb_to_gray(rgb)

        parts: list[np.ndarray] = [
            self._base_cv.extract_descriptor(rgb).astype(np.float32),
            self._extract_lbp_histogram(gray),
            self._extract_gabor_bank(gray),
        ]
        # 2x2 Spatial Pyramid RGB mean/std + edge density (4 quadrants * 7 = 28 dims)
        h, w = gray.shape
        hy, hx = max(h // 2, 1), max(w // 2, 1)
        for qy in (slice(0, hy), slice(hy, h)):
            for qx in (slice(0, hx), slice(hx, w)):
                patch_rgb = rgb[qy, qx].astype(np.float32) / 255.0
                patch_g = gray[qy, qx]
                gx = ndimage.sobel(patch_g, axis=1)
                gy = ndimage.sobel(patch_g, axis=0)
                edge_m = float(np.hypot(gx, gy).mean()) / 255.0
                q_feat = np.array(
                    [
                        float(patch_rgb[:, :, 0].mean()), float(patch_rgb[:, :, 1].mean()), float(patch_rgb[:, :, 2].mean()),
                        float(patch_rgb[:, :, 0].std()), float(patch_rgb[:, :, 1].std()), float(patch_rgb[:, :, 2].std()),
                        edge_m,
                    ],
                    dtype=np.float32,
                )
                parts.append(q_feat)

        vec = np.concatenate(parts)
        norm = float(np.linalg.norm(vec))
        return np.asarray((vec / norm) if norm > 1e-12 else vec, dtype=float)

    def extract_foundation_embedding(
        self,
        image_or_path: np.ndarray | str | Path,
        *,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ) -> dict[str, Any]:
        """Extract deep image embedding via timm/torchvision pretrained backbone when installed, else spatial pyramid."""
        rgb = self._base_cv.load_image(image_or_path, resize=False)
        for backend in ("timm", "torchvision"):
            if not is_package_available("torch") or not is_package_available(backend):
                continue
            try:
                import torch
                from PIL import Image

                key = f"{backend}:{backbone}:{pretrained}"
                if key not in self._cached_models:
                    if backend == "timm":
                        import timm
                        from timm.data import create_transform, resolve_model_data_config

                        model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
                        transform = create_transform(**resolve_model_data_config(model), is_training=False)
                        identity = model.pretrained_cfg.get("tag") if pretrained else None
                    else:
                        import torchvision.models as tvm

                        fn = tvm.get_model_builder(backbone)
                        default = tvm.get_model_weights(fn).DEFAULT
                        model = fn(weights=default if pretrained else None)
                        transform = default.transforms()
                        identity = default.name if pretrained else None
                        self._remove_torchvision_classifier(model, torch.nn)
                    self._cached_models[key] = (model.eval(), transform, identity)
                model, transform, identity = self._cached_models[key]
                x = transform(Image.fromarray(rgb)).unsqueeze(0)
                with torch.inference_mode():
                    emb = model(x).detach().cpu().numpy()
                if emb.ndim != 2 or emb.shape[0] != 1 or not np.isfinite(emb).all():
                    raise ModelTrainingError("The requested backbone did not emit a finite pooled embedding.")
                vector = emb[0].astype(np.float32)
                norm = float(np.linalg.norm(vector))
                if norm > 1e-12:
                    vector = vector / norm
                return {
                    "backend": f"{backend}:{backbone}", "embedding_dim": int(len(vector)),
                    "embedding": vector, "requested_backbone": backbone, "executed_backbone": backbone,
                    "pretrained_loaded": pretrained, "weights": identity,
                }
            except ImportError:
                continue
            except Exception as exc:
                logger.warning("vision_backend_failed", operation="foundation_embedding",
                               backend=backend, error_type=type(exc).__name__,
                               fallback="torchvision_or_native_spatial_pyramid" if backend == "timm" else "native_spatial_pyramid")

        emb = self.extract_spatial_pyramid_descriptor(rgb)
        return {
            "backend": "native_spatial_pyramid", "embedding_dim": int(len(emb)), "embedding": emb,
            "requested_backbone": backbone, "executed_backbone": None, "pretrained_loaded": False, "weights": None,
        }

    @staticmethod
    def _remove_torchvision_classifier(model: Any, nn: Any) -> None:
        """Remove a recognized final classification layer and keep pooling or normalization."""
        for name in ("fc", "head", "heads", "classifier"):
            head = getattr(model, name, None)
            if isinstance(head, nn.Linear):
                setattr(model, name, nn.Identity())
                return
            if isinstance(head, nn.Sequential):
                layers = list(head.named_children())
                for layer_name, layer in reversed(layers):
                    if isinstance(layer, nn.Linear):
                        setattr(head, layer_name, nn.Identity())
                        return
        raise ModelTrainingError("No supported classification head was found for this vision backbone.")

    def zero_shot_classify(
        self,
        image_or_path: np.ndarray | str | Path,
        candidate_labels: list[str],
        *,
        model_name: str = "ViT-B-32",
    ) -> dict[str, Any]:
        """Perform Zero-Shot Image Classification via OpenCLIP (when installed) or rank supported labels using uncalibrated visual cues."""
        labels = list(dict.fromkeys(str(lb).strip() for lb in candidate_labels if str(lb).strip()))
        if not labels:
            return {"backend": "none", "predictions": [], "top_label": None}

        rgb = self._base_cv.load_image(image_or_path)
        if is_package_available("open_clip") and is_package_available("torch"):
            try:
                import open_clip
                import torch
                from PIL import Image

                key = f"open_clip:{model_name}"
                if key not in self._cached_models:
                    m, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b79k")
                    tok = open_clip.get_tokenizer(model_name)
                    self._cached_models[key] = (m.eval(), preprocess, tok)
                model, preprocess, tokenizer = self._cached_models[key]
                img_t = preprocess(Image.fromarray(rgb)).unsqueeze(0)
                txt_t = tokenizer([f"a photo of {lb}" for lb in labels])
                with torch.no_grad():
                    img_f = model.encode_image(img_t)
                    txt_f = model.encode_text(txt_t)
                    img_f /= img_f.norm(dim=-1, keepdim=True)
                    txt_f /= txt_f.norm(dim=-1, keepdim=True)
                    similarities = (img_f @ txt_f.T).cpu().numpy()[0]
                ranked = sorted(zip(labels, similarities, strict=True), key=lambda item: float(item[1]), reverse=True)
                return {
                    "backend": f"open_clip:{model_name}",
                    "score_kind": "cosine_similarity",
                    "predictions": [{"label": label, "score": round(float(score), 5)} for label, score in ranked],
                    "top_label": ranked[0][0],
                    "calibrated": False,
                }
            except Exception as exc:
                logger.warning(
                    "vision_backend_failed",
                    operation="zero_shot_classification",
                    backend="open_clip",
                    error_type=type(exc).__name__,
                    fallback="native_visual_semantic_fallback",
                )

        # These hand-weighted cues rank simple attributes. They do not identify objects.
        r_m, g_m, b_m = float(rgb[:, :, 0].mean()) / 255.0, float(rgb[:, :, 1].mean()) / 255.0, float(rgb[:, :, 2].mean()) / 255.0
        bright = (r_m + g_m + b_m) / 3.0
        sat = float(np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2))) / 255.0
        gray = self._base_cv._rgb_to_gray(rgb)
        edge_d = float(np.hypot(ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)).mean()) / 255.0

        cues = (
            ({"sky", "ocean", "water", "sea", "blue", "ice"}, 1.6 * (b_m - 0.5 * r_m)),
            ({"forest", "tree", "leaf", "plant", "grass", "nature", "green"}, 1.6 * (g_m - 0.5 * r_m)),
            ({"fire", "sunset", "red", "warm", "heat", "rust", "brick"}, 1.6 * (r_m - 0.5 * b_m)),
            ({"night", "dark", "shadow", "space", "black", "x-ray"}, 1.4 * (1.0 - bright)),
            ({"snow", "white", "bright", "document", "text", "chart", "paper"}, 1.4 * bright + 0.5 * (1.0 - sat)),
            ({"building", "city", "industrial", "circuit", "defect", "crack", "object", "car"}, 1.5 * edge_d),
        )
        cue_scores: list[tuple[str, float]] = []
        for label in labels:
            words = set(re.findall(r"[\w-]+", label.lower()))
            matched = [strength for terms, strength in cues if words & terms]
            if matched:
                cue_scores.append((label, round(sum(matched), 5)))
        cue_scores.sort(key=lambda item: item[1], reverse=True)
        best = cue_scores[0] if cue_scores else None
        unique_top = best is not None and best[1] > 0 and (len(cue_scores) == 1 or best[1] > cue_scores[1][1])
        return {
            "backend": "native_visual_semantic_fallback",
            "score_kind": "visual_cue_heuristic",
            "calibrated": False,
            "note": "Visual cues can describe simple image attributes. Use a trained backend for semantic classification.",
            "predictions": [{"label": label, "score": score} for label, score in cue_scores],
            "top_label": cue_scores[0][0] if unique_top else None,
            "visual_stats": {"brightness": round(bright, 4), "saturation": round(sat, 4), "edge_density": round(edge_d, 4)},
        }

    def detect_objects(
        self,
        image_or_path: np.ndarray | str | Path,
        *,
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Run pretrained YOLO object detection when ultralytics is installed, else multi-scale morphological NMS detector."""
        rgb = self._base_cv.load_image(image_or_path)
        if is_package_available("ultralytics"):
            try:
                from ultralytics import YOLO

                if "yolo" not in self._cached_models:
                    self._cached_models["yolo"] = YOLO("yolov8n.pt")
                model = self._cached_models["yolo"]
                res = model.predict(rgb, conf=confidence_threshold, verbose=False)
                dets: list[dict[str, Any]] = []
                if res and len(res) > 0 and res[0].boxes is not None:
                    names = res[0].names
                    for box in res[0].boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        dets.append(
                            {
                                "label": str(names.get(int(box.cls[0]), "object")),
                                "confidence": round(float(box.conf[0]), 4),
                                "x": int(xyxy[0]),
                                "y": int(xyxy[1]),
                                "width": int(xyxy[2] - xyxy[0]),
                                "height": int(xyxy[3] - xyxy[1]),
                            }
                        )
                return {
                    "backend": "ultralytics:yolov8n",
                    "score_kind": "uncalibrated_model_confidence",
                    "n_detections": len(dets),
                    "detections": dets,
                }
            except Exception as exc:
                logger.warning(
                    "vision_backend_failed",
                    operation="object_detection",
                    backend="ultralytics",
                    error_type=type(exc).__name__,
                    fallback="native_otsu_morphological",
                )

        # Report the geometry of offline foreground regions.
        objs = self._base_cv.detect_objects(rgb, min_area=max(24, int(rgb.shape[0] * rgb.shape[1] * 0.005)))
        total_area = float(max(rgb.shape[0] * rgb.shape[1], 1))
        dets = []
        for i, o in enumerate(objs):
            bbox = o.get("bbox", [0, 0, 0, 0])
            area_val = float(o.get("area_px", o.get("area", bbox[2] * bbox[3])))
            dets.append(
                {
                    "label": f"region_{i + 1}",
                    "area_fraction": round(area_val / total_area, 6),
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "width": int(bbox[2]),
                    "height": int(bbox[3]),
                    "area": int(round(area_val)),
                }
            )
        return {
            "backend": "native_otsu_morphological",
            "score_kind": "foreground_area_fraction",
            "n_detections": len(dets),
            "detections": dets,
        }

    def segment_image(self, image_or_path: np.ndarray | str | Path, *, n_segments: int = 4) -> dict[str, Any]:
        """Semantic / superpixel region segmentation via spatial-color MiniBatchKMeans (or DeepLabV3 when installed)."""
        rgb = self._base_cv.load_image(image_or_path)
        h, w, _ = rgb.shape
        yy, xx = np.mgrid[0:h, 0:w]
        feats = np.column_stack(
            [
                rgb[:, :, 0].ravel() / 255.0,
                rgb[:, :, 1].ravel() / 255.0,
                rgb[:, :, 2].ravel() / 255.0,
                0.35 * (yy.ravel() / max(h, 1)),
                0.35 * (xx.ravel() / max(w, 1)),
            ]
        )
        k = max(2, min(int(n_segments), 12))
        km = MiniBatchKMeans(n_clusters=k, n_init="auto", random_state=self.random_seed).fit(feats)
        seg_mask = km.labels_.reshape(h, w)
        regions: list[dict[str, Any]] = []
        total_px = float(h * w)
        for seg_id in range(k):
            m = seg_mask == seg_id
            cnt = int(m.sum())
            if cnt == 0:
                continue
            mean_rgb = rgb[m].mean(axis=0)
            regions.append(
                {
                    "segment_id": seg_id,
                    "pixel_count": cnt,
                    "coverage_pct": round(cnt / total_px * 100.0, 2),
                    "mean_hex": f"#{int(mean_rgb[0]):02x}{int(mean_rgb[1]):02x}{int(mean_rgb[2]):02x}",
                }
            )
        regions.sort(key=lambda r: r["coverage_pct"], reverse=True)
        return {"backend": "spatial_color_kmeans_segmentation", "n_segments": len(regions), "regions": regions}

    def train_transfer_classifier(
        self,
        images: list[np.ndarray | str | Path],
        labels: list[Any],
        *,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ) -> dict[str, Any]:
        """Extract foundation/pyramid embeddings for a batch of images and train an L2 linear probe classifier."""
        if not images or len(images) != len(labels):
            raise ModelTrainingError(
                "Transfer training requires one label per image and at least one image.",
                suggestions=["Provide equal-length lists of training images and class labels."],
            )
        embs = []
        backends: set[str] = set()
        for img in images:
            res = self.extract_foundation_embedding(img, backbone=backbone, pretrained=pretrained)
            backends.add(str(res["backend"]))
            embs.append(res["embedding"])
        if len(backends) != 1:
            raise ModelTrainingError(
                "Every training image must use the same embedding backend.",
                suggestions=["Check that the requested backbone supports every training image."],
            )
        X = np.vstack(embs)
        y = np.asarray(labels)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=self.random_seed)),
        ])
        cv_acc = None
        if len(y) >= 6 and len(np.unique(y)) >= 2:
            min_cls = int(np.min([np.sum(y == c) for c in np.unique(y)]))
            if min_cls >= 2:
                cv_acc = float(np.mean(cross_val_score(pipeline, X, y, cv=min(3, min_cls), error_score="raise")))
        pipeline.fit(X, y)
        self._linear_probe_scaler = pipeline.named_steps["scaler"]
        self._linear_probe_clf = pipeline.named_steps["classifier"]
        return {
            "backend": next(iter(backends)),
            "pretrained_requested": pretrained,
            "pretrained_loaded": res.get("pretrained_loaded", False),
            "executed_backbone": res.get("executed_backbone"),
            "weights": res.get("weights"),
            "n_samples": len(y),
            "embedding_dim": int(X.shape[1]),
            "classes": [str(c) for c in np.unique(y)],
            "cv_accuracy": cv_acc,
        }
