"""Unified multimodal data container and modality-specific feature extractors for OmniForge."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from omniforge.core.exceptions import DataError
from omniforge.core.logging import get_logger
from omniforge.core.types import DataModality
from omniforge.core.utils import is_package_available

logger = get_logger(__name__)


@dataclass
class MultimodalDataset:
    """Container harmonizing aligned tabular, time-series, text, image, audio, graph, geo, and genomic views."""

    name: str = "multimodal_dataset"
    tabular: pd.DataFrame | None = None
    time_series: dict[str, pd.DataFrame] = field(default_factory=dict)
    text: list[str] | None = None
    images: list[np.ndarray] = field(default_factory=list)
    audio_signals: list[tuple[np.ndarray, int]] = field(default_factory=list)
    graphs: list[nx.Graph] = field(default_factory=list)
    geospatial: pd.DataFrame | None = None
    genomic_sequences: list[str] = field(default_factory=list)
    targets: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def active_modalities(self) -> list[DataModality]:
        """Return list of populated modalities inside this container."""
        mods: list[DataModality] = []
        if self.tabular is not None:
            mods.append(DataModality.TABULAR)
        if self.time_series:
            mods.append(DataModality.TIME_SERIES)
        if self.text is not None:
            mods.append(DataModality.TEXT)
        if self.images:
            mods.append(DataModality.IMAGE)
        if self.audio_signals:
            mods.append(DataModality.AUDIO)
        if self.graphs:
            mods.append(DataModality.GRAPH)
        if self.geospatial is not None:
            mods.append(DataModality.GEOSPATIAL)
        if self.genomic_sequences:
            mods.append(DataModality.GENOMIC)
        return mods


class MultimodalProcessor:
    """Transforms raw multimodal artifacts (image, video, audio, geo, IoT, genomic) into unified numeric features."""

    @staticmethod
    def extract_image_features(image: np.ndarray | str | Path) -> dict[str, float]:
        """Extract color, brightness, contrast, gradient edge, and entropy descriptors from an image."""
        if isinstance(image, (str, Path)):
            from PIL import Image

            arr = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32)
        else:
            arr = np.asarray(image, dtype=np.float32)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]

        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        gy, gx = np.gradient(gray)
        edge_mag = np.sqrt(gx**2 + gy**2)
        hist, _ = np.histogram(gray, bins=16, range=(0, 255), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))

        feats: dict[str, float] = {
            "img_height": float(arr.shape[0]),
            "img_width": float(arr.shape[1]),
            "img_aspect_ratio": float(arr.shape[1] / max(arr.shape[0], 1)),
            "img_gray_mean": float(gray.mean()),
            "img_gray_std": float(gray.std()),
            "img_edge_mean": float(edge_mag.mean()),
            "img_edge_max": float(edge_mag.max()),
            "img_entropy": entropy,
        }
        for idx, ch in enumerate(("r", "g", "b")):
            feats[f"img_{ch}_mean"] = float(arr[:, :, idx].mean())
            feats[f"img_{ch}_std"] = float(arr[:, :, idx].std())
        return feats

    @staticmethod
    def extract_audio_features(signal: np.ndarray, sample_rate: int = 16000) -> dict[str, float]:
        """Extract time-domain and spectral audio descriptors (RMS, ZCR, spectral centroid, MFCC)."""
        y = np.asarray(signal, dtype=np.float64).ravel()
        if len(y) == 0:
            raise DataError("Empty audio signal provided.")
        rms = float(np.sqrt(np.mean(y**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(y))) > 0))
        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1.0 / max(sample_rate, 1))
        spec_sum = float(np.sum(spectrum)) + 1e-12
        centroid = float(np.sum(freqs * spectrum) / spec_sum)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spec_sum))
        feats = {
            "audio_duration_s": float(len(y) / max(sample_rate, 1)),
            "audio_rms": rms,
            "audio_peak": float(np.max(np.abs(y))),
            "audio_zcr": zcr,
            "audio_spectral_centroid": centroid,
            "audio_spectral_bandwidth": bandwidth,
        }
        if is_package_available("librosa") and len(y) >= 512:
            try:
                import librosa  # type: ignore[import-not-found]

                mfccs = librosa.feature.mfcc(y=y.astype(np.float32), sr=sample_rate, n_mfcc=8)
                for i in range(mfccs.shape[0]):
                    feats[f"audio_mfcc_{i}_mean"] = float(mfccs[i].mean())
            except Exception as exc:
                logger.debug("librosa_mfcc_extraction_skipped", error=str(exc))
        return feats

    @staticmethod
    def extract_video_features(video_path: str | Path, max_frames: int = 16) -> dict[str, float]:
        """Sample frames from a video file using OpenCV and aggregate frame statistics & motion."""
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DataError(
                "OpenCV ('opencv-python') is required for video frame extraction. "
                "Install it with: pip install opencv-python",
                cause=exc,
            ) from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise DataError(f"Unable to open video file: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        indices = np.linspace(0, max(frame_count - 1, 0), min(max_frames, frame_count), dtype=int)

        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise DataError(f"No readable frames extracted from video: {video_path}")

        frame_feats = [MultimodalProcessor.extract_image_features(f) for f in frames]
        agg = {
            f"video_{k}_mean": float(np.mean([ff[k] for ff in frame_feats]))
            for k in frame_feats[0]
        }
        motions = [
            float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
            for i in range(1, len(frames))
        ]
        agg["video_fps"] = fps
        agg["video_frame_count"] = float(frame_count)
        agg["video_motion_mean"] = float(np.mean(motions)) if motions else 0.0
        return agg

    @staticmethod
    def extract_geospatial_features(
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        ref_point: tuple[float, float] | None = None,
    ) -> pd.DataFrame:
        """Compute Haversine distance, 3D Cartesian unit-sphere projections, and spatial grid bins."""
        if lat_col not in df.columns or lon_col not in df.columns:
            raise DataError(f"Geospatial columns '{lat_col}' and '{lon_col}' are required.")
        out = pd.DataFrame(index=df.index)
        lat_rad = np.radians(df[lat_col].astype(float))
        lon_rad = np.radians(df[lon_col].astype(float))
        out["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        out["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        out["geo_z"] = np.sin(lat_rad)

        ref_lat, ref_lon = (
            ref_point
            if ref_point is not None
            else (float(df[lat_col].mean()), float(df[lon_col].mean()))
        )
        r_lat, r_lon = math.radians(ref_lat), math.radians(ref_lon)
        dlat = lat_rad - r_lat
        dlon = lon_rad - r_lon
        a = np.sin(dlat / 2.0) ** 2 + np.cos(r_lat) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
        out["geo_distance_km"] = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return out

    @staticmethod
    def extract_sensor_iot_features(signal: np.ndarray, sampling_hz: float = 100.0) -> dict[str, float]:
        """Compute vibration/IoT time-frequency health features (RMS, kurtosis, crest factor, dominant freq)."""
        x = np.asarray(signal, dtype=np.float64).ravel()
        rms = float(np.sqrt(np.mean(x**2)))
        peak = float(np.max(np.abs(x))) if len(x) > 0 else 0.0
        std = float(np.std(x)) + 1e-12
        kurtosis = float(np.mean(((x - np.mean(x)) / std) ** 4))
        spectrum = np.abs(np.fft.rfft(x - np.mean(x)))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / max(sampling_hz, 1e-6))
        dom_idx = int(np.argmax(spectrum)) if len(spectrum) > 0 else 0
        return {
            "iot_mean": float(np.mean(x)),
            "iot_std": std,
            "iot_rms": rms,
            "iot_peak": peak,
            "iot_crest_factor": peak / max(rms, 1e-12),
            "iot_kurtosis": kurtosis,
            "iot_dominant_freq_hz": float(freqs[dom_idx]) if len(freqs) > 0 else 0.0,
            "iot_spectral_energy": float(np.sum(spectrum**2)),
        }

    @staticmethod
    def extract_genomic_features(sequence: str, k: int = 2) -> dict[str, float]:
        """Encode DNA/RNA nucleotide sequence into GC content, length, and normalized k-mer frequencies."""
        seq = "".join(ch for ch in sequence.upper() if ch in {"A", "C", "G", "T", "U"})
        seq = seq.replace("U", "T")
        length = max(len(seq), 1)
        gc_content = (seq.count("G") + seq.count("C")) / length
        feats: dict[str, float] = {
            "genomic_length": float(len(seq)),
            "genomic_gc_content": float(gc_content),
            "genomic_at_skew": float((seq.count("A") - seq.count("T")) / max(seq.count("A") + seq.count("T"), 1)),
            "genomic_gc_skew": float((seq.count("G") - seq.count("C")) / max(seq.count("G") + seq.count("C"), 1)),
        }
        bases = ("A", "C", "G", "T")
        if k == 2:
            total_kmers = max(len(seq) - 1, 1)
            for b1 in bases:
                for b2 in bases:
                    kmer = b1 + b2
                    count = sum(1 for i in range(len(seq) - 1) if seq[i : i + 2] == kmer)
                    feats[f"kmer_{kmer}"] = float(count / total_kmers)
        return feats
