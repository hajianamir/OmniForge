"""Portable state for optional CPU/GPU Torch models in signed OmniForge artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

from omniforge.core.exceptions import DependencyMissingError, ModelSerializationError


class TorchStateMixin:
    """Serialize tensors as CPU arrays and reconstruct only the wrapper's own architecture."""

    _torch_model_attribute = "_torch_net"

    def __getstate__(self) -> dict[str, Any]:
        state = vars(self).copy()
        net = state.get(self._torch_model_attribute)
        if net is None:
            return state
        config = state.pop("_torch_build_config", None)
        if config is None:
            raise ModelSerializationError("This Torch fit lacks a versioned architecture configuration. Refit it before saving.")
        state[self._torch_model_attribute] = None
        state["_torch_portable_state"] = {
            "version": 1,
            "config": config,
            "weights": {name: tensor.detach().cpu().numpy().copy() for name, tensor in net.state_dict().items()},
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        payload = state.pop("_torch_portable_state", None)
        vars(self).update(state)
        if payload is None:
            return
        if payload.get("version") != 1:
            raise ModelSerializationError("Unsupported Torch model state version.")
        try:
            import torch
        except ImportError as exc:
            raise DependencyMissingError("Install PyTorch to restore this model artifact.") from exc
        config = payload["config"]
        with torch.random.fork_rng(devices=[]):
            builder: Any = self
            net = builder._build_torch_net(**config)
        weights = payload["weights"]
        expected = net.state_dict()
        if weights.keys() != expected.keys():
            raise ModelSerializationError("Torch artifact parameters do not match the fitted architecture.")
        restored = {}
        for name, tensor in expected.items():
            array = weights[name]
            if not isinstance(array, np.ndarray) or array.shape != tuple(tensor.shape):
                raise ModelSerializationError("Torch artifact parameter shapes do not match the fitted architecture.")
            restored[name] = torch.from_numpy(array.copy())
            if restored[name].dtype != tensor.dtype:
                raise ModelSerializationError("Torch artifact parameter dtype does not match the fitted architecture.")
        net.load_state_dict(restored, strict=True)
        net.eval()
        setattr(self, self._torch_model_attribute, net)
        self._torch_build_config = config
