"""RigidMask inference adapter for current PyTorch/torchvision."""

import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import torch

from .vendor.rigidmask.models.VCNplus import VCN, WarpModule, flow_reg


def load_checkpoint_safely(path):
    """Load official tensor/numpy checkpoints without arbitrary pickle globals."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        if "numpy.core.multiarray._reconstruct" not in str(error):
            raise

    # The 2021 checkpoint stores mean values as numpy arrays. PyTorch 2.5's
    # restricted loader needs those exact numpy types explicitly allowlisted.
    from numpy.core.multiarray import _reconstruct
    previous_module = _reconstruct.__module__
    _reconstruct.__module__ = "numpy.core.multiarray"
    dtype_types = {
        type(np.dtype(value)) for value in (
            np.float16, np.float32, np.float64, np.int32, np.int64,
            np.uint8, np.bool_)
    }
    try:
        with torch.serialization.safe_globals(
                [_reconstruct, np.ndarray, np.dtype, *dtype_types]):
            return torch.load(path, map_location="cpu", weights_only=True)
    finally:
        _reconstruct.__module__ = previous_module


class RigidMaskModel:
    def __init__(self, rigidmask_weights: str, midas_weights: str,
                 device: str = "cuda:0", test_resolution: float = 0.5,
                 pose_ransac_threshold: float = 0.0015):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable to this process. "
                "Run rigid_mask_env_check outside a restricted sandbox.")
        self.device = torch.device(device)
        self.test_resolution = float(test_resolution)
        self.last_diagnostics = {}
        self.diagnostics_enabled = True
        self._last_diagnostics_time = -float('inf')
        self._shape: Tuple[int, int] = (640, 384)

        self.model = VCN(
            [1, self._shape[0], self._shape[1]],
            md=[4, 4, 4, 4, 4], fac=1, exp_unc=True,
            pose_ransac_threshold=pose_ransac_threshold,
            foreground_only=True)
        checkpoint = load_checkpoint_safely(rigidmask_weights)
        self.mean_left = np.asarray(checkpoint["mean_L"]).mean(0)
        self.mean_right = np.asarray(checkpoint["mean_R"]).mean(0)
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint["state_dict"].items()
        }
        model_keys = set(self.model.state_dict())
        state = {key: value for key, value in state.items()
                 if key in model_keys}
        incompatible = self.model.load_state_dict(state, strict=False)
        missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("midas.")
            and not key.endswith((".flowx", ".flowy", ".grid"))
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "RigidMask checkpoint mismatch: missing={}, unexpected={}".format(
                    missing[:20], incompatible.unexpected_keys[:20]))

        midas_checkpoint = load_checkpoint_safely(midas_weights)
        if isinstance(midas_checkpoint, dict) and "state_dict" in midas_checkpoint:
            midas_checkpoint = midas_checkpoint["state_dict"]
        if isinstance(midas_checkpoint, dict) and "model" in midas_checkpoint:
            midas_checkpoint = midas_checkpoint["model"]
        self.model.midas.load_state_dict(midas_checkpoint, strict=True)
        self.model.to(self.device).eval()
        self._configure_shape(*self._shape)

    @staticmethod
    def _rounded_network_size(width: int, height: int, scale: float) -> Tuple[int, int]:
        # The coarsest 1/64 cost volume searches +/-4 pixels, so each axis
        # must retain at least five cells after downsampling.
        scaled_width = max(320, int(width * scale))
        scaled_height = max(320, int(height * scale))
        network_width = ((scaled_width + 63) // 64) * 64
        network_height = ((scaled_height + 63) // 64) * 64
        return network_width, network_height

    def _configure_shape(self, width: int, height: int):
        if self._shape == (width, height) and all(
                module is not None for module in self.model.warp_modules[1:]):
            return
        for index in range(len(self.model.reg_modules)):
            divisor = 2 ** (6 - index)
            template = getattr(self.model, "flow_reg{}".format(divisor))
            self.model.reg_modules[index] = flow_reg(
                [1, width // divisor, height // divisor],
                ent=template.ent, maxdisp=template.md, fac=template.fac
            ).to(self.device)
        for index in range(1, len(self.model.warp_modules)):
            divisor = 2 ** (6 - index)
            self.model.warp_modules[index] = WarpModule(
                [1, width // divisor, height // divisor]).to(self.device)
        self._shape = (width, height)

    def infer(self, previous_bgr: np.ndarray, current_bgr: np.ndarray,
              camera_matrix: np.ndarray, capture_spatial=False, replay_pose=None) -> np.ndarray:
        # Explicit offline-only call argument. The ROS node never supplies it.
        # Reset on every call and in finally so a replay cannot affect later inference.
        self.model.replay_pose = None
        checked = None
        if replay_pose is not None:
            checked = {}
            for key, shape in [('rotation', (1, 3)), ('translation', (1, 3)),
                               ('essential', (1, 3, 3))]:
                value = np.asarray(replay_pose[key], dtype=np.float32)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('Invalid replay pose {}'.format(key))
                checked[key] = torch.from_numpy(value.copy())
            if not np.isclose(np.linalg.norm(checked['translation'].numpy()), 1.0, atol=1e-4):
                raise ValueError('Replay translation must be a unit direction')
        self.last_diagnostics = {}
        self.model.capture_spatial_diagnostics = bool(capture_spatial and self.diagnostics_enabled)
        self.model.spatial_diagnostics = {}
        now = time.monotonic()
        self.model.collect_diagnostics = (self.diagnostics_enabled and
                                          now - self._last_diagnostics_time >= 1.0)
        self.model.stage_diagnostics = {}
        if self.model.collect_diagnostics:
            self._last_diagnostics_time = now
        if previous_bgr.shape != current_bgr.shape:
            previous_bgr = cv2.resize(
                previous_bgr, (current_bgr.shape[1], current_bgr.shape[0]))
        height, width = current_bgr.shape[:2]
        network_width, network_height = self._rounded_network_size(
            width, height, self.test_resolution)
        self._configure_shape(network_width, network_height)

        # OpenCV input is BGR. Official code uses RGB for MiDaS and BGR for
        # the optical-flow stream; preserve that preprocessing exactly.
        previous_rgb = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2RGB)
        current_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
        previous_resized = cv2.resize(previous_rgb, (network_width, network_height))
        current_resized = cv2.resize(current_rgb, (network_width, network_height))
        midas_input = torch.from_numpy(previous_resized / 255.0)[None].float().to(self.device)

        previous_flow = previous_resized[:, :, ::-1].copy() / 255.0 - self.mean_left
        current_flow = current_resized[:, :, ::-1].copy() / 255.0 - self.mean_right
        previous_tensor = torch.from_numpy(
            np.transpose(previous_flow, (2, 0, 1))[None]).float().to(self.device)
        current_tensor = torch.from_numpy(
            np.transpose(current_flow, (2, 0, 1))[None]).float().to(self.device)

        focal = float((camera_matrix[0, 0] + camera_matrix[1, 1]) * 0.5)
        principal_x = float(camera_matrix[0, 2])
        principal_y = float(camera_matrix[1, 2])
        values = [focal, principal_x, principal_y, 1.0, 1.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  width / float(network_width),
                  height / float(network_height), focal]
        intrinsics = [torch.tensor([value], device=self.device) for value in values]
        auxiliary = [None, None, None, intrinsics, midas_input, None]

        self.model.replay_pose = checked
        try:
            with torch.inference_mode():
                outputs = self.model(
                    torch.cat((previous_tensor, current_tensor), dim=0),
                    auxiliary, None)
        finally:
            self.model.replay_pose = None
            if self.model.collect_diagnostics:
                self.last_diagnostics = dict(
                    input_wh=[width, height], network_wh=[network_width, network_height],
                    stages=self.model.stage_diagnostics)
        foreground_logits = outputs if torch.is_tensor(outputs) else outputs[4]
        probability = torch.sigmoid(foreground_logits).detach().cpu().numpy()
        return cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)


def load_camera_matrices(calibration_file: str) -> Dict[int, np.ndarray]:
    import yaml

    with open(calibration_file, "r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream)
    matrices = {}
    for camera_id, camera in content.get("cameras", {}).items():
        matrix = np.asarray(camera.get("camera_matrix"), dtype=np.float32)
        if matrix.shape == (3, 3):
            matrices[int(camera_id)] = matrix
    return matrices
