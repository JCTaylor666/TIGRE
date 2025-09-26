"""Compare TIGRE's single-channel backprojector with the multi-channel CUDA operator.

This script mirrors the workflow shared by the user: we construct a synthetic
volume, generate projections, and validate that the multi-channel operator
matches looping over ``tigre.Atb`` channel by channel.

The script expects a CUDA-capable environment with TIGRE installed from this
repository (``pip install .`` or ``pip install -e .``).
"""

from __future__ import annotations

import time

import numpy as np

import tigre


class ConeGeometrySpecial(tigre.utilities.geometry.Geometry):
    """Minimal cone-beam geometry helper using millimetre inputs."""

    def __init__(self, config: dict[str, float]) -> None:
        super().__init__()
        self.DSD = config["DSD"] / 1000.0
        self.DSO = config["DSO"] / 1000.0
        self.nDetector = np.array(config["nDetector"], dtype=np.int32)
        self.dDetector = np.array(config["dDetector"], dtype=np.float32) / 1000
        self.sDetector = self.nDetector * self.dDetector
        self.nVoxel = np.array(config["nVoxel"][::-1], dtype=np.int32)
        self.dVoxel = np.array(config["dVoxel"][::-1], dtype=np.float32) / 1000
        self.sVoxel = self.nVoxel * self.dVoxel
        self.offOrigin = np.array(config["offOrigin"][::-1], dtype=np.float32) / 1000
        self.offDetector = np.array(
            [config["offDetector"][1], config["offDetector"][0], 0], dtype=np.float32
        ) / 1000
        self.accuracy = float(config["accuracy"])
        self.mode = config["mode"]
        self.filter = config.get("filter", "ram-lak")


def atb_for_loop(projs: np.ndarray, geo, angles: np.ndarray) -> np.ndarray:
    """Serial reference that calls :func:`tigre.Atb` channel by channel."""

    outputs = []
    for channel in range(projs.shape[0]):
        backproj = tigre.Atb(projs[channel], geo, angles)
        outputs.append(backproj.transpose(2, 1, 0))
    return np.stack(outputs, axis=0)


if __name__ == "__main__":
    cfg = {
        "DSD": 1200.0,
        "DSO": 800.0,
        "nDetector": [128, 128],
        "dDetector": [1.0, 1.0],
        "nVoxel": [64, 64, 64],
        "dVoxel": [1.0, 1.0, 1.0],
        "offOrigin": [0.0, 0.0, 0.0],
        "offDetector": [0.0, 0.0],
        "accuracy": 0.5,
        "mode": "cone",
        "filter": "ram-lak",
        "n_projections": 60,
        "total_angle": 360.0,
        "start_angle": 0.0,
    }

    geo = ConeGeometrySpecial(cfg)
    angles = (
        np.linspace(
            0,
            cfg["total_angle"] / 180 * np.pi,
            cfg["n_projections"],
            endpoint=False,
        )
        + cfg["start_angle"] / 180 * np.pi
    ).astype(np.float32)

    nx, ny, nz = cfg["nVoxel"]
    vol = np.zeros((nx, ny, nz), dtype=np.float32)
    for (x, y, z) in [(32, 32, 32), (20, 40, 10), (45, 10, 55)]:
        vol[x, y, z] = 0.1

    vol_zyx = vol.transpose(2, 1, 0).copy()
    projs_single = tigre.Ax(vol_zyx, geo, angles)
    projs_single = projs_single[:, ::-1, :]

    n_channels = 8
    scales = np.linspace(0.5, 2.0, n_channels, dtype=np.float32).reshape(1, n_channels, 1, 1)
    projs_multi = np.repeat(projs_single[:, np.newaxis, :, :], n_channels, axis=1) * scales

    print("[Ax] projections:", projs_multi.shape)

    start = time.time()
    ref = atb_for_loop(projs_multi.transpose(1, 0, 2, 3), geo, angles)
    loop_time = time.time() - start

    start = time.time()
    recon = tigre.multi_channel_atb(
        projs_multi.astype(np.float32),
        geo,
        angles,
        layout="angles_channels",
        output_order="channels_xyz",
        vflip=True,
    )
    multi_time = time.time() - start

    max_err = float(np.max(np.abs(ref - recon)))
    print(f"Looped backprojection time: {loop_time:.3f} s")
    print(f"Multi-channel backprojection time: {multi_time:.3f} s")
    print("Maximum absolute difference:", max_err)
    np.testing.assert_allclose(recon, ref, rtol=1e-5, atol=1e-5)
    print("Validation passed.")
