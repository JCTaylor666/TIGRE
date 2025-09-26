import copy
from typing import Literal, Optional

import numpy as np

from _Atb_multi_channel import _Atb_multi_channel_ext
from .gpu import GpuIds


LayoutHint = Literal["angles_channels", "channels_angles"]
OutputOrder = Literal["channels_xyz", "channels_zyx"]


def _normalise_projections(projections: np.ndarray, n_angles: int, layout: LayoutHint) -> np.ndarray:
    if projections.ndim != 4:
        raise ValueError(
            "Expected a 4D array with axes (angles, channels, detV, detU) or (channels, angles, detV, detU)."
        )

    if layout == "angles_channels":
        if projections.shape[0] != n_angles:
            raise ValueError(
                f"Projection stack reports layout 'angles_channels' but first dimension ({projections.shape[0]}) "
                f"does not match the number of angles ({n_angles})."
            )
        return projections
    if layout == "channels_angles":
        if projections.shape[1] != n_angles:
            raise ValueError(
                f"Projection stack reports layout 'channels_angles' but second dimension ({projections.shape[1]}) "
                f"does not match the number of angles ({n_angles})."
            )
        return np.transpose(projections, (1, 0, 2, 3))

    raise ValueError(f"Unsupported layout hint: {layout}")


def Atb_multi_channel(
    projections: np.ndarray,
    geo,
    angles: np.ndarray,
    *,
    layout: LayoutHint = "angles_channels",
    output_order: OutputOrder = "channels_xyz",
    vflip: bool = False,
    gpuids: Optional[GpuIds] = None,
) -> np.ndarray:
    """Multi-channel wrapper around TIGRE's backprojector.

    Parameters
    ----------
    projections:
        Projection tensor with layout specified by ``layout`` and dtype ``float32``.
    geo:
        TIGRE geometry instance.
    angles:
        Acquisition angles in radians with shape ``(n_angles,)`` or ``(n_angles, 1|2)``.
    layout:
        ``"angles_channels"`` expects data laid out as ``(n_angles, n_channels, nDetecV, nDetecU)``.
        ``"channels_angles"`` expects ``(n_channels, n_angles, nDetecV, nDetecU)`` and will transpose it.
    output_order:
        ``"channels_xyz"`` returns ``(n_channels, nVoxelX, nVoxelY, nVoxelZ)``. ``"channels_zyx"`` keeps the
        native TIGRE layout ``(n_channels, nVoxelZ, nVoxelY, nVoxelX)``.
    vflip:
        If ``True`` the detector V dimension will be flipped back before calling the CUDA kernel. This mirrors
        the manual ``[::-1]`` flip often used during forward projection.
    gpuids:
        Optional :class:`~tigre.utilities.gpu.GpuIds` instance. A fresh instance is created when omitted.
    """
    if projections.dtype != np.float32:
        raise TypeError("Input data should be float32, not " + str(projections.dtype))
    if not np.isrealobj(projections):
        raise ValueError("Complex types not compatible for back projection.")

    geox = copy.deepcopy(geo)
    geox.check_geo(angles)
    geox.cast_to_single()

    n_angles = geox.angles.shape[0]

    layout_hint: LayoutHint = layout
    projections = _normalise_projections(np.asarray(projections), n_angles, layout_hint)

    if projections.shape[2] != geox.nDetector[0] or projections.shape[3] != geox.nDetector[1]:
        raise ValueError(
            "Expected detector dimensions ({} , {}) but received {}.".format(
                geox.nDetector[0], geox.nDetector[1], projections.shape[2:4]
            )
        )

    if vflip:
        projections = projections.copy()
        projections = projections[:, :, ::-1, :]

    if gpuids is None:
        gpuids = GpuIds()

    if geox.nVoxel[0] < len(gpuids):
        gpuids.devices = list(gpuids.devices[0 : geox.nVoxel[0]])

    volumes = _Atb_multi_channel_ext(projections, geox, geox.angles, gpuids)

    if output_order == "channels_xyz":
        return np.transpose(volumes, (0, 3, 2, 1))
    if output_order == "channels_zyx":
        return volumes

    raise ValueError(f"Unsupported output order: {output_order}")
