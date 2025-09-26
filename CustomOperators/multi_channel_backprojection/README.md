# Multi-channel backprojection operator

This folder contains a standalone CUDA implementation that extends TIGRE's
`voxel_backprojection` routine to handle multi-channel (e.g. spectral) projection
stacks in a single GPU launch. The host function exposed by
[`multi_channel_backprojection.hpp`](multi_channel_backprojection.hpp) is:

```c++
int voxel_backprojection_multi_channel(
    float* projections,
    Geometry geo,
    float* result,
    float const* const alphas,
    int nalpha,
    int nChannels,
    const GpuIds& gpuids);
```

## Data layout

To maximise reuse of the original kernels, both the projection and the volume
arrays must be stored in contiguous **C-order** memory with the following
strides:

* `projections[(angle * nChannels + channel) * nDetecV * nDetecU + v * nDetecU + u]`
* `result[(channel * nVoxelZ + z) * nVoxelY * nVoxelX + y * nVoxelX + x]`

This matches the layout you would obtain by stacking channels as the innermost
dimension when using NumPy/Python.

## Building

No manual compilation is required. The CUDA operator is wired into TIGRE's
standard Python build, so a regular installation will compile the extra kernel
alongside the existing extensions:

```bash
pip install .
```

If you prefer an editable install during development, simply use
`pip install -e .` instead. The build logic automatically invokes `nvcc` with
the same configuration used for the rest of TIGRE.

## Python binding and verification

A dedicated Cython wrapper ships with the repository so the operator is
available directly from Python after installation. Import it via the convenience
helper exposed at the top level:

```python
import tigre

recon = tigre.multi_channel_atb(...)
```

`multi_channel_atb` accepts a projection tensor with layout
`(n_angles, n_channels, nDetecV, nDetecU)` (or `(n_channels, n_angles, …)` when
passing `layout="channels_angles"`). The helper mirrors the single-channel API
and returns either `(channels, Z, Y, X)` or `(channels, X, Y, Z)` depending on
`output_order`.

The [`example_compare.py`](example_compare.py) script reproduces the workflow
from the user request and ensures the multi-channel implementation matches a
serial Python loop over `tigre.Atb`:

```bash
python CustomOperators/multi_channel_backprojection/example_compare.py
```

The script prints the execution time for both approaches and finishes with an
`assert_allclose` check that the results are numerically identical.
