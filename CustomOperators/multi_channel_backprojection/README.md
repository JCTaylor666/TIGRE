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

The source is intentionally independent from the Python build system. You can
compile it into a static library or an object file using `nvcc`. A minimal
example (run from the repository root) that produces an object file is:

```bash
nvcc -std=c++14 -ICommon/CUDA -ICustomOperators/multi_channel_backprojection \
     -c CustomOperators/multi_channel_backprojection/multi_channel_backprojection.cu \
     -o build/multi_channel_backprojection.o
```

You may need to create the `build/` directory first. Link the resulting object
file (or library) together with TIGRE's existing CUDA sources when building a
Python extension or a custom executable.

## Python binding and verification

A dedicated Cython wrapper is shipped with the repository so the operator is
available directly from Python once TIGRE is built in editable mode:

```bash
pip install -e .
```

This exposes `tigre.Atb_multi_channel`, which accepts a projection tensor with
layout `(n_angles, n_channels, nDetecV, nDetecU)` (or `(n_channels, n_angles, …)`
when passing `layout="channels_angles"`). The helper mirrors the single-channel
API and returns either `(channels, Z, Y, X)` or `(channels, X, Y, Z)` depending
on `output_order`.

The [`example_compare.py`](example_compare.py) script reproduces the workflow
from the user request and ensures the multi-channel implementation matches a
serial Python loop over `tigre.Atb`:

```bash
python CustomOperators/multi_channel_backprojection/example_compare.py
```

The script prints the execution time for both approaches and finishes with an
`assert_allclose` check that the results are numerically identical.
