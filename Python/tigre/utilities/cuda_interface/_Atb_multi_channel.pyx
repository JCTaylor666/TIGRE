cimport numpy as np
import numpy as np

from tigre.utilities.cuda_interface._types cimport Geometry as c_Geometry, convert_to_c_geometry, free_c_geometry
from tigre.utilities.cuda_interface._gpuUtils cimport GpuIds as c_GpuIds, convert_to_c_gpuids, free_c_gpuids

np.import_array()

from libc.stdlib cimport malloc, free
from tigre.utilities.errors import TigreCudaCallError

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from "multi_channel_backprojection.hpp":
    cdef int voxel_backprojection_multi_channel(float* projections,
                                                c_Geometry geo,
                                                float* result,
                                                float const* alphas,
                                                int nalpha,
                                                int nChannels,
                                                const c_GpuIds& gpuids)


def cuda_raise_errors(error_code):
    if error_code:
        raise TigreCudaCallError('Atb_multi_channel:', error_code)


def _Atb_multi_channel_ext(np.ndarray[np.float32_t, ndim=4] projections,
                           geometry,
                           np.ndarray[np.float32_t, ndim=2] angles,
                           gpuids=None):
    cdef int total_projections = angles.shape[0]
    cdef int nChannels = projections.shape[1]

    cdef c_GpuIds* c_gpuids = convert_to_c_gpuids(gpuids)
    if not c_gpuids:
        raise MemoryError("Error loading gpuIds")

    cdef c_Geometry* c_geometry = convert_to_c_geometry(geometry, total_projections)

    angles = np.ascontiguousarray(angles)
    projections = np.ascontiguousarray(projections)

    cdef float* c_angles = <float*> angles.data
    cdef float* c_projections = <float*> projections.data

    cdef unsigned long long voxels_per_channel = <unsigned long long>geometry.nVoxel[0] * <unsigned long long>geometry.nVoxel[1] * <unsigned long long>geometry.nVoxel[2]
    cdef float* c_model = <float*> malloc(voxels_per_channel * <unsigned long long>nChannels * sizeof(float))
    if not c_model:
        free_c_geometry(c_geometry)
        free_c_gpuids(c_gpuids)
        raise MemoryError("Unable to allocate volume buffer")

    try:
        cuda_raise_errors(
            voxel_backprojection_multi_channel(
                c_projections,
                c_geometry[0],
                c_model,
                c_angles,
                total_projections,
                nChannels,
                c_gpuids[0],
            )
        )
    except Exception:
        free_c_geometry(c_geometry)
        free_c_gpuids(c_gpuids)
        free(c_model)
        raise

    free_c_geometry(c_geometry)
    free_c_gpuids(c_gpuids)

    cdef np.npy_intp shape[4]
    shape[0] = <np.npy_intp>nChannels
    shape[1] = <np.npy_intp>(<np.npy_long>geometry.nVoxel[0])
    shape[2] = <np.npy_intp>(<np.npy_long>geometry.nVoxel[1])
    shape[3] = <np.npy_intp>(<np.npy_long>geometry.nVoxel[2])

    model = np.PyArray_SimpleNewFromData(4, shape, np.NPY_FLOAT32, c_model)
    PyArray_ENABLEFLAGS(model, np.NPY_ARRAY_OWNDATA)

    return model
