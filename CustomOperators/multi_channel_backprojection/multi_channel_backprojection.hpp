/*-------------------------------------------------------------------------
 *
 * Multi-channel CUDA backprojection interface for TIGRE.
 *
 * Based on the original single-channel implementation by Ander Biguri and
 * contributors.
 * ---------------------------------------------------------------------------
 */

#ifndef MULTI_CHANNEL_BACKPROJECTION_HPP
#define MULTI_CHANNEL_BACKPROJECTION_HPP

#include "types_TIGRE.hpp"
#include "GpuIds.hpp"

int voxel_backprojection_multi_channel(float* projections,
                                       Geometry geo,
                                       float* result,
                                       float const* const alphas,
                                       int nalpha,
                                       int nChannels,
                                       const GpuIds& gpuids);

#endif  // MULTI_CHANNEL_BACKPROJECTION_HPP
