"""Fused CUDA weighted cross products with parallel reduction over spatial DOFs.

Distributed under the Boost Software License, Version 1.0.
Tiny output matrices with a very long reduction dimension are handled by
spatial chunks rather than a skinny GEMM. Numerical work remains on CUDA.
"""

from gpu import cp


KERNEL = cp.RawKernel(
    r"""
    extern "C" __global__ void products(const double* left, const double* right,
        const double* weights, double* partial, int points, int lcols, int rcols,
        int chunks)
    {
      int entry=blockIdx.x/chunks, chunk=blockIdx.x%chunks;
      int r=entry%rcols, l=(entry/rcols)%lcols, batch=entry/(lcols*rcols);
      int first=chunk*1024+threadIdx.x;
      double value=0;
      for (int offset=0; offset<1024; offset+=256) {
        int i=first+offset;
        if (i<points)
          value += weights[i] * left[((long long)batch*points+i)*lcols+l]
                              * right[((long long)batch*points+i)*rcols+r];
      }
      __shared__ double buffer[256];
      buffer[threadIdx.x]=value;
      __syncthreads();
      for (int stride=128; stride>0; stride/=2) {
        if (threadIdx.x<stride) buffer[threadIdx.x]+=buffer[threadIdx.x+stride];
        __syncthreads();
      }
      if (threadIdx.x==0) partial[(long long)entry*chunks+chunk]=buffer[0];
    }
    """,
    "products",
)


def weighted_products(left, right, weights):
    """Return left* diag(weights) right for arrays [batch, spatial DOF, column]."""
    if left.ndim != 3 or right.ndim != 3 or left.shape[:2] != right.shape[:2]:
        raise ValueError("Expected matching batch and spatial dimensions")
    batches, points, lcols = left.shape
    rcols = right.shape[-1]
    if weights.shape != (points,):
        raise ValueError("One weight per spatial degree of freedom is required")
    if any(x.dtype != cp.float64 for x in (left, right, weights)):
        raise ValueError("The weighted reduction uses double precision")
    left, right, weights = map(cp.ascontiguousarray, (left, right, weights))
    chunks = (points + 1023) // 1024
    entries = batches * lcols * rcols
    partial = cp.empty((entries, chunks))
    KERNEL(
        (entries * chunks,),
        (256,),
        (left, right, weights, partial, points, lcols, rcols, chunks),
    )
    return cp.sum(partial, axis=1).reshape(batches, lcols, rcols)
