"""Specialized CUDA cyclic-Jacobi eigensolver for tiny symmetric Gram matrices.

Distributed under the Boost Software License, Version 1.0.
One thread owns one matrix; this deliberately targets tiny batches and orders.
The independent CuPy eigensolver remains the validation and timing reference.
"""

from gpu import cp


KERNELS = {}


def eigh(matrix):
    n = matrix.shape[-1]
    if matrix.shape[-2] != n or not 1 <= n <= 24 or matrix.dtype != cp.float64:
        raise ValueError("Expected float64 symmetric matrices of order 1 through 24")
    matrix = cp.ascontiguousarray(matrix)
    count = matrix.size // (n * n)
    eigenvalues = cp.empty(matrix.shape[:-1])
    eigenvectors = cp.empty_like(matrix)
    residual = cp.empty(count)
    if n not in KERNELS:
        source = r"""
        #define DBL_EPSILON 2.2204460492503131e-16
        #define DBL_MIN 2.2250738585072014e-308
        extern "C" __global__ void jacobi(const double* input, double* eigen,
                                         double* rotation, double* error, int count)
        {
          int b = blockIdx.x * blockDim.x + threadIdx.x;
          if (b >= count) return;
          double a[N*N], q[N*N];
          double scale = 0;
          for (int i=0; i<N*N; ++i) {
            a[i] = input[b*N*N+i];
            q[i] = (i/N == i%N) ? 1.0 : 0.0;
            scale = fmax(scale, fabs(a[i]));
          }
          for (int sweep=0; sweep<24; ++sweep) {
            bool changed = false;
            for (int p=0; p<N; ++p) for (int r=p+1; r<N; ++r) {
              double off = a[p*N+r], app = a[p*N+p], arr = a[r*N+r];
              if (fabs(off) <= DBL_EPSILON * (fabs(app)+fabs(arr))) continue;
              changed = true;
              double tau = (arr-app)/(2*off);
              double t = copysign(1.0, tau)/(fabs(tau)+hypot(1.0,tau));
              double c = 1/sqrt(1+t*t), s = t*c;
              a[p*N+p] = app-t*off;
              a[r*N+r] = arr+t*off;
              a[p*N+r] = a[r*N+p] = 0;
              for (int k=0; k<N; ++k) {
                if (k != p && k != r) {
                  double akp=a[k*N+p], akr=a[k*N+r];
                  a[k*N+p]=a[p*N+k]=c*akp-s*akr;
                  a[k*N+r]=a[r*N+k]=s*akp+c*akr;
                }
                double qkp=q[k*N+p], qkr=q[k*N+r];
                q[k*N+p]=c*qkp-s*qkr;
                q[k*N+r]=s*qkp+c*qkr;
              }
            }
            if (!changed) break;
          }
          double remaining=0;
          for (int p=0; p<N; ++p) for (int r=0; r<N; ++r)
            if (p != r) remaining=fmax(remaining,fabs(a[p*N+r]));
          error[b]=remaining/fmax(scale,DBL_MIN);
          for (int p=0; p<N; ++p) {
            int smallest=p;
            for (int r=p+1; r<N; ++r)
              if (a[r*N+r] < a[smallest*N+smallest]) smallest=r;
            if (smallest != p) {
              double swap=a[p*N+p];
              a[p*N+p]=a[smallest*N+smallest]; a[smallest*N+smallest]=swap;
              for (int k=0; k<N; ++k) {
                swap=q[k*N+p]; q[k*N+p]=q[k*N+smallest]; q[k*N+smallest]=swap;
              }
            }
            eigen[b*N+p]=a[p*N+p];
          }
          for (int i=0; i<N*N; ++i) rotation[b*N*N+i]=q[i];
        }
        """
        KERNELS[n] = cp.RawKernel(f"#define N {n}\n" + source, "jacobi")
    KERNELS[n](
        ((count + 31) // 32,),
        (32,),
        (matrix, eigenvalues, eigenvectors, residual, count),
    )
    maximum = float(cp.max(residual))
    if not maximum <= 1e-12:
        raise RuntimeError(f"Tiny Gram Jacobi iteration failed: residual {maximum}")
    return eigenvalues, eigenvectors
