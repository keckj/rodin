/* Distributed under the Boost Software License, Version 1.0.
 * DG/Fourier cross moments, supplied by Gaussian integration.
 */
__device__ double2 complex_product(double2 a, double2 b)
{
  return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

extern "C" __global__ void dg_to_fourier(const double2* transformed,
  const double2* moments, double2* output, int n, int modes, int fields)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= modes * modes * fields)
    return;
  int f = index % fields, ix = (index / fields) % modes, iy = index / (fields * modes);
  int half = modes / 2;
  int kx = ix <= half ? ix : ix - modes;
  int ky = iy <= half ? iy : iy - modes;
  int cell = ((ky % n + n) % n) * n + (kx % n + n) % n;
  double2 sum = {0., 0.};
  for (int j = 0; j < NP; ++j)
  {
    for (int i = 0; i < NP; ++i)
    {
      double2 factor = complex_product(moments[iy * NP + j], moments[ix * NP + i]);
      double2 value =
        complex_product(transformed[(cell * fields + f) * NP * NP + j * NP + i], factor);
      sum.x += value.x;
      sum.y += value.y;
    }
  }
  output[index] = {sum.x / (n * n), sum.y / (n * n)};
}

extern "C" __global__ void fourier_to_dg(const double2* coefficients,
  const double2* moments, const double* weights, double2* output, int n, int modes,
  int fields)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n * n * fields * NP * NP)
    return;
  int i = index % NP, j = (index / NP) % NP, f = (index / (NP * NP)) % fields;
  int rx = (index / (NP * NP * fields)) % n, ry = index / (NP * NP * fields * n);
  int half = modes / 2;
  int firstx = rx - ((rx + half) / n) * n;
  int firsty = ry - ((ry + half) / n) * n;
  double2 sum = {0., 0.};
  for (int ky = firsty; ky <= half; ky += n)
  {
    for (int kx = firstx; kx <= half; kx += n)
    {
      int ix = (kx + modes) % modes, iy = (ky + modes) % modes;
      double2 factor = complex_product(moments[iy * NP + j], moments[ix * NP + i]);
      factor.y = -factor.y;
      double2 value =
        complex_product(coefficients[(iy * modes + ix) * fields + f], factor);
      sum.x += value.x;
      sum.y += value.y;
    }
  }
  double scale = weights[i] * weights[j];
  output[index] = {sum.x / scale, sum.y / scale};
}
