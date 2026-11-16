/* Distributed under the Boost Software License, Version 1.0.
 * Cell-local dense action, including CUDA graph capture.
 */
extern "C" __global__ void apply_local_matrix(
  const double* matrix, const double* input, double* output, int cells, int fields)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cells * fields * NB)
    return;
  int i = index % NB, field = (index / NB) % fields, cell = index / (NB * fields);
  double value = 0.;
  for (int j = 0; j < NB; ++j)
    value += matrix[(cell * NB + i) * NB + j] * input[(cell * fields + field) * NB + j];
  output[index] = value;
}
