/* Distributed under the Boost Software License, Version 1.0.
 * Apply independent local orthogonal matrices to mass-normalized DG values.
 */
extern "C" __global__ void local_rotation(const int* cells, const double* matrix,
  const double* input, double* output, int groups, int slots, int fields)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= groups * slots * fields * NB)
    return;
  int j = item % NB, field = (item / NB) % fields;
  int slot = (item / (NB * fields)) % slots, group = item / (NB * fields * slots);
  int size = slots * NB, row = slot * NB + j;
  double value = 0.;
  for (int k = 0; k < size; ++k)
  {
    int cell = cells[group * slots + k / NB];
    value += matrix[(group * size + row) * size + k] *
      input[(cell * fields + field) * NB + k % NB];
  }
  output[(cells[group * slots + slot] * fields + field) * NB + j] = value;
}
