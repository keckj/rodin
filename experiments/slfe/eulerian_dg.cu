/* Distributed under the Boost Software License, Version 1.0.
 * Matrix-free application of preassembled central-flux DG transport blocks.
 */
extern "C" __global__ void evaluate_gradients(
  const double* points, double* output, int count)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= count)
    return;
  double dr[NB], ds[NB];
  basis_gradient(points[2 * item], points[2 * item + 1], dr, ds);
  for (int i = 0; i < NB; ++i)
  {
    output[(item * NB + i) * 2] = dr[i];
    output[(item * NB + i) * 2 + 1] = ds[i];
  }
}

extern "C" __global__ void dg_rhs(const double* diagonal, const double* offdiagonal,
  const int* neighbor, const double* input, double* output, int cells, int fields)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= cells * fields * NB)
    return;
  int i = item % NB, field = (item / NB) % fields, cell = item / (NB * fields);
  double value = 0.;
  for (int j = 0; j < NB; ++j)
    value += diagonal[(cell * NB + i) * NB + j] * input[(cell * fields + field) * NB + j];
  for (int face = 0; face < 3; ++face)
  {
    int source = neighbor[cell * 3 + face];
    for (int j = 0; j < NB; ++j)
    {
      value += offdiagonal[((cell * 3 + face) * NB + i) * NB + j] *
        input[(source * fields + field) * NB + j];
    }
  }
  output[item] = value;
}
