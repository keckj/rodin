/* Distributed under the Boost Software License, Version 1.0.
 * Evaluate the actual cell-local DG polynomial at visualization points.
 * The inverse grading is used only to find topology candidates; containment
 * and evaluation use the physical affine triangles.
 */
__device__ double undo_grading(double x, double offset, double grading)
{
  constexpr double two_pi = 6.28318530717958647693;
  double lower = x - grading / two_pi, upper = x + grading / two_pi;
  for (int iteration = 0; iteration < 48; ++iteration)
  {
    double middle = (lower + upper) / 2.;
    if (middle + grading / two_pi * sin(two_pi * (middle + offset)) < x)
      lower = middle;
    else
      upper = middle;
  }
  return (lower + upper) / 2.;
}

extern "C" __global__ void sample_fields(const double* points, int count,
  const double* triangles, int n, double grading, const double* coefficients, int fields,
  double* output, int* owners)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= count)
    return;
  Point point = {points[2 * item], points[2 * item + 1]};
  int gx = (int)floor(n * undo_grading(point.x, .15, grading));
  int gy = (int)floor(n * undo_grading(point.y, .05, grading));
  owners[item] = -1;
  for (int iy = gy - 2; iy <= gy + 2; ++iy)
  {
    for (int ix = gx - 2; ix <= gx + 2; ++ix)
    {
      for (int half = 0; half < 2; ++half)
      {
        double triangle[6];
        int cell;
        target_triangle(triangles, n, ix, iy, half, triangle, &cell);
        Point local = inverse(triangle, point);
        if (local.x < -1.e-12 || local.y < -1.e-12 || local.x + local.y > 1. + 1.e-12)
          continue;
        double values[NB];
        basis(local.x, local.y, values);
        for (int field = 0; field < fields; ++field)
        {
          double value = 0.;
          for (int j = 0; j < NB; ++j)
            value += coefficients[(cell * fields + field) * NB + j] * values[j];
          output[item * fields + field] = value;
        }
        owners[item] = cell;
        return;
      }
    }
  }
}
