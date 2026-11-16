/* Distributed under the Boost Software License, Version 1.0.
 * Independent RK4 trajectories: physical Fourier velocities, no grid remaps.
 */
struct ModePoint
{
  double x, y;
};

__device__ ModePoint mode_rhs(
  ModePoint p, const double* modes, int count, double mean_x, double mean_y)
{
  ModePoint result = {mean_x, mean_y};
  for (int mode = 0; mode < count; ++mode)
  {
    const double* m = modes + 5 * mode;
    double value = sin(6.28318530717958647693 * (m[0] * p.x + m[1] * p.y) + m[4]);
    result.x += m[2] * value;
    result.y += m[3] * value;
  }
  return result;
}

extern "C" __global__ void fourier_feet(const double* points, double* result, int count,
  const double* modes, int mode_count, double mean_x, double mean_y, double duration,
  int steps)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  ModePoint x = {points[2 * index], points[2 * index + 1]};
  double h = duration / steps;
  for (int k = 0; k < steps; ++k)
  {
    ModePoint a = mode_rhs(x, modes, mode_count, mean_x, mean_y);
    ModePoint b =
      mode_rhs({x.x + h * a.x / 2, x.y + h * a.y / 2}, modes, mode_count, mean_x, mean_y);
    ModePoint c =
      mode_rhs({x.x + h * b.x / 2, x.y + h * b.y / 2}, modes, mode_count, mean_x, mean_y);
    ModePoint d =
      mode_rhs({x.x + h * c.x, x.y + h * c.y}, modes, mode_count, mean_x, mean_y);
    x.x += h / 6 * (a.x + 2 * b.x + 2 * c.x + d.x);
    x.y += h / 6 * (a.y + 2 * b.y + 2 * c.y + d.y);
  }
  result[2 * index] = x.x;
  result[2 * index + 1] = x.y;
}
