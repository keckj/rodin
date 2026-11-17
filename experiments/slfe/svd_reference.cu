/* Distributed under the Boost Software License, Version 1.0.
 * Independent characteristics for an analytically nonseparable vortex.
 */
struct SvdPoint
{
  double x, y;
};

__device__ SvdPoint svd_velocity(SvdPoint p)
{
  constexpr double pi2 = 6.28318530717958647693;
  double sx, cx, sy, cy, sd, cd;
  sincos(pi2 * (p.x - 0.5), &sx, &cx);
  sincos(pi2 * (p.y - 0.5), &sy, &cy);
  sincos(pi2 * (p.x + p.y - 1.), &sd, &cd);
  double potential = 0.03 * exp(4. * (cx + cy - 2.) + 1.2 * (cd - 1.));
  return {
    -pi2 * (4. * sy + 1.2 * sd) * potential, pi2 * (4. * sx + 1.2 * sd) * potential};
}

extern "C" __global__ void svd_feet(
  const double* points, double* result, int count, double duration, int steps)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  SvdPoint p = {points[2 * index], points[2 * index + 1]};
  double h = duration / steps;
  for (int k = 0; k < steps; ++k)
  {
    SvdPoint a = svd_velocity(p);
    SvdPoint b = svd_velocity({p.x + h * a.x / 2, p.y + h * a.y / 2});
    SvdPoint c = svd_velocity({p.x + h * b.x / 2, p.y + h * b.y / 2});
    SvdPoint d = svd_velocity({p.x + h * c.x, p.y + h * c.y});
    p.x += h * (a.x + 2 * b.x + 2 * c.x + d.x) / 6;
    p.y += h * (a.y + 2 * b.y + 2 * c.y + d.y) / 6;
  }
  result[2 * index] = p.x;
  result[2 * index + 1] = p.y;
}
