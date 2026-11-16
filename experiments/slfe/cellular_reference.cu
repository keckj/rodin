/* Distributed under the Boost Software License, Version 1.0.
 * Independent autonomous RK4 characteristic reference for the cellular flow.
 */
struct ReferencePoint
{
  double x, y;
};

__device__ ReferencePoint cellular_rhs(ReferencePoint p)
{
  double sx, cx, sy, cy;
  sincos(6.28318530717958647693 * p.x, &sx, &cx);
  sincos(6.28318530717958647693 * p.y, &sy, &cy);
  return {.5 * (1. - cx) * sy, -.5 * (1. - cy) * sx};
}

extern "C" __global__ void reference_feet(
  const double* points, double* result, int count, double duration, int steps)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  ReferencePoint x = {points[2 * index], points[2 * index + 1]};
  double h = duration / steps;
  for (int k = 0; k < steps; ++k)
  {
    ReferencePoint a = cellular_rhs(x);
    ReferencePoint b = cellular_rhs({x.x + h * a.x / 2, x.y + h * a.y / 2});
    ReferencePoint c = cellular_rhs({x.x + h * b.x / 2, x.y + h * b.y / 2});
    ReferencePoint d = cellular_rhs({x.x + h * c.x, x.y + h * c.y});
    x.x += h / 6 * (a.x + 2 * b.x + 2 * c.x + d.x);
    x.y += h / 6 * (a.y + 2 * b.y + 2 * c.y + d.y);
  }
  result[2 * index] = x.x;
  result[2 * index + 1] = x.y;
}
