/* Distributed under the Boost Software License, Version 1.0.
 * Independent backward RK4 for transport with a spatially varying sink.
 * Appended to cellular_reference.cu to reuse the prescribed velocity only.
 */
__device__ double reaction_rate(ReferencePoint p)
{
  return 1. + .5 * sin(6.28318530717958647693 * p.x);
}

extern "C" __global__ void reaction_reference(
  const double* points, double* result, int count, double end, int steps)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  ReferencePoint x = {points[2 * index], points[2 * index + 1]};
  double h = -end / steps;
  double exponent = 0.;
  for (int k = 0; k < steps; ++k)
  {
    ReferencePoint a = cellular_rhs(x);
    ReferencePoint xb = {x.x + h * a.x / 2, x.y + h * a.y / 2};
    ReferencePoint b = cellular_rhs(xb);
    ReferencePoint xc = {x.x + h * b.x / 2, x.y + h * b.y / 2};
    ReferencePoint c = cellular_rhs(xc);
    ReferencePoint xd = {x.x + h * c.x, x.y + h * c.y};
    ReferencePoint d = cellular_rhs(xd);
    exponent += h / 6 *
      (reaction_rate(x) + 2 * reaction_rate(xb) + 2 * reaction_rate(xc) +
        reaction_rate(xd));
    x.x += h / 6 * (a.x + 2 * b.x + 2 * c.x + d.x);
    x.y += h / 6 * (a.y + 2 * b.y + 2 * c.y + d.y);
  }
  result[3 * index] = x.x;
  result[3 * index + 1] = x.y;
  result[3 * index + 2] = exponent;
}
