/* Distributed under the Boost Software License, Version 1.0.
 * Independent RK4 characteristics for a sum of product Gaussian vortices.
 */
struct ProductPoint
{
  double x, y;
};

__device__ ProductPoint product_velocity(ProductPoint p, const double* terms, int rank)
{
  constexpr double pi2 = 6.28318530717958647693;
  ProductPoint result = {0., 0.};
  for (int j = 0; j < rank; ++j)
  {
    const double* term = terms + 4 * j;
    double sx, cx, sy, cy;
    sincos(pi2 * (p.x - term[2]), &sx, &cx);
    sincos(pi2 * (p.y - term[3]), &sy, &cy);
    double potential = term[0] * exp(term[1] * (cx + cy - 2.));
    result.x -= pi2 * term[1] * sy * potential;
    result.y += pi2 * term[1] * sx * potential;
  }
  return result;
}

extern "C" __global__ void product_feet(const double* points, double* result, int count,
  const double* terms, int rank, double duration, int steps)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  ProductPoint p = {points[2 * index], points[2 * index + 1]};
  double h = duration / steps;
  for (int k = 0; k < steps; ++k)
  {
    ProductPoint a = product_velocity(p, terms, rank);
    ProductPoint b =
      product_velocity({p.x + h * a.x / 2, p.y + h * a.y / 2}, terms, rank);
    ProductPoint c =
      product_velocity({p.x + h * b.x / 2, p.y + h * b.y / 2}, terms, rank);
    ProductPoint d = product_velocity({p.x + h * c.x, p.y + h * c.y}, terms, rank);
    p.x += h * (a.x + 2 * b.x + 2 * c.x + d.x) / 6;
    p.y += h * (a.y + 2 * b.y + 2 * c.y + d.y) / 6;
  }
  result[2 * index] = p.x;
  result[2 * index + 1] = p.y;
}
