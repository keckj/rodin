/* Distributed under the Boost Software License, Version 1.0.
 * Positive quadrature of the difference between FE and auxiliary functions.
 * Geometry primitives and polynomial evaluations are shared with Carrier.
 */
extern "C" __global__ void residual_energy(const double* triangles, int n,
  const double* edges, int cells, int slots, const double* quadrature, int nq,
  const double* nodes, const double* fe, const double* auxiliary, int fields,
  double* output, int* errors)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= cells * slots * fields)
    return;
  output[item] = 0.;
  int field = item % fields, pair = item / fields;
  int source = pair / slots, candidate = pair % slots;
  const double* t = triangles + 6 * source;
  int xmin = (int)floor(fmin(t[0], fmin(t[2], t[4])) * n) - 1;
  int xmax = (int)floor(fmax(t[0], fmax(t[2], t[4])) * n) + 1;
  int ymin = (int)floor(fmin(t[1], fmin(t[3], t[5])) * n) - 1;
  int ymax = (int)floor(fmax(t[1], fmax(t[3], t[5])) * n) + 1;
  if constexpr (NONUNIFORM)
  {
    xmin = tensor_interval(fmin(t[0], fmin(t[2], t[4])), edges, n) - 1;
    xmax = tensor_interval(fmax(t[0], fmax(t[2], t[4])), edges, n) + 1;
    ymin = tensor_interval(fmin(t[1], fmin(t[3], t[5])), edges + n + 1, n) - 1;
    ymax = tensor_interval(fmax(t[1], fmax(t[3], t[5])), edges + n + 1, n) + 1;
  }
  int width = xmax - xmin + 1, height = ymax - ymin + 1;
  if (width * height > slots)
  {
    if (candidate == 0 && field == 0)
      atomicAdd(errors, 1);
    return;
  }
  if (candidate >= width * height)
    return;
  int ix = xmin + candidate % width, iy = ymin + candidate / width;
  double x = double(ix) / n, y = double(iy) / n, hx = 1. / n, hy = hx;
  if constexpr (NONUNIFORM)
  {
    x = tensor_edge(ix, edges, n);
    y = tensor_edge(iy, edges + n + 1, n);
    hx = tensor_edge(ix + 1, edges, n) - x;
    hy = tensor_edge(iy + 1, edges + n + 1, n) - y;
  }
  int target = modulo(iy, n) * n + modulo(ix, n);
  double total = 0.;
  for (int half = 0; half < 2; ++half)
  {
    double rectangle[6] = {x, y, x + hx, y, x + hx, y + hy};
    if (half == 1)
    {
      rectangle[2] = x + hx;
      rectangle[3] = y + hy;
      rectangle[4] = x;
      rectangle[5] = y + hy;
    }
    Point poly[12];
    int count = intersection(t, rectangle, poly);
    for (int fan = 1; fan < count - 1; ++fan)
    {
      Point e = sub(poly[fan], poly[0]), f = sub(poly[fan + 1], poly[0]);
      double det = cross(e, f);
      if (!(det > 0.))
        continue;
      for (int q = 0; q < nq; ++q)
      {
        Point a = {poly[0].x + quadrature[3 * q] * e.x + quadrature[3 * q + 1] * f.x,
          poly[0].y + quadrature[3 * q] * e.y + quadrature[3 * q + 1] * f.y};
        Point position = map(t, a.x, a.y);
        double b[NB], bx[NP], by[NP];
        basis(a.x, a.y, b);
        nodal_values((position.x - x) / hx, nodes, bx);
        nodal_values((position.y - y) / hy, nodes, by);
        double value = 0., other = 0.;
        for (int k = 0; k < NB; ++k)
          value += fe[(source * fields + field) * NB + k] * b[k];
        for (int j = 0; j < NP; ++j)
          for (int i = 0; i < NP; ++i)
            other +=
              auxiliary[(target * fields + field) * NC + j * NP + i] * bx[i] * by[j];
        double residual = value - other;
        total += determinant(t) * det * quadrature[3 * q + 2] * residual * residual;
      }
    }
  }
  output[item] = total;
}
