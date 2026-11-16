/* Distributed under the Boost Software License, Version 1.0.
 * Integrate modal data along a straight characteristic, splitting at faces.
 * Reuses Point, inverse, target_triangle and basis from the CUDA prototype.
 */
extern "C" __global__ void integrate_rays(const double* triangles, int n,
  const double* origins, const double* displacement, int rays, int slots,
  const double* nodes, const double* weights, int nq, int* owners, double* moments,
  int* errors)
{
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= rays)
    return;
  for (int slot = 0; slot < slots; ++slot)
    owners[ray * slots + slot] = -1;
  Point origin = {origins[2 * ray], origins[2 * ray + 1]};
  Point delta = {displacement[2 * ray], displacement[2 * ray + 1]};
  double size = n * fmax(fabs(delta.x), fabs(delta.y));
  if (size == 0.)
    return;
  Point finish = {origin.x + delta.x, origin.y + delta.y};
  int xmin = (int)floor(fmin(origin.x, finish.x) * n) - 1;
  int xmax = (int)floor(fmax(origin.x, finish.x) * n) + 1;
  int ymin = (int)floor(fmin(origin.y, finish.y) * n) - 1;
  int ymax = (int)floor(fmax(origin.y, finish.y) * n) + 1;
  int slot = 0;
  double coverage = 0.;
  // Independent line-triangle intervals avoid epsilon-based face stepping at
  // near-vertex crossings. For a conforming mesh the intervals partition [0,1].
  for (int iy = ymin; iy <= ymax; ++iy)
  {
    for (int ix = xmin; ix <= xmax; ++ix)
    {
      for (int half = 0; half < 2; ++half)
      {
        double cell[6];
        int owner;
        target_triangle(triangles, n, ix, iy, half, cell, &owner);
        Point start = inverse(cell, origin);
        Point e = {cell[2] - cell[0], cell[3] - cell[1]};
        Point f = {cell[4] - cell[0], cell[5] - cell[1]};
        double determinant = cross(e, f);
        Point d = {cross(delta, f) / determinant, cross(e, delta) / determinant};
        double begin = 0., end = 1.;
        for (int k = 0; k < 3; ++k)
        {
          int j = (k + 1) % 3;
          Point a = {cell[2 * k], cell[2 * k + 1]}, b = {cell[2 * j], cell[2 * j + 1]};
          double sign = 1.;
          // Canonical edge arithmetic gives both neighbours the same crossing
          // time, rather than two independently rounded barycentric quotients.
          if (a.x > b.x || (a.x == b.x && a.y > b.y))
          {
            Point temp = a;
            a = b;
            b = temp;
            sign = -1.;
          }
          Point edge = sub(b, a);
          double lambda = sign * cross(edge, sub(origin, a));
          double dlambda = sign * cross(edge, delta);
          if (dlambda > 0.)
            begin = fmax(begin, -lambda / dlambda);
          else if (dlambda < 0.)
            end = fmin(end, -lambda / dlambda);
          else if (lambda < 0. || (lambda == 0. && sign < 0.))
            end = -1.;
        }
        if (!(end > begin))
          continue;
        if (slot >= slots)
        {
          atomicAdd(errors + 2, 1);
          return;
        }
        owners[ray * slots + slot] = owner;
        double integral[NB];
        for (int k = 0; k < NB; ++k)
          integral[k] = 0.;
        for (int q = 0; q < nq; ++q)
        {
          double t = begin + (end - begin) * nodes[q];
          double values[NB];
          basis(start.x + t * d.x, start.y + t * d.y, values);
          for (int k = 0; k < NB; ++k)
            integral[k] += (end - begin) * weights[q] * values[k];
        }
        for (int k = 0; k < NB; ++k)
          moments[(ray * slots + slot) * NB + k] = integral[k];
        coverage += end - begin;
        ++slot;
      }
    }
  }
  // Flux error scales with characteristic length. Nearly stagnant rays can
  // have ill-conditioned relative interval lengths but negligible flux error.
  if (fabs(coverage - 1.) * size > 1.e-10)
    atomicAdd(errors, 1);
}
