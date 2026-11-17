/* Distributed under the Boost Software License, Version 1.0.
 * Parallel quadrature staging for the existing cross-mass integrals.
 */
extern "C" __global__ void carrier_blocks_tiled(const double* triangles, int n,
  const double* edges, int first, int cells, int slots, const double* quadrature, int nq,
  const double* nodes, const double* weights, int* destinations, double* blocks,
  int* errors)
{
  // Each warp stages 32 quadrature nodes in parallel before forming outer products.
  constexpr int lanes = 32, entries = (NC * NB + lanes - 1) / lanes;
  const int lane = threadIdx.x % lanes, warp = threadIdx.x / lanes;
  int pair = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
  __shared__ Point polygons[4][12];
  __shared__ int counts[4];
  __shared__ double source_basis[4][32][NB], basis_x[4][32][NP], basis_y[4][32][NP];
  __shared__ double quadrature_weight[4][32];
  if (pair >= cells * slots)
    return;
  int source = first + pair / slots, candidate = pair % slots;
  if (lane == 0)
    destinations[pair] = -1;
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
    if (candidate == 0 && lane == 0)
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
  double matrix[entries];
  for (int k = 0; k < entries; ++k)
    matrix[k] = 0.;
  double covered = 0.;
  for (int half = 0; half < 2; ++half)
  {
    double rectangle_half[6] = {x, y, x + hx, y, x + hx, y + hy};
    if (half == 1)
    {
      rectangle_half[2] = x + hx;
      rectangle_half[3] = y + hy;
      rectangle_half[4] = x;
      rectangle_half[5] = y + hy;
    }
    Point* poly = polygons[warp];
    if (lane == 0)
      counts[warp] = intersection(t, rectangle_half, poly);
    __syncwarp();
    int count = counts[warp];
    for (int fan = 1; fan < count - 1; ++fan)
    {
      Point e = sub(poly[fan], poly[0]), f = sub(poly[fan + 1], poly[0]);
      double det = cross(e, f);
      if (!(det > 0.))
        continue;
      covered += det;
      for (int firstq = 0; firstq < nq; firstq += lanes)
      {
        const int q = firstq + lane;
        if (q < nq)
        {
          Point a = {poly[0].x + quadrature[3 * q] * e.x + quadrature[3 * q + 1] * f.x,
            poly[0].y + quadrature[3 * q] * e.y + quadrature[3 * q + 1] * f.y};
          Point position = map(t, a.x, a.y);
          basis(a.x, a.y, source_basis[warp][lane]);
          nodal_values(NONUNIFORM ? (position.x - x) / hx : n * position.x - ix, nodes,
            basis_x[warp][lane]);
          nodal_values(NONUNIFORM ? (position.y - y) / hy : n * position.y - iy, nodes,
            basis_y[warp][lane]);
          quadrature_weight[warp][lane] = determinant(t) * det * quadrature[3 * q + 2] *
            (NONUNIFORM ? 1. / (hx * hy) : double(n * n)) * CARRIER_WEIGHT(position);
        }
        __syncwarp();
        const int available = min(lanes, nq - firstq);
        for (int qlocal = 0; qlocal < available; ++qlocal)
        {
          const double w = quadrature_weight[warp][qlocal];
          for (int item = 0; item < entries; ++item)
          {
            const int index = item * lanes + lane;
            if (index < NC * NB)
            {
              const int k = index % NB, i = (index / NB) % NP, j = index / (NB * NP);
              matrix[item] += w * basis_x[warp][qlocal][i] * basis_y[warp][qlocal][j] *
                source_basis[warp][qlocal][k];
            }
          }
        }
        __syncwarp();
      }
    }
    __syncwarp();
  }
  if (!(covered > 0.))
    return;
  if (lane == 0)
    destinations[pair] = modulo(iy, n) * n + modulo(ix, n);
  for (int item = 0; item < entries; ++item)
  {
    const int index = item * lanes + lane;
    if (index < NC * NB)
    {
      const int i = (index / NB) % NP, j = index / (NB * NP);
      blocks[pair * NC * NB + index] = matrix[item] / (weights[i] * weights[j]);
    }
  }
}
