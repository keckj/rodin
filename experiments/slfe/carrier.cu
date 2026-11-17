/* Distributed under the Boost Software License, Version 1.0.
 * Static exact cross-mass between affine DG Pp triangles and Cartesian Qp.
 * Reuses the reference-triangle clipping helpers of transfer.cu.
 */
__device__ void nodal_values(double x, const double* nodes, double* values)
{
  for (int i = 0; i < NP; ++i)
  {
    values[i] = 1.;
    for (int j = 0; j < NP; ++j)
      if (j != i)
        values[i] *= (x - nodes[j]) / (nodes[i] - nodes[j]);
  }
}

extern "C" __global__ void build_shear_blocks(const double* velocity,
  const double* duration, int n, int rows, int degree, const double* polynomials,
  int* shifts, double* blocks)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int entries = 2 * NP * NP;
  if (item >= rows * entries)
    return;
  int row = item / entries, entry = item % entries;
  double displacement = n * duration[0] * velocity[row];
  double integer = floor(displacement), fraction = displacement - integer;
  if (entry == 0)
    shifts[row] = (int)integer;
  bool reverse = fraction > .5;
  double z = reverse ? 1. - fraction : fraction;
  double value = 0., bernstein = 1.;
  for (int k = 0; k < degree; ++k)
    bernstein *= 1. - z;
  double ratio = z / (1. - z);
  for (int k = 0; k <= degree; ++k)
  {
    int coefficient = reverse ? degree - k : k;
    value += polynomials[entry * (degree + 1) + coefficient] * bernstein;
    bernstein *= double(degree - k) / (k + 1) * ratio;
  }
  blocks[item] = value;
}

#ifndef CARRIER_WEIGHT
#define CARRIER_WEIGHT(position) 1.0
#endif

extern "C" __global__ void carrier_blocks(const double* triangles, int n,
  const double* edges, int first, int cells, int slots, const double* quadrature, int nq,
  const double* nodes, const double* weights, int* destinations, double* blocks,
  int* errors)
{
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= cells * slots)
    return;
  int source = first + pair / slots, candidate = pair % slots;
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
    if (candidate == 0)
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
  double matrix[NC * NB];
  for (int k = 0; k < NC * NB; ++k)
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
    Point poly[12];
    int count = intersection(t, rectangle_half, poly);
    for (int fan = 1; fan < count - 1; ++fan)
    {
      Point e = sub(poly[fan], poly[0]), f = sub(poly[fan + 1], poly[0]);
      double det = cross(e, f);
      if (!(det > 0.))
        continue;
      covered += det;
      for (int q = 0; q < nq; ++q)
      {
        Point a = {poly[0].x + quadrature[3 * q] * e.x + quadrature[3 * q + 1] * f.x,
          poly[0].y + quadrature[3 * q] * e.y + quadrature[3 * q + 1] * f.y};
        Point position = map(t, a.x, a.y);
        double sb[NB], bx[NP], by[NP];
        basis(a.x, a.y, sb);
        nodal_values(NONUNIFORM ? (position.x - x) / hx : n * position.x - ix, nodes, bx);
        nodal_values(NONUNIFORM ? (position.y - y) / hy : n * position.y - iy, nodes, by);
        double w = determinant(t) * det * quadrature[3 * q + 2] *
          (NONUNIFORM ? 1. / (hx * hy) : double(n * n));
        w *= CARRIER_WEIGHT(position);
        for (int j = 0; j < NP; ++j)
        {
          for (int i = 0; i < NP; ++i)
          {
            for (int k = 0; k < NB; ++k)
            {
              matrix[(j * NP + i) * NB + k] +=
                w * bx[i] * by[j] * sb[k] / (weights[i] * weights[j]);
            }
          }
        }
      }
    }
  }
  if (!(covered > 0.))
    return;
  destinations[pair] = modulo(iy, n) * n + modulo(ix, n);
  for (int k = 0; k < NC * NB; ++k)
    blocks[pair * NC * NB + k] = matrix[k];
}

extern "C" __global__ void carrier_blocks_warp(const double* triangles, int n,
  const double* edges, int first, int cells, int slots, const double* quadrature, int nq,
  const double* nodes, const double* weights, int* destinations, double* blocks,
  int* errors)
{
  // Four warps cooperate independently; each lane owns disjoint matrix entries.
  constexpr int lanes = 32, entries = (NC * NB + lanes - 1) / lanes;
  const int lane = threadIdx.x % lanes, warp = threadIdx.x / lanes;
  int pair = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
  __shared__ Point polygons[4][12];
  __shared__ int counts[4];
  __shared__ double source_basis[4][NB], basis_x[4][NP], basis_y[4][NP];
  __shared__ double quadrature_weight[4];
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
      for (int q = 0; q < nq; ++q)
      {
        if (lane == 0)
        {
          Point a = {poly[0].x + quadrature[3 * q] * e.x + quadrature[3 * q + 1] * f.x,
            poly[0].y + quadrature[3 * q] * e.y + quadrature[3 * q + 1] * f.y};
          Point position = map(t, a.x, a.y);
          basis(a.x, a.y, source_basis[warp]);
          nodal_values(NONUNIFORM ? (position.x - x) / hx : n * position.x - ix, nodes,
            basis_x[warp]);
          nodal_values(NONUNIFORM ? (position.y - y) / hy : n * position.y - iy, nodes,
            basis_y[warp]);
          quadrature_weight[warp] = determinant(t) * det * quadrature[3 * q + 2] *
            (NONUNIFORM ? 1. / (hx * hy) : double(n * n)) * CARRIER_WEIGHT(position);
        }
        __syncwarp();
        const double w = quadrature_weight[warp];
        for (int item = 0; item < entries; ++item)
        {
          const int index = item * lanes + lane;
          if (index < NC * NB)
          {
            const int k = index % NB, i = (index / NB) % NP, j = index / (NB * NP);
            matrix[item] += w * basis_x[warp][i] * basis_y[warp][j] *
              source_basis[warp][k] / (weights[i] * weights[j]);
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
      blocks[pair * NC * NB + index] = matrix[item];
  }
}

extern "C" __global__ void project_carrier(const int* sources, const int* targets,
  const double* blocks, int pairs, const double* input, double* output, int fields)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= pairs * fields * NC)
    return;
  int i = item % NC, field = (item / NC) % fields, pair = item / (NC * fields);
  double value = 0.;
  for (int j = 0; j < NB; ++j)
  {
    value +=
      blocks[(pair * NC + i) * NB + j] * input[(sources[pair] * fields + field) * NB + j];
  }
  atomicAdd(output + (targets[pair] * fields + field) * NC + i, value);
}

extern "C" __global__ void adjoint_carrier(const int* sources, const int* targets,
  const double* blocks, const double* source_areas, const double* target_mass, int pairs,
  const double* input, double* output, int fields)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= pairs * fields * NB)
    return;
  int j = item % NB, field = (item / NB) % fields, pair = item / (NB * fields);
  double value = 0.;
  for (int i = 0; i < NC; ++i)
  {
    value += blocks[(pair * NC + i) * NB + j] *
      target_mass[NONUNIFORM ? targets[pair] * NC + i : i] *
      input[(targets[pair] * fields + field) * NC + i];
  }
  atomicAdd(output + (sources[pair] * fields + field) * NB + j,
    value / source_areas[sources[pair]]);
}

extern "C" __global__ void shift_carrier(const double* input, double* output, int n,
  int fields, const int* shifts, const double* blocks, int axis)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= n * n * fields * NC)
    return;
  int local = item % NC, field = (item / NC) % fields, cell = item / (NC * fields);
  int x = cell % n, y = cell / n, lx = local % NP, ly = local / NP;
  int row = axis == 0 ? y * NP + ly : x * NP + lx;
  int i = axis == 0 ? lx : ly;
  double value = 0.;
  for (int side = 0; side < 2; ++side)
  {
    for (int j = 0; j < NP; ++j)
    {
      int offset = shifts[row] + (1 - side);
      int sx = axis == 0 ? modulo(x - offset, n) : x;
      int sy = axis == 1 ? modulo(y - offset, n) : y;
      int node = axis == 0 ? ly * NP + j : j * NP + lx;
      value += blocks[((row * 2 + side) * NP + i) * NP + j] *
        input[((sy * n + sx) * fields + field) * NC + node];
    }
  }
  output[item] = value;
}
