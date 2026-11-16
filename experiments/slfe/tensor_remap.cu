/* Distributed under the Boost Software License, Version 1.0.
 * Exact one-dimensional DG remapping on each transverse Gauss row.
 * Every overlap integrates degree 2*(NP-1) with NP Gauss points.
 */
extern "C" __global__ void tensor_counts(const double* edges, const double* source_edges,
  int n, const double* velocity, const double* duration, int* starts, long long* counts)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n * n * NP)
    return;
  int target = row % n, transverse = row / n;
  double shift = duration[0] * velocity[transverse];
  int first = tensor_interval(edges[target] - shift, source_edges, n);
  int last = tensor_interval(edges[target + 1] - shift, source_edges, n);
  starts[row] = first;
  counts[row] = last - first + 1;
}

extern "C" __global__ void tensor_blocks(const double* edges, const double* source_edges,
  int n, const double* velocity, const double* duration, const int* starts,
  const long long* offsets, const int* rows, long long pairs, const double* nodes,
  const double* weights, double* blocks)
{
  long long item = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= pairs * NP * NP)
    return;
  long long pair = item / (NP * NP);
  int i = (item / NP) % NP, j = item % NP, row = rows[pair];
  int target = row % n, transverse = row / n;
  int source = starts[row] + (int)(pair - offsets[row]);
  double shift = duration[0] * velocity[transverse];
  double t0 = edges[target], ht = edges[target + 1] - t0;
  double s0 = tensor_edge(source, source_edges, n);
  double hs = tensor_edge(source + 1, source_edges, n) - s0;
  double lower = fmax(t0, s0 + shift), upper = fmin(t0 + ht, s0 + hs + shift);
  double value = 0.;
  if (upper > lower)
  {
    for (int q = 0; q < NP; ++q)
    {
      double x = lower + (upper - lower) * nodes[q];
      double tx = (x - t0) / ht, sx = (x - shift - s0) / hs;
      double bt = 1., bs = 1.;
      for (int k = 0; k < NP; ++k)
      {
        if (k != i)
          bt *= (tx - nodes[k]) / (nodes[i] - nodes[k]);
        if (k != j)
          bs *= (sx - nodes[k]) / (nodes[j] - nodes[k]);
      }
      value += weights[q] * bt * bs;
    }
    value *= (upper - lower) / (ht * weights[i]);
  }
  blocks[item] = value;
}

extern "C" __global__ void tensor_apply(const double* input, double* output, int n,
  int fields, int axis, const int* starts, const long long* offsets, const double* blocks)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= n * n * fields * NC)
    return;
  int local = item % NC, field = (item / NC) % fields, cell = item / (NC * fields);
  int x = cell % n, y = cell / n, lx = local % NP, ly = local / NP;
  int row = axis == 0 ? (y * NP + ly) * n + x : (x * NP + lx) * n + y;
  int i = axis == 0 ? lx : ly;
  double value = 0.;
  for (long long pair = offsets[row]; pair < offsets[row + 1]; ++pair)
  {
    int source = modulo(starts[row] + (int)(pair - offsets[row]), n);
    int source_cell = axis == 0 ? y * n + source : source * n + x;
    for (int j = 0; j < NP; ++j)
    {
      int node = axis == 0 ? ly * NP + j : j * NP + lx;
      value += blocks[(pair * NP + i) * NP + j] *
        input[(source_cell * fields + field) * NC + node];
    }
  }
  output[item] = value;
}
