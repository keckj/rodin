/* Distributed under the Boost Software License, Version 1.0.
 * Common positive quadrature of pointwise physics on the complete state.
 * Four warps stage quadrature values and accumulate FE and auxiliary moments.
 */
extern "C" __global__ void quadrature_moments(const double* triangles, int n,
  const double* edges, int cells, int slots, const double* quadrature, int nq,
  const double* nodes, const double* weights, const double* areas, const double* fe,
  const double* detail, const double* reference, int has_reference, int fields,
  double duration, double amplitude, int reaction, double* observation, double* moment,
  double* energies, int* errors)
{
  constexpr int lanes = 32, entries = (NB + NC + lanes - 1) / lanes;
  int lane = threadIdx.x % lanes, warp = threadIdx.x / lanes;
  int item = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
  if (item >= cells * slots * fields)
    return;
  int field = item % fields, pair = item / fields;
  int source = pair / slots, candidate = pair % slots;
  __shared__ Point polygons[4][12];
  __shared__ int counts[4];
  __shared__ double b[4][32][NB], bx[4][32][NP], by[4][32][NP];
  __shared__ double weighted[4][32];
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
    if (candidate == 0 && field == 0 && lane == 0)
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
  double integrals[entries], old_energy = 0., new_energy = 0., defect = 0.;
  for (int k = 0; k < entries; ++k)
    integrals[k] = 0.;
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
    Point* poly = polygons[warp];
    if (lane == 0)
      counts[warp] = intersection(t, rectangle, poly);
    __syncwarp();
    for (int fan = 1; fan < counts[warp] - 1; ++fan)
    {
      Point e = sub(poly[fan], poly[0]), f = sub(poly[fan + 1], poly[0]);
      double det = cross(e, f);
      if (!(det > 0.))
        continue;
      for (int firstq = 0; firstq < nq; firstq += lanes)
      {
        int q = firstq + lane;
        if (q < nq)
        {
          Point a = {poly[0].x + quadrature[3 * q] * e.x + quadrature[3 * q + 1] * f.x,
            poly[0].y + quadrature[3 * q] * e.y + quadrature[3 * q + 1] * f.y};
          Point position = map(t, a.x, a.y);
          basis(a.x, a.y, b[warp][lane]);
          nodal_values((position.x - x) / hx, nodes, bx[warp][lane]);
          nodal_values((position.y - y) / hy, nodes, by[warp][lane]);
          double value = 0.;
          for (int k = 0; k < NB; ++k)
            value += fe[(source * fields + field) * NB + k] * b[warp][lane][k];
          for (int j = 0; j < NP; ++j)
            for (int i = 0; i < NP; ++i)
              value += detail[(target * fields + field) * NC + j * NP + i] *
                bx[warp][lane][i] * by[warp][lane][j];
          double rate = 1. + amplitude * sin(6.28318530717958647693 * position.x);
          double evolved = reaction == 0
            ? value * exp(-duration * rate)
            : value / sqrt(1. + 2. * duration * rate * value * value);
          double weight = determinant(t) * det * quadrature[3 * q + 2];
          weighted[warp][lane] = weight * evolved;
          old_energy += weight * value * value;
          new_energy += weight * evolved * evolved;
          if (has_reference)
          {
            double difference = evolved;
            for (int k = 0; k < NB; ++k)
              difference -=
                reference[(source * fields + field) * NB + k] * b[warp][lane][k];
            defect += weight * difference * difference;
          }
        }
        __syncwarp();
        for (int localq = 0; localq < min(lanes, nq - firstq); ++localq)
        {
          for (int k = 0; k < entries; ++k)
          {
            int index = k * lanes + lane;
            if (index < NB)
              integrals[k] += weighted[warp][localq] * b[warp][localq][index];
            else if (index < NB + NC)
              integrals[k] += weighted[warp][localq] *
                bx[warp][localq][(index - NB) % NP] * by[warp][localq][(index - NB) / NP];
          }
        }
        __syncwarp();
      }
    }
    __syncwarp();
  }
  for (int offset = lanes / 2; offset; offset /= 2)
  {
    old_energy += __shfl_down_sync(0xffffffff, old_energy, offset);
    new_energy += __shfl_down_sync(0xffffffff, new_energy, offset);
    defect += __shfl_down_sync(0xffffffff, defect, offset);
  }
  if (lane == 0)
  {
    energies[3 * item] = old_energy;
    energies[3 * item + 1] = new_energy;
    energies[3 * item + 2] = defect;
  }
  for (int k = 0; k < entries; ++k)
  {
    int index = k * lanes + lane;
    if (index < NB)
      atomicAdd(observation + (source * fields + field) * NB + index,
        integrals[k] / areas[source]);
    else if (index < NB + NC)
      atomicAdd(moment + (target * fields + field) * NC + index - NB,
        integrals[k] /
          (hx * hy * weights[(index - NB) % NP] * weights[(index - NB) / NP]));
  }
}
