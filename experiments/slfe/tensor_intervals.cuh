/* Distributed under the Boost Software License, Version 1.0.
 * Periodic interval queries on a nonuniform tensor partition of period one.
 */
__device__ int tensor_interval(double x, const double* edges, int n)
{
  int period = (int)floor(x - edges[0]);
  double local = x - period;
  int lower = 0, upper = n;
  while (upper - lower > 1)
  {
    int middle = (lower + upper) / 2;
    if (edges[middle] <= local)
      lower = middle;
    else
      upper = middle;
  }
  return period * n + lower;
}

__device__ double tensor_edge(int index, const double* edges, int n)
{
  int local = modulo(index, n);
  return edges[local] + (index - local) / n;
}
