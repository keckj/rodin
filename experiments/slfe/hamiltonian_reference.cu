/* Distributed under the Boost Software License, Version 1.0.
 * Independent RK4 characteristics for a supplied Fourier streamfunction.
 */
struct HamiltonianPoint
{
  double x, y;
};

__device__ HamiltonianPoint hamiltonian_velocity(
  HamiltonianPoint p, const double* modes, int count)
{
  HamiltonianPoint value = {0., 0.};
  for (int j = 0; j < count; ++j)
  {
    const double* mode = modes + 4 * j;
    double sine, cosine;
    sincos(6.28318530717958647693 * (mode[0] * p.x + mode[1] * p.y), &sine, &cosine);
    double derivative = 6.28318530717958647693 * (-mode[2] * sine + mode[3] * cosine);
    value.x += mode[1] * derivative;
    value.y -= mode[0] * derivative;
  }
  return value;
}

extern "C" __global__ void hamiltonian_reference(const double* points, double* result,
  int count, const double* modes, int modes_count, double duration, int steps)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  HamiltonianPoint x = {points[2 * index], points[2 * index + 1]};
  double h = duration / steps;
  for (int step = 0; step < steps; ++step)
  {
    HamiltonianPoint a = hamiltonian_velocity(x, modes, modes_count);
    HamiltonianPoint b =
      hamiltonian_velocity({x.x + h * a.x / 2, x.y + h * a.y / 2}, modes, modes_count);
    HamiltonianPoint c =
      hamiltonian_velocity({x.x + h * b.x / 2, x.y + h * b.y / 2}, modes, modes_count);
    HamiltonianPoint d =
      hamiltonian_velocity({x.x + h * c.x, x.y + h * c.y}, modes, modes_count);
    x.x += h / 6 * (a.x + 2 * b.x + 2 * c.x + d.x);
    x.y += h / 6 * (a.y + 2 * b.y + 2 * c.y + d.y);
  }
  result[2 * index] = x.x;
  result[2 * index + 1] = x.y;
}
