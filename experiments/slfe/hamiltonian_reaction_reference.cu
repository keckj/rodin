/* Distributed under the Boost Software License, Version 1.0.
 * Independent backward RK4 for Fourier-velocity transport and a spatial sink.
 * Appended to hamiltonian_reference.cu for the analytic velocity evaluator.
 */
__device__ double hamiltonian_reaction_rate(HamiltonianPoint p)
{
  return 1. + .5 * sin(6.28318530717958647693 * p.x);
}

extern "C" __global__ void hamiltonian_reaction_reference(const double* points,
  double* result, int count, const double* modes, int modes_count, double end, int steps)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  HamiltonianPoint x = {points[2 * index], points[2 * index + 1]};
  double h = -end / steps;
  double exponent = 0.;
  for (int k = 0; k < steps; ++k)
  {
    HamiltonianPoint a = hamiltonian_velocity(x, modes, modes_count);
    HamiltonianPoint xb = {x.x + h * a.x / 2, x.y + h * a.y / 2};
    HamiltonianPoint b = hamiltonian_velocity(xb, modes, modes_count);
    HamiltonianPoint xc = {x.x + h * b.x / 2, x.y + h * b.y / 2};
    HamiltonianPoint c = hamiltonian_velocity(xc, modes, modes_count);
    HamiltonianPoint xd = {x.x + h * c.x, x.y + h * c.y};
    HamiltonianPoint d = hamiltonian_velocity(xd, modes, modes_count);
    exponent += h / 6 *
      (hamiltonian_reaction_rate(x) + 2 * hamiltonian_reaction_rate(xb) +
        2 * hamiltonian_reaction_rate(xc) + hamiltonian_reaction_rate(xd));
    x.x += h / 6 * (a.x + 2 * b.x + 2 * c.x + d.x);
    x.y += h / 6 * (a.y + 2 * b.y + 2 * c.y + d.y);
  }
  result[3 * index] = x.x;
  result[3 * index + 1] = x.y;
  result[3 * index + 2] = exponent;
}
