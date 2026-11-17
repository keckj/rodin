"""GPU Fourier truncation study for a non-bandlimited FE streamfunction.

Distributed under the Boost Software License, Version 1.0.
Independent analytic-velocity RK4 characteristics separate truncation error
from transport error. FFT moments of the analytic function independently
control the FE moment assembly. No CPU numerical path is used.
"""

import argparse
import json
from pathlib import Path
import time

from gpu import cp, Mesh, Space, launch
from carrier import Carrier
from tensor_hamiltonian import ComposedHamiltonian, streamfunction_modes
from hamiltonian_study import profile, render_fields
from memory_transport import MemoryTransport
from krylov_study import ErrorOracle
from campaign import provenance

REFERENCE = r"""
struct Point { double x, y; };
__device__ Point velocity(Point p) {
    const double pi2=6.28318530717958647693;
    double sx,cx,sy,cy;
    sincos(pi2*(p.x-.5), &sx, &cx);
    sincos(pi2*(p.y-.5), &sy, &cy);
    double psi=.04*exp(.7*cx+.45*cy-1.15);
    return {-pi2*.45*sy*psi, pi2*.7*sx*psi};
}
extern "C" __global__ void broad_reference(const double* p, double* out,
    int count, double duration, int steps) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    Point x={p[2*i],p[2*i+1]};
    double h=duration/steps;
    for(int j=0;j<steps;++j) {
        Point a=velocity(x);
        Point b=velocity({x.x+h*a.x/2,x.y+h*a.y/2});
        Point c=velocity({x.x+h*b.x/2,x.y+h*b.y/2});
        Point d=velocity({x.x+h*c.x,x.y+h*c.y});
        x.x+=h/6*(a.x+2*b.x+2*c.x+d.x);
        x.y+=h/6*(a.y+2*b.y+2*c.y+d.y);
    }
    out[2*i]=x.x; out[2*i+1]=x.y;
}
"""


def psi(points, time=0):
    angle = 2 * cp.pi * (points - 0.5)
    return (
        0.04 * cp.exp(0.7 * cp.cos(angle[..., 0]) + 0.45 * cp.cos(angle[..., 1]) - 1.15)
    )[..., None]


def analytic_velocity(points):
    angle = 2 * cp.pi * (points - 0.5)
    return (
        2
        * cp.pi
        * psi(points)
        * cp.stack(
            (-0.45 * cp.sin(angle[..., 1]), 0.7 * cp.sin(angle[..., 0])), axis=-1
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--bands", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--fields", action="store_true")
    args = parser.parse_args()
    meta = provenance()
    duration = 0.5
    kernel = cp.RawKernel(REFERENCE, "broad_reference")
    kernel.compile()

    def feet(points, steps=512):
        flat = cp.ascontiguousarray(points.reshape(-1, 2))
        result = cp.empty_like(flat)
        launch(kernel, len(flat), (flat, result, len(flat), -duration, steps))
        return result.reshape(points.shape)

    def exact(points):
        return profile(feet(points), 0)

    probe = cp.random.RandomState(541).uniform(size=(8192, 2))
    ref_error = float(cp.max(cp.abs(feet(probe) - feet(probe, 1024))))
    # Autonomous Hamiltonian characteristics must preserve this streamfunction.
    invariant_error = float(cp.max(cp.abs(psi(feet(probe)) - psi(probe))))
    assert max(ref_error, invariant_error) < 1e-11
    exact_velocity = analytic_velocity(probe)
    grid = cp.arange(256, dtype=cp.float64) / 256
    gx, gy = cp.meshgrid(grid, grid, indexing="ij")
    fft = cp.fft.fft2(psi(cp.stack((gx, gy), axis=-1))[..., 0]) / 256**2
    with args.output.open("w") as output:
        for n in args.sizes:
            space = Space(Mesh(n, jitter=0.18, seed=421), 4)
            initial = space.project(profile, degree=20)
            projected_psi = space.project(psi, degree=20)
            oracle = ErrorOracle(space, profile, duration, reference=exact)
            norm = cp.sqrt(
                cp.sum(oracle.reference**2 * oracle.weights[..., None], axis=(0, 1))
            )
            floor = (cp.asarray(oracle.errors(oracle.best)["l2"]) / norm).tolist()
            carrier = Carrier(space, n=2 * n, builder="tiled")
            for band in args.bands:
                vectors = [
                    (k, l)
                    for k in range(band + 1)
                    for l in range(-band, band + 1)
                    if k > 0 or l > 0
                ]
                waves = cp.asarray(vectors, dtype=cp.float64)
                begin = time.perf_counter()
                coefficients = streamfunction_modes(space, projected_psi, vectors)
                cp.cuda.Stream.null.synchronize()
                moments_time = time.perf_counter() - begin
                exact_coefficients = cp.stack(
                    [
                        cp.stack((2 * fft[k, l].real, -2 * fft[k, l].imag))
                        for k, l in vectors
                    ]
                )
                coefficient_error = float(
                    cp.linalg.norm(coefficients - exact_coefficients)
                    / cp.linalg.norm(exact_coefficients)
                )
                angle = 2 * cp.pi * cp.einsum("...i,mi->...m", probe, waves)
                derivative = (
                    2
                    * cp.pi
                    * (
                        -cp.sin(angle) * coefficients[:, 0]
                        + cp.cos(angle) * coefficients[:, 1]
                    )
                )
                filtered = cp.stack(
                    (derivative @ waves[:, 1], -derivative @ waves[:, 0]), axis=-1
                )
                velocity_error = float(
                    cp.linalg.norm(filtered - exact_velocity)
                    / cp.linalg.norm(exact_velocity)
                )
                for steps in args.steps:
                    begin = time.perf_counter()
                    flow = ComposedHamiltonian(
                        carrier, vectors, coefficients, duration / steps, "yoshida6"
                    )
                    flow.apply_carrier(carrier.project(initial))
                    cp.cuda.Stream.null.synchronize()
                    setup = time.perf_counter() - begin
                    fields = {}
                    for method in ("memory", "reset"):
                        state = MemoryTransport(carrier, initial)
                        cp.cuda.Stream.null.synchronize()
                        begin = time.perf_counter()
                        for _ in range(steps):
                            state.advect(flow.apply_carrier)
                            if method == "reset":
                                state.restart(carrier)
                        cp.cuda.Stream.null.synchronize()
                        elapsed = time.perf_counter() - begin
                        errors = oracle.errors(state.value)
                        mass_error = float(
                            cp.max(
                                cp.abs(space.mass(state.value) - space.mass(initial))
                            )
                        )
                        energy_ratio = float(
                            cp.max(state.energy() / space.energy(initial))
                        )
                        assert mass_error < 1e-9 and energy_ratio <= 1 + 1e-9
                        result = dict(
                            nfe=n,
                            naux=2 * n,
                            degree=4,
                            end=duration,
                            band=band,
                            modes=len(vectors),
                            steps=steps,
                            method=method,
                            scheme="yoshida6",
                            relative_l2=(cp.asarray(errors["l2"]) / norm).tolist(),
                            relative_fe_floor=floor,
                            fourier_moment_relative_error=coefficient_error,
                            velocity_relative_probe_error=velocity_error,
                            reference_refinement_error=ref_error,
                            reference_invariant_error=invariant_error,
                            mass_error=mass_error,
                            energy_ratio=energy_ratio,
                            moments_seconds=moments_time,
                            flow_setup_seconds=setup,
                            evolution_seconds=elapsed,
                            **errors,
                        )
                        output.write(
                            json.dumps(dict(result=result, provenance=meta)) + "\n"
                        )
                        output.flush()
                        print(json.dumps(result), flush=True)
                        if (
                            args.fields
                            and band == max(args.bands)
                            and steps == max(args.steps)
                        ):
                            fields[method] = state.value.copy()
                    del flow
                    cp.get_default_memory_pool().free_all_blocks()
                    if fields:
                        render_fields(
                            args.output.parent / "figures",
                            space,
                            exact,
                            fields,
                            n,
                            steps,
                            f"broad{band}-yoshida6",
                        )


if __name__ == "__main__":
    main()
