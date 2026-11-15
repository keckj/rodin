"""Numerical property checks on the actual CUDA implementation.

Distributed under the Boost Software License, Version 1.0.
"""
import argparse
import json
from math import factorial
from pathlib import Path
import time

from gpu import cp, Mesh, Space, Transfer, quadrature, translation, smooth, constant


def run():
    results = []
    # Quadrature moments use an independent exact factorial formula.
    for degree in [1, 2, 4, 6, 8, 12, 18]:
        q = quadrature(degree)
        worst = 0.
        for a in range(degree+1):
            for b in range(degree+1-a):
                exact = factorial(a)*factorial(b)/factorial(a+b+2)
                actual = float(cp.sum(q[:, 2]*q[:, 0]**a*q[:, 1]**b))
                worst = max(worst, abs(actual-exact))
        assert worst < 2e-14, (degree, worst)
        results.append({"check": "quadrature_moments", "degree": degree, "error": worst})
    for degree in range(4):
        mesh = Mesh(8)
        space = Space(mesh, degree)
        q = quadrature(max(2*degree, 1))
        b = space.basis(q[:, :2])
        gram_error = float(cp.max(cp.abs(b.T@(2*q[:, 2, None]*b)-cp.eye(space.nb))))
        assert gram_error < 2e-12, (degree, gram_error)
        random = cp.random.RandomState(91).normal(size=(mesh.cells, 2, space.nb))
        identity = Transfer(space, mesh.triangles)
        identity_error = float(cp.max(cp.abs(identity.apply(random)-random)))
        assert identity_error < 2e-11, (degree, identity_error)
        for dt in [0.037, 0.5, 1.3]:
            transfer = Transfer(space, mesh.mapped(translation, 0, dt))
            one = cp.zeros((mesh.cells, 1, space.nb), dtype=cp.float64)
            one[:, :, 0] = 1.
            constant_error = float(cp.max(cp.abs(transfer.apply(one)-one)))
            advected = transfer.apply(random)
            mass_error = float(cp.max(cp.abs(space.mass(advected)-space.mass(random))))
            energy_ratio = float(cp.max(space.energy(advected)/space.energy(random)))
            # Discrete adjoint identity, independent random trial/test vectors.
            other = cp.random.RandomState(92).normal(size=random.shape)
            adjoint_error = abs(float(cp.sum(other*advected)-cp.sum(transfer.apply(other, transpose=True)*random)))
            assert constant_error < 2e-11, (degree, dt, constant_error)
            assert mass_error < 2e-12, (degree, dt, mass_error)
            assert energy_ratio <= 1+2e-11, (degree, dt, energy_ratio)
            assert adjoint_error < 2e-10, adjoint_error
            results.append({"check": "geometric_invariants", "degree": degree, "dt": dt,
                            "gram_error": gram_error, "identity_error": identity_error,
                            "constant_error": constant_error, "mass_error": mass_error,
                            "random_energy_ratio": energy_ratio, "adjoint_error": adjoint_error})
        # A second integration degree must recover the same polynomial transfer.
        moved = mesh.mapped(translation, 0, .037)
        t1 = Transfer(space, moved)
        t2 = Transfer(space, moved, qdegree=12)
        independent_quad_error = float(cp.max(cp.abs(t1.apply(random)-t2.apply(random))))
        assert independent_quad_error < 2e-11, independent_quad_error
        results.append({"check": "independent_overlap_quadrature", "degree": degree,
                        "difference": independent_quad_error})
        # Source-point deposition conserves total mass, including deterministic
        # ownership at boundary-aligned configurations. It need not keep constants.
        for dt in [0., .037, 1.]:
            raw = Transfer(space, mesh.mapped(translation, 0, dt), mode="quadrature", qdegree=12)
            y = raw.deposit(random)
            error = float(cp.max(cp.abs(space.mass(y)-space.mass(random))))
            assert error < 2e-12, (degree, dt, error)
            if dt == .037:
                difference = float(cp.max(cp.abs(y-raw.apply(random))))
                assert difference < 2e-11, difference
            results.append({"check": "particle_mass", "degree": degree, "dt": dt, "error": error})
        # Mutation: losing one percent of deposited mass must fail the same gate.
        mutated = .99*t1.apply(one)
        assert float(cp.max(cp.abs(space.mass(mutated)-space.mass(one)))) > 1e-3
    # Reject public invalid data and geometry instead of silently falling back.
    for n, jitter in [(3, .1), (8, .25)]:
        try:
            Mesh(n, jitter)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid mesh accepted")
    mesh = Mesh(8)
    space = Space(mesh, 1)
    inverted = mesh.triangles[:, [0, 2, 1]].copy()
    try:
        Transfer(space, inverted)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Inverted characteristic cells accepted")
    # Refinement oracle: translation with a fixed number of remaps, smooth data.
    for degree in [1, 2, 3]:
        errors = []
        for n in [8, 16, 32]:
            mesh = Mesh(n)
            space = Space(mesh, degree)
            c = space.project(smooth)
            transfer = Transfer(space, mesh.mapped(translation, 0, .073))
            for _ in range(4):
                c = transfer.apply(c)
            errors.append(space.errors(c, smooth, 4*.073)["l2"][0])
        rates = [float(cp.log2(cp.asarray(errors[k]/errors[k+1]))) for k in range(2)]
        assert min(rates) > degree+.45, (degree, errors, rates)
        results.append({"check": "translation_convergence", "degree": degree,
                        "n": [8, 16, 32], "l2": errors, "rates": rates})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    checks = run()
    record = {"passed": True, "seconds": time.perf_counter()-start,
              "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "checks": checks}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))
