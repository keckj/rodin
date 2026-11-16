"""Unitary CUDA transport for separable-product streamfunctions.

Distributed under the Boost Software License, Version 1.0.
For psi=sum a_r(y)b_r(x), advection is the sum of commutators
[-a_r(y) d_x, -b_r(x) d_y]. Each elementary exponential is a Fourier shear.
The PCP16 coefficients are from Casas, Escorihuela-Tomas and Moreno Casares,
Quantum Information Processing 24, 47 (2025), Table 4, Eq. (22):
https://doi.org/10.1007/s11128-025-04659-z (CC BY 4.0).
Their order five in sqrt(dt) gives order two in physical time at fixed grid.
Neither that formula nor generic Hamiltonian splitting is claimed as new.
"""

from gpu import cp


class ProductHamiltonianFlow:
    @classmethod
    def from_streamfunction(cls, values, velocity_tolerance=1e-6, method="pcp16"):
        """GPU SVD with a tail criterion in the discrete velocity L2 norm.

        Singular vectors are orthogonal, so the squared velocity error of
        each discarded product adds exactly. This selects an SVD prefix,
        not the optimal low-rank approximation in an H1 norm. The criterion
        controls sampled spectral velocity, not an FE interpolation error or
        a pointwise velocity bound. A periodic harmonic mean flow is excluded.
        """
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("Expected a square periodic streamfunction grid")
        if not 0 <= velocity_tolerance < 1:
            raise ValueError("Expected a relative velocity tolerance in [0,1)")
        n = values.shape[0]
        u, sigma, vh = cp.linalg.svd(values - cp.mean(values), full_matrices=False)
        frequency = 2j * cp.pi * cp.fft.fftfreq(n, d=1 / n)
        du = cp.fft.ifft(cp.fft.fft(u, axis=0) * frequency[:, None], axis=0).real
        dv = cp.fft.ifft(cp.fft.fft(vh, axis=1) * frequency[None, :], axis=1).real
        energies = sigma**2 * (cp.sum(du**2, axis=0) + cp.sum(dv**2, axis=1)) / n**2
        tails = cp.concatenate((cp.cumsum(energies[::-1])[::-1][1:], cp.zeros(1)))
        total = cp.sum(energies)
        rank = int(cp.argmax(tails <= velocity_tolerance**2 * total)) + 1
        root = cp.sqrt(sigma[:rank])[:, None]
        factors = cp.stack((root * u[:, :rank].T, root * vh[:rank]), axis=1)
        result = cls(factors, method)
        result.approximation = dict(
            rank=rank,
            velocity_tolerance=velocity_tolerance,
            velocity_l2=float(cp.sqrt(total)),
            velocity_error_l2=float(cp.sqrt(tails[rank - 1])),
        )
        result.factors = factors
        return result

    def __init__(self, factors, method="pcp16"):
        if factors.ndim != 3 or factors.shape[1] != 2:
            raise ValueError("Expected (rank, 2, n) real sampled factors a(y), b(x)")
        self.n = factors.shape[-1]
        if self.n < 3 or self.n % 2 != 1:
            raise ValueError("Fourier shears require an odd grid")
        if method == "group4":
            self.word = [(0, 1.0), (1, 1.0), (0, -1.0), (1, -1.0)]
        elif method == "pcp16":
            c = [
                0.2969175443796203417835,
                1.418243492034305431995,
                0.4347212029859471608694,
                -0.127142127469064995044,
                -2.014276365712093993010,
                0.8493401946712687892513,
                -0.305642216160471071886,
            ]
            c = [-sum(c)] + c
            self.word = [
                (1 - i % 2, coefficient) for i, coefficient in enumerate(c + c[::-1])
            ]
        else:
            raise ValueError("Expected group4 or pcp16 product formula")
        self.method = method
        self.approximation = None
        k = cp.fft.fftfreq(self.n, d=1 / self.n)
        self.phases = [
            (
                -2j * cp.pi * a[:, None] * k[None, :],
                -2j * cp.pi * k[:, None] * b[None, :],
            )
            for a, b in factors
        ]
        last = len(self.phases) - 1
        self.stages = (
            [(i, 0.5) for i in range(last)]
            + [(last, 1.0)]
            + [(i, 0.5) for i in reversed(range(last))]
        )

    def generator(self, value):
        result = cp.zeros_like(value)
        for phase_a, phase_b in self.phases:

            def a(v):
                return cp.fft.ifft(
                    cp.fft.fft(v, axis=1) * phase_a[..., None], axis=1
                ).real

            def b(v):
                return cp.fft.ifft(
                    cp.fft.fft(v, axis=0) * phase_b[..., None], axis=0
                ).real

            result += a(b(value)) - b(a(value))
        return result

    def step(self, value, duration):
        """Advance by a nonnegative duration; compatible with CUDA capture."""
        for term, fraction in self.stages:
            h = cp.sqrt(fraction * duration)
            # The paper writes products left-to-right; rightmost acts first.
            for operator, weight in reversed(self.word):
                axis = 1 - operator
                phase = cp.exp(weight * h * self.phases[term][operator])
                value = cp.fft.ifft(
                    cp.fft.fft(value, axis=axis) * phase[..., None], axis=axis
                ).real
        return value

    def inverse_step(self, value, duration):
        """Exact discrete inverse of step(value, duration), up to roundoff."""
        for term, fraction in reversed(self.stages):
            h = cp.sqrt(fraction * duration)
            for operator, weight in self.word:
                axis = 1 - operator
                phase = cp.exp(-weight * h * self.phases[term][operator])
                value = cp.fft.ifft(
                    cp.fft.fft(value, axis=axis) * phase[..., None], axis=axis
                ).real
        return value
