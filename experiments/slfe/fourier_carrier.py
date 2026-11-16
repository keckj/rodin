"""Contractive FE/DG/Fourier coupling and unitary cellular split transport.

Distributed under the Boost Software License, Version 1.0.
The global trigonometric space uses an odd number of nodes per direction.
"""

from gpu import cp, HERE, launch
from carrier import Carrier
from flux import gauss_interval


class FourierCarrier:
    def __init__(self, space, factor=1, degree=None, modes=None, quadrature_order=32):
        self.dg = Carrier(space, n=space.mesh.n * factor, degree=degree)
        c = self.dg
        self.n = modes or c.n * c.np - 1
        if self.n % 2 != 1 or self.n < 3:
            raise ValueError(
                "Fourier space needs an odd number of at least three nodes"
            )
        frequency = cp.fft.fftfreq(self.n, d=1 / self.n)
        points, weights = gauss_interval(quadrature_order)
        basis = cp.ones((quadrature_order, c.np))
        for i in range(c.np):
            for j in range(c.np):
                if i != j:
                    basis[:, i] *= (points - c.nodes[j]) / (c.nodes[i] - c.nodes[j])
        self.moments = cp.ascontiguousarray(
            cp.exp((-2j * cp.pi / c.n) * frequency[:, None] * points)
            @ (weights[:, None] * basis)
        )
        module = cp.RawModule(
            code=f"#define NP {c.np}\n" + (HERE / "fourier_carrier.cu").read_text()
        )
        self.project_kernel = module.get_function("dg_to_fourier")
        self.adjoint_kernel = module.get_function("fourier_to_dg")

    def project_dg(self, coefficients):
        c = self.dg
        fields = coefficients.shape[1]
        values = coefficients.reshape(c.n, c.n, fields, c.nc)
        transformed = cp.ascontiguousarray(cp.fft.fft2(values, axes=(0, 1)))
        fourier = cp.empty((self.n, self.n, fields), dtype=cp.complex128)
        launch(
            self.project_kernel,
            fourier.size,
            (transformed, self.moments, fourier, c.n, self.n, fields),
        )
        return (cp.fft.ifft2(fourier, axes=(0, 1)) * self.n**2).real.copy()

    def adjoint_dg(self, values):
        c = self.dg
        fields = values.shape[2]
        coefficients = cp.ascontiguousarray(
            cp.fft.fft2(values, axes=(0, 1)) / self.n**2
        )
        transformed = cp.empty((c.n, c.n, fields, c.nc), dtype=cp.complex128)
        launch(
            self.adjoint_kernel,
            transformed.size,
            (coefficients, self.moments, c.weights, transformed, c.n, self.n, fields),
        )
        result = (cp.fft.ifft2(transformed, axes=(0, 1)) * c.n**2).real
        return cp.ascontiguousarray(result.reshape(c.n * c.n, fields, c.nc))

    def project(self, coefficients):
        return self.project_dg(self.dg.project(coefficients))

    def adjoint(self, values):
        return self.dg.adjoint(self.adjoint_dg(values))


class UnitaryCellular:
    """Four Hamiltonian modes, exact coordinate permutations, Fourier shears.

    The discrete transport is real, mass preserving and unitary up to roundoff.
    Negative substeps are therefore permitted in a fourth-order composition.
    """

    def __init__(self, n):
        if n % 2 != 1:
            raise ValueError("Odd grids avoid an unpaired Nyquist Fourier mode")
        self.n = n
        y, x = cp.meshgrid(cp.arange(n), cp.arange(n), indexing="ij")
        self.plus = y * n + (x - y) % n
        self.minus = y * n + (x + y) % n
        wave = cp.sin(2 * cp.pi * cp.arange(n) / n)
        frequency = cp.fft.fftfreq(n, d=1 / n)
        self.xphase = (-2j * cp.pi * 0.5) * wave[:, None] * frequency[None, :]
        self.yphase = (-2j * cp.pi * (-0.5)) * frequency[:, None] * wave[None, :]
        self.diagonalphase = (-2j * cp.pi * 0.25) * frequency[:, None] * wave[None, :]

    def generator(self, value):
        """Exact generator of the discrete four-mode split system."""

        def derivative(v, axis, phase):
            return cp.fft.ifft(
                cp.fft.fft(v, axis=axis) * phase[..., None], axis=axis
            ).real

        def coordinate(v, indices):
            return v.reshape(self.n * self.n, -1)[indices]

        return (
            derivative(value, 1, self.xphase)
            + derivative(value, 0, self.yphase)
            + coordinate(
                derivative(coordinate(value, self.plus), 0, self.diagonalphase),
                self.minus,
            )
            + coordinate(
                derivative(coordinate(value, self.minus), 0, self.diagonalphase),
                self.plus,
            )
        )

    def second_order(self, value, duration):
        def shear(v, axis, phase):
            return cp.fft.ifft(
                cp.fft.fft(v, axis=axis) * phase[..., None], axis=axis
            ).real

        def coordinate(v, indices):
            return v.reshape(self.n * self.n, -1)[indices]

        px = cp.exp((duration / 2) * self.xphase)
        py = cp.exp((duration / 2) * self.yphase)
        pd = cp.exp((duration / 2) * self.diagonalphase)
        v = shear(value, 1, px)
        v = shear(v, 0, py)
        v = coordinate(shear(coordinate(v, self.plus), 0, pd), self.minus)
        v = coordinate(shear(coordinate(v, self.minus), 0, pd * pd), self.plus)
        v = coordinate(shear(coordinate(v, self.plus), 0, pd), self.minus)
        v = shear(v, 0, py)
        return shear(v, 1, px)

    def step(self, value, duration, order=4):
        if order == 2:
            return self.second_order(value, duration)
        if order != 4:
            raise ValueError("Implemented composition orders are 2 and 4")
        a = 1 / (2 - 2 ** (1 / 3))
        for weight in (a, 1 - 2 * a, a):
            value = self.second_order(value, weight * duration)
        return value
