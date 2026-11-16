"""Unitary splitting of finite Fourier divergence-free velocities on the torus.

Distributed under the Boost Software License, Version 1.0.
A mode (kx, ky, beta, phase) has velocity
beta*(-b,a)*sin(2*pi*(kx*x+ky*y)+phase), where (a,b) is primitive.
Integer coordinate changes are exact grid permutations. High-frequency
aliasing is still possible: unitarity is not a consistency certificate.
"""

from math import gcd
from gpu import cp, HERE, launch


def bezout(a, b):
    """Return integers s,t satisfying s*a+t*b=gcd(a,b)."""
    old_r, r, old_s, s, old_t, t = a, b, 1, 0, 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    sign = 1 if old_r > 0 else -1
    return sign * old_s, sign * old_t


class FourierModeFlow:
    def __init__(self, n, modes, mean=(0.0, 0.0)):
        if n < 3 or n % 2 != 1:
            raise ValueError("An odd grid of at least three points is required")
        self.n = n
        self.modes = tuple(tuple(mode) for mode in modes)
        self.mean = tuple(mean)
        self.operators = []
        y, x = cp.meshgrid(cp.arange(n), cp.arange(n), indexing="ij")
        frequency = cp.fft.fftfreq(n, d=1 / n)
        physical = []
        if any(mean):
            phase = (
                -2j
                * cp.pi
                * (mean[0] * frequency[None, :] + mean[1] * frequency[:, None])
            )
            self.operators.append((None, None, phase))
        for kx, ky, beta, offset in self.modes:
            if int(kx) != kx or int(ky) != ky or (kx == 0 and ky == 0):
                raise ValueError(
                    "Each Fourier wave vector must be a nonzero integer pair"
                )
            kx, ky = int(kx), int(ky)
            g = gcd(kx, ky)
            a, b = kx // g, ky // g
            d, minus_c = bezout(a, b)
            c = -minus_c
            assert a * d - b * c == 1
            forward = ((-c * x + a * y) % n) * n + (d * x - b * y) % n
            backward = ((c * x + d * y) % n) * n + (a * x + b * y) % n
            velocity = beta * cp.sin(2 * cp.pi * g * cp.arange(n) / n + offset)
            phase = -2j * cp.pi * frequency[:, None] * velocity[None, :]
            self.operators.append((forward, backward, phase))
            physical.append((kx, ky, -beta * b, beta * a, offset))
        self.physical = cp.asarray(physical, dtype=cp.float64).reshape(-1, 5)
        self.reference_kernel = cp.RawKernel(
            (HERE / "fourier_reference.cu").read_text(), "fourier_feet"
        )

    def apply_mode(self, value, operator, duration=None):
        forward, backward, phase = operator
        if forward is None:
            multiplier = phase if duration is None else cp.exp(duration * phase)
            return cp.fft.ifft2(
                cp.fft.fft2(value, axes=(0, 1)) * multiplier[..., None], axes=(0, 1)
            ).real.copy()
        v = value.reshape(self.n * self.n, -1)[forward]
        multiplier = phase if duration is None else cp.exp(duration * phase)
        v = cp.fft.ifft(cp.fft.fft(v, axis=0) * multiplier[..., None], axis=0).real
        return v.reshape(self.n * self.n, -1)[backward]

    def generator(self, value):
        result = cp.zeros_like(value)
        for operator in self.operators:
            result += self.apply_mode(value, operator)
        return result

    def second_order(self, value, duration):
        if not self.operators:
            return value.copy()
        for operator in self.operators[:-1]:
            value = self.apply_mode(value, operator, duration / 2)
        value = self.apply_mode(value, self.operators[-1], duration)
        for operator in reversed(self.operators[:-1]):
            value = self.apply_mode(value, operator, duration / 2)
        return value

    def step(self, value, duration, order=4):
        if order == 2:
            return self.second_order(value, duration)
        if order != 4:
            raise ValueError("Implemented orders are 2 and 4")
        a = 1 / (2 - 2 ** (1 / 3))
        for weight in (a, 1 - 2 * a, a):
            value = self.second_order(value, weight * duration)
        return value

    def feet(self, points, time, steps=1024):
        points = cp.ascontiguousarray(points)
        result = cp.empty_like(points)
        launch(
            self.reference_kernel,
            points.size // 2,
            (
                points,
                result,
                points.size // 2,
                self.physical,
                len(self.modes),
                float(self.mean[0]),
                float(self.mean[1]),
                -float(time),
                steps,
            ),
        )
        return result


CELLULAR = ((0, 1, -0.5, 0.0), (1, 0, -0.5, 0.0), (1, 1, 0.25, 0.0), (1, -1, 0.25, 0.0))
GENERAL = (
    (2, 1, 0.12, 0.3),
    (1, 2, -0.09, -0.5),
    (3, -2, 0.04, 0.2),
    (0, 3, 0.1, 0.7),
    (2, 2, 0.06, -0.3),
    (-2, 3, 0.035, 0.15),
)
