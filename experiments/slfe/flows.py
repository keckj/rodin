"""Analytic characteristic maps and transported densities, evaluated on CUDA.

Distributed under the Boost Software License, Version 1.0.
"""
from gpu import cp, smooth


def shear(points, t0, t1, amplitude=.2):
    result = points.copy()
    result[..., 0] += amplitude*cp.sin(2*cp.pi*points[..., 1])*(t1-t0)
    return result


def shear_solution(points, time):
    return smooth(shear(points, time, 0), 0)


def compressible(points, t0, t1, amplitude=.15):
    """Exact periodic flow of u=(a*sin(2*pi*x), 0), with an unwrapped x."""
    result = points.copy()
    x = points[..., 0]
    whole = cp.floor(x)
    theta = cp.pi*(x-whole)
    factor = cp.exp(cp.asarray(2*cp.pi*amplitude*(t1-t0)))
    result[..., 0] = whole+cp.arctan2(factor*cp.sin(theta), cp.cos(theta))/cp.pi
    return result


def compressible_jacobian(points, t0, t1, amplitude=.15):
    theta = cp.pi*points[..., 0]
    factor = cp.exp(cp.asarray(2*cp.pi*amplitude*(t1-t0)))
    return factor/(cp.cos(theta)**2+factor**2*cp.sin(theta)**2)


def compressible_solution(points, time):
    feet = compressible(points, time, 0)
    return smooth(feet, 0)/compressible_jacobian(feet, 0, time)


def cellular_velocity(points, time, period=1.):
    x, y = points[..., 0], points[..., 1]
    amplitude = cp.cos(cp.asarray(cp.pi*time/period))
    return cp.stack((cp.sin(cp.pi*x)**2*cp.sin(2*cp.pi*y),
                     -cp.sin(cp.pi*y)**2*cp.sin(2*cp.pi*x)), axis=-1)*amplitude


def reversible(points, t0, t1, substeps=8):
    """RK4 flow of a divergence-free cellular field reversing over [0,1].

    The exact full-period map is identity. Substepping is explicit and recorded;
    numerical trajectories are not asserted to preserve volume exactly.
    """
    x = points.copy()
    h = (t1-t0)/substeps
    for i in range(substeps):
        t = t0+i*h
        k1 = cellular_velocity(x, t)
        k2 = cellular_velocity(x+h/2*k1, t+h/2)
        k3 = cellular_velocity(x+h/2*k2, t+h/2)
        k4 = cellular_velocity(x+h*k3, t+h)
        x += h/6*(k1+2*k2+2*k3+k4)
    return x
