"""GPU Cartesian reference using symbolically reconstructed HySoP lambda kernels.

This is an isolated CUDA realization of the same mathematical kernels, not a
benchmark of the full HySoP runtime. Distributed under Boost Software License 1.0.
"""
import json

from gpu import cp, HERE, launch


class CartesianTransfer:
    def __init__(self, n, dt, kernel="L4_2", velocity=(1., .37)):
        self.n = n
        self.dt = dt
        self.kernel = kernel
        definition = json.loads((HERE / "symbolic.json").read_text())["kernels"][kernel]
        radius = definition["radius"]
        self.radius = radius
        offsets = cp.arange(-radius+1, radius+1, dtype=cp.int32)
        shifts = []
        weights = []
        for v in velocity:
            displacement = v*dt*n
            integer = int(displacement//1)
            fraction = displacement-integer
            x = cp.abs(fraction-offsets)
            w = cp.zeros(x.shape, dtype=cp.float64)
            for interval, coefficients in enumerate(definition["coefficients"]):
                value = cp.zeros_like(w)
                for coefficient in reversed(coefficients):
                    value = value*x+coefficient
                w += cp.where((x >= interval) & (x < interval+1), value, 0)
            shifts.append(offsets+integer)
            weights.append(w)
        self.shifts = cp.stack(shifts)
        self.weights = cp.stack(weights)
        width = 2*radius
        source = f"""
extern "C" __global__ void remap(const double* input, double* output,
    int n, int fields, const int* shifts, const double* weights, int axis)
{{
  int item = blockDim.x*blockIdx.x+threadIdx.x;
  if (item >= n*n*fields) return;
  int f = item%fields, x = (item/fields)%n, y = item/(fields*n);
  double value = 0.;
  for (int k=0; k<{width}; ++k)
  {{
    int a = (axis == 0 ? x : y)-shifts[axis*{width}+k];
    a = ((a%n)+n)%n;
    int index = axis == 0 ? y*n+a : a*n+x;
    value += weights[axis*{width}+k]*input[index*fields+f];
  }}
  output[item] = value;
}}
"""
        self.operation = cp.RawKernel(source, "remap", options=("--std=c++17",))
        self.operation.compile()

    def apply(self, coefficients, output=None, temporary=None):
        if output is None:
            output = cp.empty_like(coefficients)
        if temporary is None:
            temporary = cp.empty_like(coefficients)
        fields = coefficients.shape[2]
        launch(self.operation, coefficients.size,
               (coefficients, temporary, self.n, fields, self.shifts, self.weights, 0))
        launch(self.operation, coefficients.size,
               (temporary, output, self.n, fields, self.shifts, self.weights, 1))
        return output


def coordinates(n):
    y, x = cp.meshgrid(cp.arange(n, dtype=cp.float64)/n,
                       cp.arange(n, dtype=cp.float64)/n, indexing="ij")
    return cp.stack((x, y), axis=-1)
