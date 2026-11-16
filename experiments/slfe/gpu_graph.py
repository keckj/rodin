"""CUDA graph replay of a fixed-shape evolution operator.

Distributed under the Boost Software License, Version 1.0.
The private allocator retains temporary addresses used by captured kernels.
"""

from gpu import cp


class EvolutionGraph:
    def __init__(self, operation, initial, duration):
        cp.cuda.Stream.null.synchronize()
        self.pool = cp.cuda.MemoryPool()
        self.stream = cp.cuda.Stream(non_blocking=True)
        with cp.cuda.using_allocator(self.pool.malloc), self.stream:
            self.value = initial.copy()
            self.duration = cp.asarray(duration, dtype=cp.float64).copy()
            operation(self.value, self.duration)
        self.stream.synchronize()
        with cp.cuda.using_allocator(self.pool.malloc), self.stream:
            self.stream.begin_capture()
            self.output = operation(self.value, self.duration)
            cp.copyto(self.value, self.output)
            self.graph = self.stream.end_capture()
        # Graph replay also needs cuFFT workspace and plans to remain alive.
        self.plans = [node.plan for _, node in cp.fft.config.get_plan_cache()]

    def reset(self, value):
        cp.cuda.Stream.null.synchronize()
        with self.stream:
            cp.copyto(self.value, value)

    def launch(self, duration):
        with self.stream:
            if isinstance(duration, cp.ndarray):
                cp.copyto(self.duration, duration)
            else:
                self.duration.fill(duration)
            self.graph.launch(self.stream)

    def result(self):
        self.stream.synchronize()
        return self.value
