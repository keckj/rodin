"""Fit both marginal cubature rules on GPU, then transfer particle values.

Research hypothesis: positive weights reproducing degree-2p moments on both
meshes give exact marginal Gram matrices, hence an L2-contractive transfer for
volume-preserving maps. Feasibility, solve cost and consistency are measured,
not assumed. Signed weights are an explicit diagnostic variant.

Distributed under the Boost Software License, Version 1.0.
"""
import time

from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import LinearOperator, cg

from gpu import cp, Space, quadrature, launch


class MomentTransfer:
    def __init__(self, space, flow, t0, t1, qdegree=12, subdivisions=1,
                 fit="signed", tolerance=1e-10, maxiter=2000, active_passes=20,
                 moment_degree=None, solver="cg", jacobian=None, coarse_radius=2,
                 positivity="active", projection_steps=10000, particle_data=None):
        if fit not in ("raw", "signed", "positive"):
            raise ValueError(fit)
        if space.degree > 2:
            raise ValueError("Moment fit currently supports DG P0/P1/P2")
        self.space = space
        self.fit = fit
        self.solver = solver
        self.positivity = positivity
        if positivity not in ("active", "dykstra"):
            raise ValueError(positivity)
        if solver not in ("cg", "dense", "coarse"):
            raise ValueError(solver)
        self.diagnostics = []
        moment_degree = 2*space.degree if moment_degree is None else moment_degree
        moment_space = Space(space.mesh, moment_degree)
        q = quadrature(max(qdegree, 2*moment_degree), subdivisions)
        mesh = space.mesh
        count = mesh.cells*q.shape[0]
        self.nq = q.shape[0]
        self.count = count
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        self.source = cp.repeat(cp.arange(mesh.cells, dtype=cp.int32), q.shape[0])
        references = cp.tile(q[:, :2], (mesh.cells, 1))
        physical = mesh.points(q).reshape(-1, 2)
        initial_weights = (2*mesh.areas[:, None]*q[None, :, 2]).ravel()
        if particle_data is not None:
            if solver == "coarse":
                raise ValueError("Fourier diagnostic assumes the original tensor particle layout")
            self.source, references, initial_weights = particle_data
            count = self.source.size
            self.count = count
            self.nq = None
            t = mesh.triangles[self.source]
            physical = t[:, 0]+references[:, :1]*(t[:, 1]-t[:, 0])+references[:, 1:]*(t[:, 2]-t[:, 0])
            if float(cp.min(initial_weights)) <= 0:
                raise ValueError("Custom particle weights must be strictly positive")
        moved = cp.ascontiguousarray(flow(physical, t0, t1))
        jac = cp.ones(count, dtype=cp.float64) if jacobian is None else jacobian(physical, t0, t1).reshape(-1)
        if float(cp.min(jac)) <= 0:
            raise ValueError("Flow Jacobian must be positive")
        self.jacobian_min = float(cp.min(jac))
        self.jacobian_max = float(cp.max(jac))
        self.target = cp.empty(count, dtype=cp.int32)
        target_references = cp.empty_like(moved)
        errors = cp.zeros(1, dtype=cp.int32)
        launch(space.module.get_function("locate_points"), count,
               (mesh.triangles, mesh.n, moved, count, self.target, target_references, errors))
        if int(errors[0]):
            raise RuntimeError("Moment particle location failed")
        self.sb = space.basis(references)
        self.tb = space.basis(target_references)
        self.weights = initial_weights
        initial = self.weights.copy()
        ms = moment_space.basis(references)
        mt = moment_space.basis(target_references)
        nm = moment_space.nb
        # Weighted least change: w = w0 + sqrt(w0)*delta. Row scaling by
        # sqrt(cell area) avoids resolution-dependent conditioning.
        rootw = cp.sqrt(initial)
        rootarea = cp.sqrt(mesh.areas)
        k = cp.arange(nm, dtype=cp.int32)
        rows_source = self.source[:, None]*nm+k[None, :]
        rows_target = (mesh.cells+self.target[:, None])*nm+k[None, :]
        cols = cp.broadcast_to(cp.arange(count, dtype=cp.int32)[:, None], (count, nm))
        data_source = ms*rootw[:, None]/rootarea[self.source, None]
        data_target = mt*(jac*rootw)[:, None]/rootarea[self.target, None]
        self.constraint = sparse.coo_matrix((
            cp.concatenate((data_source.ravel(), data_target.ravel())),
            (cp.concatenate((rows_source.ravel(), rows_target.ravel())),
             cp.concatenate((cols.ravel(), cols.ravel())))),
            shape=(2*mesh.cells*nm, count)).tocsr()
        a = self.constraint
        rhs = cp.zeros((2, mesh.cells, nm), dtype=cp.float64)
        rhs[:, :, 0] = rootarea
        rhs = rhs.ravel()
        self.initial_residual = float(cp.max(cp.abs(a@rootw-rhs)))
        cp.cuda.Stream.null.synchronize()
        self.location_setup_seconds = time.perf_counter()-start
        self.iterations = 0
        coarse = None
        if solver == "coarse":
            # Diagnostic two-level preconditioner for the periodic benchmark:
            # almost-null multiplier pairs represent the same smooth function
            # in source and target coordinates. Fourier modes identify that
            # mechanism without introducing a CPU solve. This is not a general
            # unstructured-domain multigrid preconditioner.
            frequencies = [(i, j) for i in range(-coarse_radius, coarse_radius+1)
                           for j in range(-coarse_radius, coarse_radius+1)
                           if (i > 0 or (i == 0 and j > 0))]
            frequencies = cp.asarray(frequencies, dtype=cp.float64).T
            if frequencies.size == 0:
                raise ValueError("coarse_radius must be positive")
            ps = 2*cp.pi*(moved@frequencies)
            pt = 2*cp.pi*(physical@frequencies)
            fs = cp.concatenate((cp.sin(ps), cp.cos(ps)), axis=1).reshape(mesh.cells, self.nq, -1)
            ft = cp.concatenate((cp.sin(pt), cp.cos(pt)), axis=1).reshape(mesh.cells, self.nq, -1)
            reference_moments = moment_space.basis(q[:, :2])
            source_coeff = -cp.einsum("cqf,qb,q->cbf", fs, reference_moments, 2*q[:, 2])*rootarea[:, None, None]
            target_coeff = cp.einsum("cqf,qb,q->cbf", ft, reference_moments, 2*q[:, 2])*rootarea[:, None, None]
            coarse = cp.concatenate((source_coeff.reshape(mesh.cells*nm, -1),
                                     target_coeff.reshape(mesh.cells*nm, -1)))
            coarse /= cp.linalg.norm(coarse, axis=0)[None, :]
            self.coarse_columns = coarse.shape[1]
        free = cp.ones(count, dtype=cp.float64)
        passes = 1 if fit == "signed" else active_passes
        z = rootw.copy()
        if fit == "positive" and positivity == "dykstra":
            if a.shape[0] > 4096:
                raise ValueError("Dense Dykstra diagnostic is limited to 4096 constraint rows")
            dense = a.toarray()
            u, singular, vh = cp.linalg.svd(dense, full_matrices=False)
            rank = int(cp.sum(singular > singular[0]*1e-13))
            pseudoinverse = (vh[:rank].T/singular[:rank][None, :])@u[:, :rank].T
            self.dense_rank = rank
            self.dense_nonzero_condition = float(singular[0]/singular[rank-1])
            affine_memory = cp.zeros_like(z)
            positive_memory = cp.zeros_like(z)
            for iteration in range(projection_steps):
                candidate = z+affine_memory
                projected = candidate+pseudoinverse@(rhs-a@candidate)
                affine_memory = candidate-projected
                candidate = projected+positive_memory
                z = cp.maximum(candidate, 0)
                positive_memory = candidate-z
                if iteration % 50 == 0 or iteration == projection_steps-1:
                    residual = cp.abs(a@z-rhs).reshape(2, mesh.cells, nm)/rootarea[None, :, None]
                    error = float(cp.max(residual))
                    if error < tolerance:
                        break
            self.iterations = iteration+1
            self.weights = rootw*z
            self.diagnostics.append({"method": "dykstra", "iterations": self.iterations,
                                     "relative_residual": error})
        elif fit != "raw":
            for iteration in range(passes):
                af = a.multiply(free[None, :]).tocsr()
                at = af.T.tocsr()
                residual = rhs-af@rootw
                diagonal = cp.asarray(af.multiply(af).sum(axis=1)).ravel()
                diagonal = cp.maximum(diagonal, 1e-15)
                operator = LinearOperator((a.shape[0], a.shape[0]),
                                          matvec=lambda x: af@(at@x), dtype=cp.float64)
                preconditioner = LinearOperator(operator.shape, matvec=lambda x: x/diagonal,
                                                dtype=cp.float64)
                if coarse is not None:
                    sampled_coarse = at@coarse
                    gram = sampled_coarse.T@sampled_coarse
                    values, vectors = cp.linalg.eigh(gram)
                    # Drop directions below the double-precision resolution of
                    # the coarse Gram spectrum; report the retained size.
                    keep = values > cp.max(values)*1e-13
                    self.coarse_retained = int(cp.sum(keep))
                    whitened = coarse@(vectors[:, keep]/cp.sqrt(values[keep])[None, :])
                    preconditioner = LinearOperator(operator.shape,
                        matvec=lambda x: x/diagonal+whitened@(whitened.T@x), dtype=cp.float64)
                steps = [0]

                def count_step(x):
                    steps[0] += 1

                if solver == "dense":
                    if a.shape[0] > 4096:
                        raise ValueError("Dense GPU SVD is a small-system diagnostic, limited to 4096 rows")
                    correction, _, rank, singular = cp.linalg.lstsq(af.toarray(), residual, rcond=1e-12)
                    z = free*(rootw+correction)
                    info = 0
                    steps[0] = 1
                    self.dense_rank = int(rank)
                    self.dense_nonzero_condition = float(singular[0]/singular[int(rank)-1])
                else:
                    dual, info = cg(operator, residual, tol=tolerance, atol=1e-14,
                                    maxiter=maxiter, M=preconditioner, callback=count_step)
                    z = free*(rootw+at@dual)
                w = rootw*z
                error = float(cp.max(cp.abs(a@z-rhs)))
                minimum = float(cp.min(w/initial))
                self.iterations += steps[0]
                self.diagnostics.append({"pass": iteration, "iterations": steps[0],
                                         "cg_info": info, "residual": error,
                                         "minimum_weight_ratio": minimum})
                if fit == "signed" or minimum >= -1e-12:
                    break
                # A restricted positive active-set experiment: remove negative
                # sites and refit. It can fail even when a different active set
                # is feasible; a failure is not an infeasibility certificate.
                free *= (w >= 0).astype(cp.float64)
            self.weights = rootw*z
        self.final_residual = float(cp.max(cp.abs(a@(self.weights/rootw)-rhs)))
        relative = cp.abs(a@(self.weights/rootw)-rhs).reshape(2, mesh.cells, nm)/rootarea[None, :, None]
        self.relative_moment_residual = float(cp.max(relative))
        self.minimum_weight_ratio = float(cp.min(self.weights/initial))
        self.negative_weight_fraction = float(cp.mean((self.weights < 0).astype(cp.float64)))
        self.weight_change_rms = float(cp.sqrt(cp.mean((self.weights/initial-1)**2)))
        cp.cuda.Stream.null.synchronize()
        self.build_seconds = time.perf_counter()-start
        self.storage_bytes = sum(x.nbytes for x in [self.source, self.target, self.sb, self.tb, self.weights])
        self.constraint_storage_bytes = a.data.nbytes+a.indices.nbytes+a.indptr.nbytes

    def apply(self, coefficients, output=None):
        if output is None:
            output = cp.zeros_like(coefficients)
        else:
            if output.data.ptr == coefficients.data.ptr:
                raise ValueError("In-place transfer is not defined")
            output.fill(0)
        launch(self.space.module.get_function("weighted_particles"), self.count*coefficients.shape[1],
               (self.source, self.target, self.sb, self.tb, self.weights, self.space.mesh.areas,
                self.count, coefficients, output, coefficients.shape[1]))
        return output

    def summary(self):
        return {"fit": self.fit, "solver": self.solver, "positivity": self.positivity,
                "particles": self.count, "quadrature_per_cell": self.nq,
                "build_seconds": self.build_seconds, "iterations": self.iterations,
                "initial_residual": self.initial_residual, "final_residual": self.final_residual,
                "relative_moment_residual": self.relative_moment_residual,
                "minimum_weight_ratio": self.minimum_weight_ratio,
                "negative_weight_fraction": self.negative_weight_fraction,
                "weight_change_rms": self.weight_change_rms,
                "storage_bytes": self.storage_bytes, "diagnostics": self.diagnostics,
                "dense_nonzero_condition": getattr(self, "dense_nonzero_condition", None),
                "coarse_columns": getattr(self, "coarse_columns", None),
                "coarse_retained": getattr(self, "coarse_retained", None),
                "jacobian_min": self.jacobian_min, "jacobian_max": self.jacobian_max}
