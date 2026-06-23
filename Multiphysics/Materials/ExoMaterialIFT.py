from .ExoMaterial import ExoMaterial
import jax



class ExoMaterialIFT(ExoMaterial):
    """Material base for local Newton updates with IFT-based consistent tangents."""

    @classmethod
    def localResidualBlockNames(cls):
        """
        Return list of names for each local Newton block.
        Override in material implementations.

        Default: unknown blocks -> numbered.
        """
        return None

    def computeQpJacobian(self):
        return self.computeQpJacobianIFT

    @staticmethod
    def getQpJacobianInAxes():
        raise NotImplementedError(
            "getQpJacobianInAxes() must be overridden by an IFT material class."
        )

    @staticmethod
    def getQpJacobianStaticArgnums():
        return ()

    @staticmethod
    def computeQpJacobianIFT(driving_force, hist, hist_n, glob, par):
        raise NotImplementedError(
            "computeQpJacobianIFT() must be overridden by an IFT material class."
        )

    def computeJacobian(self):
        self.computeProperties()
        _qp_jac = self.computeQpJacobian()
        axes = self.getQpJacobianInAxes()
        static_argnums = self.getQpJacobianStaticArgnums()
        self.jac = jax.jit(
            jax.vmap(_qp_jac, in_axes=axes), static_argnums=static_argnums
        )

    def evaluateTangent(self, variables):
        import numpy as np
        from mpi4py import MPI

        # --- reshape inputs ---
        vars_ = variables.reshape((-1, self.driving_force.ufl_shape[0]))
        old_hist = self.hist.x.array[:].reshape((-1, self.hist_len))

        # --- compute globals once ---
        self.computeGlobalValues()

        # --- Step 1: local constitutive update (Gauss-point solver) ---
        # _flux returns (unused_main), (hist_new, converged, niter, history)
        _, (hist_new, converged, niter, history) = self._flux(
            vars_, old_hist, self.global_vals, self.par
        )

        # --- Step 2: local diagnostics & rank-0 logging ---
        # Convert to host/NumPy for safe printing and reductions.
        conv_np = np.array(converged)
        niter_np = np.array(niter)
        hist_np = np.array(history)

        # Backward-compatible shape normalization:
        # old format:
        #   conv: (batch,), niter: (batch,), hist: (batch, max_it)
        # new format (recommended):
        #   conv: (batch, 2), niter: (batch, 2), hist: (batch, 2, max_it)
        if conv_np.ndim == 1:
            conv_np = conv_np[:, None]  # (batch, 1)
        if niter_np.ndim == 1:
            niter_np = niter_np[:, None]  # (batch, 1)
        if hist_np.ndim == 2:
            hist_np = hist_np[:, None, :]  # (batch, 1, max_it)

        # Helper for compact residual history printing
        def _trim_hist(h):
            # Trim trailing zeros if padded, keep at least one entry
            nz = np.flatnonzero(h != 0.0)
            end = nz[-1] + 1 if nz.size > 0 else 1
            return h[:end]

        blk_names = None
        if hasattr(self, "localResidualBlockNames"):
            blk_names = self.localResidualBlockNames()
        
        if not blk_names:
            # fallback to numbered blocks
            blk_names = [f"block{i}" for i in range(conv_np.shape[1])]

        if self.domain.comm.rank == 0:
            # Print only if there was any iteration somewhere
            any_iter = int(np.max(niter_np)) > 0
            if any_iter:
                # print("      Solving internal variables...", flush=True)

                # Prefer printing the first failing qp if any, otherwise qp 0
                fail = np.where(~conv_np)
                if fail[0].size > 0:
                    qp0 = int(fail[0][0])
                else:
                    qp0 = 0

                nblk = conv_np.shape[1]
                for blk in range(nblk):
                    name = blk_names[blk] if blk < len(blk_names) else f"block{blk}"
                    h0_trim = _trim_hist(hist_np[qp0, blk])

                    head = f"            Local |R| ({name}) = : {h0_trim[0]}\n"
                    tail = "".join(
                        f"                          {val}\n" for val in h0_trim[1:]
                    )
                    # print(head + tail, end="", flush=True)

                    ok0 = bool(conv_np[qp0, blk])
                    it0 = int(niter_np[qp0, blk])
                    # print(
                    #    f"            The local residuum ({name}) has "
                    #    f"{'converged' if ok0 else '\033[31m not\033[0m converged'} "
                    #    f"in {it0} steps",
                    #    flush=True,
                    # )

        # --- Step 3: parallel-safe success check BEFORE calling jac ---
        # All blocks at all qps must converge, and residual history must be finite.
        local_ok = bool(conv_np.all() and np.isfinite(hist_np).all())

        # Reduce to a single global boolean via MIN (1 => ok, 0 => fail)
        comm = self.domain.comm
        sendbuf = np.array(1 if local_ok else 0, dtype=np.int32)
        recvbuf = np.array(0, dtype=np.int32)
        comm.Allreduce(sendbuf, recvbuf, op=MPI.MIN)
        global_ok = bool(recvbuf)

        if not global_ok:
            comm = self.domain.comm

            # fail_mask: True where not converged
            fail_mask = ~conv_np  # shape (batch, nblk)

            # Global fail counts per block across all ranks
            local_fail_counts = fail_mask.sum(axis=0).astype(np.int64)  # (nblk,)
            global_fail_counts = np.zeros_like(local_fail_counts)
            comm.Allreduce(local_fail_counts, global_fail_counts, op=MPI.SUM)

            # Build a compact message: only which block(s) failed globally
            if comm.rank == 0:
                failed_blocks = []
                nblk = int(global_fail_counts.shape[0])
                for blk in range(nblk):
                    name = blk_names[blk] if blk < len(blk_names) else f"block{blk}"
                    if int(global_fail_counts[blk]) > 0:
                        failed_blocks.append(name)

                if failed_blocks:
                    blk_text = ", ".join(failed_blocks)
                    err_msg = f"Local constitutive update did not converge (global): {blk_text}."
                else:
                    # Should not happen if global_ok is False, but keep it robust
                    err_msg = "Local constitutive update did not converge (global): unknown block."

                print(f"\033[31m      {err_msg}\033[0m", flush=True)
            else:
                err_msg = None

            # Broadcast one identical message to all ranks, then raise
            err_msg = comm.bcast(err_msg, root=0)
            raise RuntimeError(err_msg)

        # --- Step 4: compute consistent tangent on the converged state ---
        # Keep A's explicit API: jac(vars_, hist_new, old_hist, globals, par)
        cijkl_ = self.jac(vars_, hist_new, old_hist, self.global_vals, self.par)
        if not np.isfinite(cijkl_).all():
            raise RuntimeError("NaN in consistent tangent BEFORE FEM assembly")

        # --- Step 5: last-resort numerical sanitization (harmless if clean) ---
        cijkl_ = np.nan_to_num(np.array(cijkl_), nan=0.0, posinf=0.0, neginf=0.0)
        hist_ok = np.nan_to_num(np.array(hist_new), nan=0.0, posinf=0.0, neginf=0.0)

        return cijkl_.reshape(-1), hist_ok.reshape(-1)

    def computeFunctionDerivatives(self):
        legacy_impl = type(self).__dict__.get("computeFunctionDerivates")
        if legacy_impl is not None and legacy_impl is not ExoMaterialIFT.computeFunctionDerivates:
            return legacy_impl(self)

        raise NotImplementedError(
            "computeFunctionDerivatives() must be implemented by the concrete "
            "ExoMaterialIFT subclass. See TVPkinIFT_heat for the required pattern."
        )

    def computeFunctionDerivates(self):
        return self.computeFunctionDerivatives()

