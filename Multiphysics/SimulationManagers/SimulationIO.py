import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from mpi4py import MPI
import dolfinx


from Multiphysics.SimulationManagers.FailureReport import (
    TeeStdout,
    toMapping,
    serializeValue,
)

from Multiphysics.Postprocessors import Postprocessor

try:
    import adios2

    _HAS_ADIOS2 = True
except ImportError:
    _HAS_ADIOS2 = False


class PostManager:
    """
    A manager class for handling multiple postprocessors.
    """

    def __init__(self):
        self._processors = []

    def register(self, post):
        self._processors.append(post)

    def run(self, method_name, *args, **kwargs):
        for p in self._processors:
            method = getattr(p, method_name, None)
            if callable(method):
                method(*args, **kwargs)


class SimulationIO:
    """
    Handles file output, logging, parameter serialization, and post-processing setup.
    """

    def __init__(self, domain, outputs, postprocessors_config):
        self.domain = domain
        self.comm = domain.comm
        self.rank = domain.comm.rank

        # Configuration
        self.outputs = outputs
        self.postprocessors_config = postprocessors_config

        # State
        self.postman = PostManager()
        self._post = None  # The main Postprocessor instance
        self._tee = None
        self._old_stdout = None

    def setupOutput(self, path):
        """Creates directories and sets up logging."""
        bp_dir = os.path.abspath(path)

        # 1. Create directory (Rank 0 only)
        if self.rank == 0:
            os.makedirs(bp_dir, exist_ok=True)
        self.comm.barrier()

        # 2. Setup Logging (Rank 0 only)
        if self.rank == 0:
            log_path = os.path.join(bp_dir, "simulation.log")
            self._tee = TeeStdout(log_path)
            self._old_stdout = sys.stdout
            sys.stdout = self._tee
            print(
                f"[{datetime.now().isoformat()}] Simulation starts, output: {path}",
                flush=True,
            )

    def teardownOutput(self):
        """Restores stdout and closes logs."""
        if self.rank == 0 and self._tee:
            sys.stdout = self._old_stdout
            self._tee.close()

    def setupPostprocessing(self, path, fields, material, action):
        """
        Initializes the main Postprocessor and additional user-defined postprocessors.
        """
        # Main VTK/ADIOS writer
        post = Postprocessor(self.domain, path=path)
        for i, j in self.outputs:
            post.register(i, expr=j)
        post.setupWriter()
        self.postman.register(post)
        self._post = post

        # User-defined postprocessors (e.g. HistvarAtNode)
        for P in self.postprocessors_config:
            # P is tuple: (Class, params_dict)
            if isinstance(P, tuple) and len(P) > 1:
                p_cls, p_params = P
                instance = p_cls(fields, material, action, parameters=p_params)
                self.postman.register(instance)

    def writeStep(self, t):
        """Writes output for current time t."""
        self.postman.run("run")
        self._post.write(t)

    def finalize(self):
        """Closes postprocessors."""
        self.postman.run("close")
        if self._post:
            self._post.close()

    def log(self, *args, **kwargs):
        """
        Wrapper around print that only runs on rank 0.
        Perfectly mimics print() behavior (supports end, sep, flush, multiple args).
        """
        if self.rank == 0:
            # Default to flush=True for safety (logs shouldn't be buffered too long)
            kwargs.setdefault("flush", True)
            print(*args, **kwargs)

    def printAndSaveParams(self, path, material, kernels):
        """Rank 0: Prints params to log and saves them to BP/JSON."""
        if self.rank != 0:
            return

        params = self._gatherParams(material, kernels)
        params_json = json.dumps(params, indent=2, ensure_ascii=False)

        # 1. Print to log
        print("==== Simulation parameters ====", flush=True)
        print("[Material]", flush=True)
        print(json.dumps(params.get("material", {}), indent=2, default=str), flush=True)
        print("[Kernels]", flush=True)
        print(json.dumps(params.get("kernels", []), indent=2, default=str), flush=True)
        print("================================", flush=True)

        # 2. Write Metadata to BP (via Postprocessor hook or ADIOS2 directly)
        self._writeMetadataToBP(path, params, params_json)

        # 3. Write Sidecar JSON
        bp_dir = os.path.abspath(path)
        sidecar = os.path.join(bp_dir, "parameters.json")
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write(params_json)
        print(f"[INFO] Wrote parameters sidecar: {sidecar}", flush=True)

    def _gatherParams(self, material, kernels):
        """(Internal) Collects params from material and kernel objects."""
        meta = {}
        try:
            # -------- 1. Material --------
            mat_par_raw = getattr(material, "par", None)
            mat_map = toMapping(mat_par_raw)

            meta["material"] = {k: serializeValue(v) for k, v in mat_map.items()}

            # -------- 2. Kernels --------
            k_list = []
            for k in kernels:
                try:
                    k_name = "Unknown"
                    k_par = {}


                    if isinstance(k, (tuple, list)):

                        if len(k) > 0:
                            k_obj = k[0]
                            k_name = getattr(k_obj, "__name__", str(k_obj))



                        for item in k:
                            if isinstance(item, dict):
                                k_par = item
                                break


                    elif hasattr(k, "par"):
                        k_name = type(k).__name__
                        k_par = k.par


                    else:
                        k_name = getattr(k, "__name__", str(k))
                        k_par = {}




                    safe_par = {}
                    for kk, vv in k_par.items():

                        if hasattr(vv, "ufl_shape") or "dolfinx" in str(type(vv)):
                            safe_par[kk] = "<dolfinx object>"
                        else:
                            safe_par[kk] = serializeValue(vv)

                    k_list.append({"name": k_name, "par": safe_par})

                except Exception as e:

                    k_list.append(
                        {"name": str(k)[:50] + "...", "par": f"<error parsing: {e}>"}
                    )

            meta["kernels"] = k_list

        except Exception as e:

            if self.rank == 0:
                print(f"[WARN] Failed to gather parameters: {e}", flush=True)
            meta = {"material": {}, "kernels": []}

        return meta

    def _writeMetadataToBP(self, bp_path, params, params_json):
        """(Internal) Tries to inject metadata into the ADIOS file."""
        used_post = False

        # Try hook on Postprocessor
        if self._post:
            try:
                if hasattr(self._post, "addGlobalMetadata"):
                    self._post.addGlobalMetadata(
                        {"material": params["material"], "kernels": params["kernels"]}
                    )
                    used_post = True
                elif hasattr(self._post, "add_attribute"):
                    self._post.add_attribute("parameters_json", params_json)
                    used_post = True
            except Exception:
                pass

        # Fallback to ADIOS2 direct write
        if not used_post and _HAS_ADIOS2:
            try:
                ad = adios2.ADIOS() if hasattr(adios2, "ADIOS") else adios2
                io = ad.DeclareIO("sim_params_io")
                # Split if too long (ADIOS2 limitation in some versions)
                if len(params_json) <= 200000:
                    io.DefineAttribute("parameters_json", params_json)
                else:
                    chunk = 60000
                    for i in range(0, len(params_json), chunk):
                        io.DefineAttribute(
                            f"params_part_{i//chunk}", params_json[i : i + chunk]
                        )

                eng = io.Open(bp_path, adios2.Mode.Append)
                eng.Close()
                print(
                    "[INFO] Wrote parameters as ADIOS2 attributes into BP.", flush=True
                )
            except Exception as e:
                print(f"[WARN] ADIOS2 attribute writing failed: {e}", flush=True)
