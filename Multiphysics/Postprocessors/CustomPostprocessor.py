import numpy as np
import os
from mpi4py import MPI

class CustomPostprocessor:
    def __init__(self, fields, mat, action, aux_action=None, parameters={}):
        self.fields = fields
        self.action = action
        self.material = mat
        self.aux_action = aux_action
        self.par = parameters
        self.results = []
        self.filename = parameters.get("filename", "force_temp.csv")
        self.initialize()

    def initialize(self):
        return

    def preprocess(self):
        return

    def run(self):
        comm = self.fields.domain.comm
        rank = comm.rank
        
        # 1. Stress (Force) - Component 0 is sigma_xx
        stress_field = self.fields.get("stress")
        s_sum_local = 0.0
        s_cnt_local = 0
        if stress_field is not None:
            local_stress = stress_field.x.array.reshape(-1, 6)
            if local_stress.size > 0:
                s_sum_local = np.sum(local_stress[:, 0])
                s_cnt_local = local_stress.shape[0]
        
        # 2. Temperature and Displacement (Load) from 'u'
        u_field = self.fields.get("u")
        ux_max_local = -1e10
        t_sum_local = 0.0
        t_cnt_local = 0
        if u_field is not None:
            # u = (ux, uy, uz, T)
            # Standard field degree 1 has 4 components per node
            local_u = u_field.x.array.reshape(-1, 4)
            if local_u.size > 0:
                ux_max_local = np.max(local_u[:, 0])
                t_sum_local = np.sum(local_u[:, 3])
                t_cnt_local = local_u.shape[0]
        
        # Gather data to Rank 0
        s_sum = comm.reduce(s_sum_local, op=MPI.SUM)
        s_cnt = comm.reduce(s_cnt_local, op=MPI.SUM)
        u_load = comm.reduce(ux_max_local, op=MPI.MAX)
        t_sum = comm.reduce(t_sum_local, op=MPI.SUM)
        t_cnt = comm.reduce(t_cnt_local, op=MPI.SUM)
        
        if rank == 0:
            stress_xx = s_sum / s_cnt if s_cnt > 0 else 0.0
            temp_avg = t_sum / t_cnt if t_cnt > 0 else 0.0
            self.results.append([u_load, stress_xx, temp_avg])

    def close(self):
        if self.fields.domain.comm.rank == 0:
            if self.results:
                data = np.array(self.results)
                header = "load_disp,stress_xx,temperature"
                
                # Create directory if needed
                dirname = os.path.dirname(self.filename)
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
                
                np.savetxt(self.filename, data, header=header, delimiter=",")
                print(f"\n[CustomPostprocessor] Results saved to {self.filename}")
