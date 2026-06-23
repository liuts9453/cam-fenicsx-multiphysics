from .CustomPostprocessor import CustomPostprocessor
import numpy as np
import os
import ufl

class Extractor(CustomPostprocessor):
    """
    Extracts values from fields at specific indices and saves to CSV.
    Supports:
    - node_index + component: for standard FEM fields (e.g. 'u')
    - qp_index + component: for quadrature point fields (e.g. 'stress')
    - index: direct array indexing
    """
    def initialize(self):
        self.filename = self.par.get("filename", "extraction.csv")
        self.items = self.par.get("items", [])
        self.results = []
        
        labels = []
        for it in self.items:
            if "label" in it:
                labels.append(it["label"])
            else:
                name = it["field"]
                if "index" in it:
                    labels.append(f"{name}_{it['index']}")
                elif "node_index" in it:
                    labels.append(f"{name}_n{it['node_index']}_c{it.get('component', 0)}")
                elif "qp_index" in it:
                    labels.append(f"{name}_qp{it['qp_index']}_c{it.get('component', 0)}")
                else:
                    labels.append(f"{name}_0")
        
        self.header = ",".join(labels)

    def run(self):
        comm = self.fields.domain.comm
        if comm.rank != 0:
            return
            
        row = []
        for it in self.items:
            name = it["field"]
            f = self.fields.get(name)
            
            val = np.nan
            if f is not None and hasattr(f, "x"):
                try:
                    if "index" in it:
                        idx = it["index"]
                    else:
                        # Determine number of components
                        shape = ufl.shape(f)
                        n_comp = np.prod(shape, dtype=int) if shape else 1
                        comp = it.get("component", 0)
                        
                        if "node_index" in it:
                            idx = it["node_index"] * n_comp + comp
                        elif "qp_index" in it:
                            idx = it["qp_index"] * n_comp + comp
                        else:
                            idx = comp
                    
                    if idx < f.x.array.size:
                        val = f.x.array[idx]
                except Exception:
                    pass
            
            row.append(val)
        
        self.results.append(row)

    def close(self):
        if self.fields.domain.comm.rank == 0 and self.results:
            dirname = os.path.dirname(self.filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            np.savetxt(self.filename, np.array(self.results), header=self.header, delimiter=",", comments='')
            print(f"\n[Extractor] Data saved to {self.filename}")
