from collections import defaultdict
from typing import Dict, List
import csv
import jax.numpy as jnp

def load_tables_csv(path: str) -> Dict[str, jnp.ndarray]:
    """Read a CSV file and return a dict mapping column names (spaces removed) 
    to jax.numpy arrays of floats."""
    table: Dict[str, List[float]] = defaultdict(list)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                clean_key = key.replace(" ", "")
                table[clean_key].append(float(value))


    return {k: jnp.asarray(v) for k, v in table.items()}

