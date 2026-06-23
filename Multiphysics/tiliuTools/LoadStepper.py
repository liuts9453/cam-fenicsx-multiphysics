from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Literal, Optional


Mode = Literal["total_load+n_steps", "total_time+n_steps", "total_load+dt", "total_time+dt"]


@dataclass(frozen=True)
class LoadStepper:
    """
    Rate-controlled 1D loading schedule.

    State variables are abstract:
      - load(t): a scalar control quantity (could be displacement, strain, temperature, etc.)
      - rate: d(load)/dt (constant here)

    You can specify the schedule in several equivalent ways:
      1) total_load + n_steps  -> infer dt and total_time
      2) total_time + n_steps  -> infer dt and total_load
      3) total_load + dt       -> infer n_steps (rounded) and total_time
      4) total_time + dt       -> infer n_steps (rounded) and total_load
    """
    rate: float
    mode: Mode

    total_load: Optional[float] = None
    total_time: Optional[float] = None
    n_steps: Optional[int] = None
    dt: Optional[float] = None

    t0: float = 0.0
    load0: float = 0.0

    endpoint: bool = True  # include last point (t_end, load_end)
    dtype: type = np.float64

    def __post_init__(self):
        if not np.isfinite(self.rate) or self.rate <= 0.0:
            raise ValueError(f"rate must be positive and finite, got {self.rate}")

        # Basic presence checks per mode
        if self.mode == "total_load+n_steps":
            if self.total_load is None or self.n_steps is None:
                raise ValueError("mode 'total_load+n_steps' requires total_load and n_steps")
            if self.total_load <= 0 or self.n_steps <= 0:
                raise ValueError("total_load > 0 and n_steps > 0 required")

        elif self.mode == "total_time+n_steps":
            if self.total_time is None or self.n_steps is None:
                raise ValueError("mode 'total_time+n_steps' requires total_time and n_steps")
            if self.total_time <= 0 or self.n_steps <= 0:
                raise ValueError("total_time > 0 and n_steps > 0 required")

        elif self.mode == "total_load+dt":
            if self.total_load is None or self.dt is None:
                raise ValueError("mode 'total_load+dt' requires total_load and dt")
            if self.total_load <= 0 or self.dt <= 0:
                raise ValueError("total_load > 0 and dt > 0 required")

        elif self.mode == "total_time+dt":
            if self.total_time is None or self.dt is None:
                raise ValueError("mode 'total_time+dt' requires total_time and dt")
            if self.total_time <= 0 or self.dt <= 0:
                raise ValueError("total_time > 0 and dt > 0 required")

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def build(self) -> dict[str, np.ndarray]:
        """
        Returns:
          {
            "dt": scalar float,
            "time_steps": (n_steps,) array,
            "time": (n_steps,) array of cumulative time (t0 excluded, like your original),
            "load": (n_steps,) array of load values at each step
          }
        """
        rate = float(self.rate)

        if self.mode == "total_load+n_steps":
            total_load = float(self.total_load)  # type: ignore[arg-type]
            n_steps = int(self.n_steps)  # type: ignore[arg-type]
            total_time = total_load / rate
            dt = total_time / n_steps

        elif self.mode == "total_time+n_steps":
            total_time = float(self.total_time)  # type: ignore[arg-type]
            n_steps = int(self.n_steps)  # type: ignore[arg-type]
            dt = total_time / n_steps
            total_load = rate * total_time

        elif self.mode == "total_load+dt":
            total_load = float(self.total_load)  # type: ignore[arg-type]
            dt = float(self.dt)  # type: ignore[arg-type]
            total_time = total_load / rate
            n_steps = int(np.round(total_time / dt))
            n_steps = max(n_steps, 1)
            # Recompute dt so the schedule hits total_load exactly at the end
            dt = total_time / n_steps

        elif self.mode == "total_time+dt":
            total_time = float(self.total_time)  # type: ignore[arg-type]
            dt = float(self.dt)  # type: ignore[arg-type]
            n_steps = int(np.round(total_time / dt))
            n_steps = max(n_steps, 1)
            dt = total_time / n_steps
            total_load = rate * total_time

        else:
            raise RuntimeError("unreachable")

        time_steps = np.full((n_steps,), dt, dtype=self.dtype)

        # Match your original convention:
        # time = cumsum(time_steps) starts at dt, ends at n_steps*dt
        time = self.t0 + np.cumsum(time_steps)
        load = self.load0 + rate * (time - self.t0)

        if not self.endpoint:
            # drop the last point (useful if you want "increments" only)
            time_steps = time_steps[:-1]
            time = time[:-1]
            load = load[:-1]

        return {
            "dt": np.asarray(dt, dtype=self.dtype),
            "time_steps": time_steps,
            "time": time,
            "load": load,
        }


# -------------------------
# Example: your original numbers
# -------------------------
if __name__ == "__main__":
    chi = 0.23
    total_load = 0.1
    n_steps = 100
    rate = 50 / 60  # eps_dot in your snippet

    stepper = LoadStepper(rate=rate, mode="total_load+n_steps", total_load=total_load, n_steps=n_steps)
    sched = stepper.build()

    dt = float(sched["dt"])
    time = sched["time"]
    load = sched["load"]

    print("dt =", dt)
    print("time[:3] =", time[:3])
    print("load[:3] =", load[:3])
    print("load[-1] =", load[-1])

