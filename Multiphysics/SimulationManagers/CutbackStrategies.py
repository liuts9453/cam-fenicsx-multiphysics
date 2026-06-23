# Multiphysics/SimulationManagers/CutbackStrategies.py

class BaseCutbackStrategy:
    """
    Cutback 
    """
    def reset(self, target_dt: float):
        """"""
        raise NotImplementedError

    def isDone(self) -> bool:
        """"""
        raise NotImplementedError

    def getStepInfo(self) -> tuple[float, float]:
        """
        
        : (dt_try, alpha_next) 
              dt_try: 
              alpha_next:  ()
        """
        raise NotImplementedError

    def update(self, success: bool, n_its: int = -1, is_unknown_block: bool = False):
        """
        
        """
        raise NotImplementedError


class AdaptiveCutbackStrategy(BaseCutbackStrategy):
    """
     Cutback  Newton-Raphson 
    """
    def __init__(self, maxCuts=20, growthFactor=1.25, recoverFactor=2.0, fastConvIts=5):
        self.maxCuts = maxCuts
        self.growthFactor = growthFactor
        self.recoverFactor = recoverFactor
        self.fastConvIts = fastConvIts
        
        self.target_dt = 0.0
        self.current_dt = 0.0
        self.accumulated_dt = 0.0
        self.cuts = 0
        self._is_done = False

    def reset(self, target_dt: float):
        self.target_dt = float(target_dt)
        self.current_dt = self.target_dt
        self.accumulated_dt = 0.0
        self.cuts = 0
        self._is_done = False

    def isDone(self) -> bool:

        if self.accumulated_dt >= self.target_dt - 1e-12:
            self._is_done = True
        return self._is_done

    def getStepInfo(self) -> tuple[float, float]:

        dt_remaining = self.target_dt - self.accumulated_dt
        dt_try = min(self.current_dt, dt_remaining)
        

        alpha_next = (self.accumulated_dt + dt_try) / self.target_dt
        return dt_try, alpha_next

    def update(self, success: bool, n_its: int = -1, is_unknown_block: bool = False):
        dt_remaining = self.target_dt - self.accumulated_dt
        dt_try = min(self.current_dt, dt_remaining)

        if success:

            self.accumulated_dt += dt_try
            self.cuts = 0
            

            if 0 < n_its <= self.fastConvIts and not is_unknown_block:
                self.current_dt = min(self.current_dt * self.growthFactor, self.target_dt)
            else:

                self.current_dt = min(self.current_dt * self.recoverFactor, self.target_dt)
        else:

            self.cuts += 1
            if self.cuts > self.maxCuts:
                raise RuntimeError(f"AdaptiveCutbackStrategy failed: Exceeded maximum cuts ({self.maxCuts}).")
            

            self.current_dt *= 0.5


class ConstantCutbackStrategy(BaseCutbackStrategy):
    """
    
    ()
    """
    def __init__(self):
        self.cuts = 0

    def reset(self, target_dt: float):
        self._dt = float(target_dt)
        self._done = False

    def isDone(self) -> bool:
        return self._done

    def getStepInfo(self) -> tuple[float, float]:
        return self._dt, 1.0

    def update(self, success: bool, n_its: int = -1, is_unknown_block: bool = False):
        if not success:
            raise RuntimeError("ConstantCutbackStrategy: Solver diverged and cutback is not allowed.")
        self._done = True
