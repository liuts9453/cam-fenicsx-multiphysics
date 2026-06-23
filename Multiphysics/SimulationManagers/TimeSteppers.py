from abc import ABC, abstractmethod

class TimeStepper(ABC):
    """
    Abstract base class for time stepping strategies.
    Methods use lowerCamelCase to match the project style.
    """
    
    @abstractmethod
    def reset(self, dtMacro: float):
        """Prepare for a new macro load step."""
        pass

    @abstractmethod
    def isDone(self) -> bool:
        """Check if the current macro step is finished."""
        pass

    @abstractmethod
    def getStepInfo(self):
        """
        Calculate and return the parameters for the next trial step.
        :return: (dtTry, alphaNext)
                 where alphaNext is the interpolation factor (0.0 to 1.0)
                 for the end of this trial step.
        """
        pass

    @abstractmethod
    def update(self, success: bool, **kwargs):
        """
        Feedback the result of the solver to the stepper.
        :param success: True if converged, False otherwise.
        :param kwargs: Additional info (e.g., n_its, error_type).
        """
        pass
    
    @property
    @abstractmethod
    def cuts(self) -> int:
        """Return current number of cutbacks (for logging)."""
        pass


class AdaptiveTimeStepper(TimeStepper):
    """
    Implements the adaptive strategy with cutbacks, growth, 
    and UnknownBlock recovery logic.
    """

    def __init__(
        self, 
        maxCuts=20, 
        growthFactor=1.25, 
        recoverFactor=2.0, 
        fastConvIts=5,
        tol=1e-14
    ):
        self.maxCuts = maxCuts
        self.growthFactor = growthFactor
        self.recoverFactor = recoverFactor
        self.fastConvIts = fastConvIts
        self._tol = tol
        
        # Internal state
        self.dtMacro = 0.0
        self.done = 0.0
        self.dtTry = 0.0
        self._cuts = 0
        self.wantRecover = False

    def reset(self, dtMacro: float):
        self.dtMacro = float(dtMacro)
        self.done = 0.0
        self.dtTry = self.dtMacro
        self._cuts = 0
        self.wantRecover = False

    def isDone(self) -> bool:
        return self.done >= 1.0 - self._tol

    @property
    def cuts(self) -> int:
        return self._cuts

    def getStepInfo(self):
        rem = 1.0 - self.done
        # Implicitly enforce min(dtTry, dtMacro) via rem
        self.dtTry = min(self.dtTry, self.dtMacro * rem)
        
        # Calculate target interpolation factor
        alphaNext = self.done + self.dtTry / self.dtMacro
        
        # Clamp to 1.0
        if alphaNext > 1.0:
            alphaNext = 1.0
            
        return self.dtTry, alphaNext

    def update(self, success: bool, **kwargs):
        # Extract optional arguments with defaults
        nIts = kwargs.get("n_its", -1)
        isUnknownBlock = kwargs.get("is_unknown_block", False)

        if success:
            self._handleSuccess(nIts)
        else:
            self._handleFailure(isUnknownBlock)

    def _handleSuccess(self, nIts):
        # Advance progress
        fraction = self.dtTry / self.dtMacro
        self.done += fraction
        
        # Determine NEXT dt
        dtNext = self.dtTry
        grew = False
        
        # Strategy: Recover
        if self.wantRecover:
            candidate = self.dtTry * self.recoverFactor
            if candidate > dtNext:
                dtNext = candidate
            if dtNext > self.dtTry * (1.0 + self._tol):
                grew = True
            if dtNext >= self.dtMacro * (1.0 - self._tol):
                self.wantRecover = False

        # Strategy: Fast Convergence Growth
        if (nIts is not None) and (nIts >= 0) and (nIts < self.fastConvIts):
            candidate = self.dtTry * self.growthFactor
            if candidate > dtNext:
                dtNext = candidate
            if dtNext > self.dtTry * (1.0 + self._tol):
                grew = True

        # Bonus: reduce cut count if we grew
        if grew and self._cuts > 0:
            self._cuts -= 1
            
        self.dtTry = dtNext

    def _handleFailure(self, isUnknownBlock):
        self._cuts += 1
        if self._cuts > self.maxCuts:
            raise RuntimeError(f"Too many cutbacks ({self._cuts}) inside macro step.")
            
        self.dtTry *= 0.5
        
        if isUnknownBlock:
            self.wantRecover = True
