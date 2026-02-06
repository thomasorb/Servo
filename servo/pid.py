import math
from dataclasses import dataclass

@dataclass
class PIDConfig:
    kp: float                 # Proportional gain (output per unit error)
    ki: float                 # Integral gain (per second)
    kd: float                 # Derivative gain (seconds)
    dt: float                 # Sample time (seconds)
    out_min: float            # Command lower bound
    out_max: float            # Command upper bound
    deriv_filter_hz: float = 50.0  # Low-pass cutoff for D term (Hz)
    kaw: float = 0.0               # Anti-windup back-calculation gain (0=off)
    deadband: float = 0.0          # Optional error deadband (units)

class PID:
    def __init__(self, cfg: PIDConfig):
        self.cfg = cfg
        self._i = 0.0
        self._prev_meas = None
        self._d_meas_f = 0.0  # filtered derivative of measurement

        # Precompute filter coefficient for derivative term
        #  1st-order LPF: alpha = dt / (RC + dt), RC = 1/(2*pi*fc)
        if cfg.deriv_filter_hz > 0.0:
            rc = 1.0 / (2.0 * math.pi * cfg.deriv_filter_hz)
            self._alpha = cfg.dt / (rc + cfg.dt)
        else:
            self._alpha = 1.0  # no filtering (not recommended)

    def reset(self, integral: float = 0.0):
        self._i = integral
        self._prev_meas = None
        self._d_meas_f = 0.0

    def update(self, control: float, setpoint: float, measurement: float) -> float:
        """Compute the new control output."""
        cfg = self.cfg

        # Error and optional deadband
        error = setpoint - measurement
        if cfg.deadband > 0.0 and abs(error) < cfg.deadband:
            error = 0.0

        # Derivative on measurement (better noise behavior)
        if self._prev_meas is None:
            d_meas = 0.0
        else:
            d_meas = (measurement - self._prev_meas) / cfg.dt
        self._prev_meas = measurement

        # Low-pass filter the measurement derivative
        self._d_meas_f = (1.0 - self._alpha) * self._d_meas_f + self._alpha * d_meas

        # PID components
        p = cfg.kp * error        
    
        i_candidate = self._i + cfg.ki * error * cfg.dt
        d = -cfg.kd * self._d_meas_f  # negative because derivative on measurement

        # Unsaturated output
        u_unsat = control + p #+ i_candidate + d

        # Apply output limits
        u = max(cfg.out_min, min(cfg.out_max, u_unsat))

        # Anti-windup: integrator clamping + optional back-calculation
        saturated = (u != u_unsat)
        if saturated:
            # If the integrator would drive further into saturation, freeze it
            if ((u_unsat > cfg.out_max and error > 0.0) or
                (u_unsat < cfg.out_min and error < 0.0)):
                # keep previous integrator (no growth)
                pass
            else:
                self._i = i_candidate
            # Optional back-calculation (improves recovery)
            if cfg.kaw > 0.0:
                self._i += cfg.kaw * (u - u_unsat) * cfg.dt
        else:
            self._i = i_candidate

        return u
