import math
from dataclasses import dataclass
import logging
from . import config

log = logging.getLogger(__name__)

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

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
        # Recompute filter coefficient if deriv_filter_hz changed
        if 'deriv_filter_hz' in kwargs:
            if self.cfg.deriv_filter_hz > 0.0:
                rc = 1.0 / (2.0 * math.pi * self.cfg.deriv_filter_hz)
                self._alpha = self.cfg.dt / (rc + self.cfg.dt)
            else:
                self._alpha = 1.0

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
        u_unsat = control + p + i_candidate + d

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

class PiezoPID(PID):

    def __init__(self, data, coeff_key, dt, out_min, out_max,
                 deriv_filter_hz=50.0, kaw=0.0, deadband=0.0):

        self.data = data
        self.coeff_key = coeff_key
        
        cfg = PIDConfig(
            dt=dt, out_min=out_min, out_max=out_max,
            deriv_filter_hz=deriv_filter_hz, kaw=kaw, deadband=deadband,
            kp=0.0, ki=0.0, kd=0.0)
        
        super().__init__(cfg)
        
        self.update_coeffs()

    def update_coeffs(self):
        kp, ki, kd = self.data[self.coeff_key][:3]
        self.cfg.kp = float(kp)
        self.cfg.ki = float(ki)
        self.cfg.kd = float(kd)

    def update_config(self, **kwargs):
        """Update PID configuration parameters and recompute coefficients if needed.

        Note that the PID gains (kp, ki, kd) are obtained via
        shared data and cannot be directly set via kwargs.
        """
        super().update_config(**kwargs)
        self.update_coeffs()


import numpy as np
from collections import deque
from pathlib import Path

class AdaptiveFIRShortMIMO:
    """
    Short-term adaptive FIR (MIMO) for vibration rejection and cross-axis coupling.

    Shared input history:
        e(t) = [e_opd, e_tip, e_tilt]

    Independent outputs:
        u_ff_opd, u_ff_da1, u_ff_da2
    """

    AXES = ("OPD", "DA1", "DA2")

    def __init__(
        self,
        name,
        shared_data,
        n_taps=20,
        u_max=0.2,
    ):
        self.name = name
        self.data = shared_data

        self.n_taps = n_taps
        self.u_max = u_max

        self.dim_in = 3 * n_taps
        self.e_hist = deque(maxlen=n_taps)

        # One weight vector per axis
        self.w = {
            axis: np.zeros(self.dim_in, dtype=np.float64)
            for axis in self.AXES
        }

        self._init_storage()

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
    def _init_storage(self):
        base = Path.home() / ".local" / "state" / "servo"
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"fir_short_mimo_{self.name}.npz"
        self.load()

    def save(self):
        np.savez(self.path, **{axis: self.w[axis] for axis in self.AXES})

    def load(self):
        if self.path.exists():
            data = np.load(self.path)
            for axis in self.AXES:
                self.w[axis] = data[axis]

    def reset(self):
        self.e_hist.clear()
        for axis in self.AXES:
            self.w[axis].fill(0.0)

    # --------------------------------------------------
    # Main update
    # --------------------------------------------------
    def update(self, e_opd, e_tip, e_tilt):
        """
        Returns:
            dict with keys OPD, DA1, DA2
        """
        # --- Update shared history ---
        self.data[f"FIR.short.{self.name}.e_opd"][0] = e_opd
        self.data[f"FIR.short.{self.name}.e_tip"][0] = e_tip
        self.data[f"FIR.short.{self.name}.e_tilt"][0] = e_tilt

        e_opd_n  = e_opd  / self.data['params.FIR_SCALE_OPD'][0]
        e_tip_n  = e_tip  / self.data['params.FIR_SCALE_ANG'][0]
        e_tilt_n = e_tilt / self.data['params.FIR_SCALE_ANG'][0]
        
        self.e_hist.appendleft(np.array([e_opd_n, e_tip_n, e_tilt_n]))

        if len(self.e_hist) < self.n_taps:
            return {axis: 0.0 for axis in self.AXES}

        # Regressor vector (shared)
        x = np.concatenate(self.e_hist)  # shape: (3*n_taps,)

        outputs = {}

        self.mu = self.data['params.FIR_LMS_MU'][0]
        for axis in self.AXES:
            # --- FIR output ---
            u_raw = float(np.dot(self.w[axis], x))

            # --- LMS update ---
            err_axis = {
                "OPD": e_opd_n,
                "DA1": e_tip_n,
                "DA2": e_tilt_n,
            }[axis]

            norm_x2 = np.dot(x, x) + 1e-6
            self.w[axis] += self.mu * err_axis * x / norm_x2

            # --- Gain from shared data ---
            gain_key = f"params.FIR_GAIN_{axis}"
            try:
                gain = float(self.data[gain_key][0])
            except KeyError:
                gain = 0.  # default gain if not set
                log.warning(f"Gain key {gain_key} not found in shared data, using 0.0")
                
            gain = np.clip(gain, 0.0, 1.0)

            u = gain * u_raw
            u = float(np.clip(u, -self.u_max, self.u_max))

            outputs[axis] = u

            # --- Telemetry ---
            prefix = f"FIR.short.{self.name}.{axis}"
            self.data[f"{prefix}.u_raw"][0] = u_raw
            self.data[f"{prefix}.u"][0] = u
            self.data[f"{prefix}.w_norm"][0] = float(np.linalg.norm(self.w[axis]))

        return outputs
