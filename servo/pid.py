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
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# Small generic MLP for contextual bias
# ============================================================
class _ContextualBiasNN(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Configuration
# ============================================================
@dataclass
class NNTrackingConfig:
    controller_id: str

    hidden: int = 32
    #nn_gain: float = 0.2
    #nn_clip: float = 1.0
    i_scale: float = 0.2
    learning_rate: float = 1e-3
    enable_training: bool = True
    device: str = "cpu"


# ============================================================
# Neural Regulator (generic, tracking + velocity ready)
# ============================================================
class NeuralRegulator:
    """
    Generic PID + NN regulator.

    u = u_pid + u_nn

    The NN learns a CONTEXTUAL SLOW BIAS from multi-frequency tracker data.
    The exact tracker signals used depend on controller_id.
    """

    # --------------------------------------------------------
    # Mapping controller_id → tracker base key
    # --------------------------------------------------------
    _TRACKER_MAP = {
        "TRACK_OPD": "opd",
        "TRACK_DA1": "tip",
        "TRACK_DA2": "tilt",
        "TRACK_VEL_OPD": "velocity",
    }

    def __init__(self, pid_controller, cfg: NNTrackingConfig, shared_data):
        self.pid = pid_controller
        self.cfg = cfg
        self.data = shared_data
        self.device = torch.device(cfg.device)

        # --- Automatic normalization of pid_i ---
        self.pid_i_abs_ema = None
        
        # Time constant for normalization (seconds)
        self.norm_tau = 5.0   # adaptation time (~5 s, safe)
        self.dt = 0.01        # control loop period (100 Hz)
        
        # Bounds to keep things safe
        self.i_scale_min = 0.0002
        self.i_scale_max = 20.0

        if cfg.controller_id not in self._TRACKER_MAP:
            raise ValueError(f"Unknown controller_id: {cfg.controller_id}")

        self.tracker_key = self._TRACKER_MAP[cfg.controller_id]

        # Features:
        # e_fast, e_mid, e_slow, trend_slow, hf_resid, pid_i
        self.input_dim = 6
        self.net = _ContextualBiasNN(self.input_dim, hidden=cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)

        # Persistence
        base_dir = Path.home() / ".local" / "state" / "servo"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = base_dir / f"nn_{cfg.controller_id.lower()}.pt"

        self._load()

    # --------------------------------------------------------
    # PID compatibility
    # --------------------------------------------------------
    def reset(self):
        self.pid.reset()

    def update_coeffs(self):
        if hasattr(self.pid, "update_coeffs"):
            self.pid.update_coeffs()
        else:
            log.warning("PID controller does not support dynamic coefficient updates")

    def update_config(self, **kwargs):
        if hasattr(self.pid, "update_config"):
            self.pid.update_config(**kwargs)
        else:
            log.warning("PID controller does not support dynamic config updates")
            
            #self.update_coeffs()
        #self.net.train()  # ensure we're in training mode for any config changes

    # --------------------------------------------------------
    # Tracker access helpers
    # --------------------------------------------------------
    def _get_tracker(self, hz: int):
        return float(self.data[f"Tracker.{self.tracker_key}_{hz}"][0])

    # --------------------------------------------------------
    # Main update
    # --------------------------------------------------------
    def update(self, control, setpoint, measurement):
        # --- PID (authoritative) ---
        u_pid = self.pid.update(
            control=control,
            setpoint=setpoint,
            measurement=measurement,
        )

        # --- Read tracker (multi-frequency) ---
        try:
            x_fast = self._get_tracker(100)
            x_mid  = self._get_tracker(10)
            x_slow = self._get_tracker(1)
        except Exception:
            return u_pid  # tracker not ready

        # --- Build contextual features ---
        e_fast = x_fast - setpoint
        e_mid  = x_mid  - setpoint
        e_slow = x_slow - setpoint

        trend_slow = x_mid - x_slow
        hf_resid   = x_fast - x_mid

        pid_i = getattr(self.pid, "_i", 0.0)

        x = np.array(
            [e_fast, e_mid, e_slow, trend_slow, hf_resid, pid_i],
            dtype=np.float32,
        )

        x_t = torch.from_numpy(x).to(self.device).unsqueeze(0)

        # --- NN forward ---
        raw = self.net(x_t)
        u_nn = torch.tanh(raw)[0, 0]

        #u_nn = self.cfg.nn_gain * torch.clamp(
        #    u_nn, -self.cfg.nn_clip, self.cfg.nn_clip
        #)

        u = float(u_pid) + u_nn.detach().item()

        u = np.clip(float(u), float(self.pid.cfg.out_min), float(self.pid.cfg.out_max))

        # --- Online learning ---
        if self.cfg.enable_training:
            i_scale = self._update_i_scale(pid_i)
            target = -pid_i / i_scale
            target_t = torch.tensor(target, device=self.device, dtype=torch.float32)

            loss = (u_nn - target_t).pow(2)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        # --- Telemetry ---
        prefix = f"NN.{self.cfg.controller_id}"
        self.data[f"{prefix}.u_nn"][0] = float(u_nn.detach())
        self.data[f"{prefix}.pid_i"][0] = float(pid_i)
        self.data[f"{prefix}.u_pid"][0] = float(u_pid)
        self.data[f"{prefix}.i_scale"][0] = float(i_scale)
        self.data[f"{prefix}.pid_i_abs_ema"][0] = float(self.pid_i_abs_ema)
        return u

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------
    def save(self):
        torch.save(
            {"state_dict": self.net.state_dict()},
            self.model_path,
        )

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            payload = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
            self.net.load_state_dict(payload["state_dict"])
        except Exception as e:
            print(f"[NeuralRegulator] Failed to load {self.model_path}: {e}")

    def _update_i_scale(self, pid_i: float) -> float:
        """
        Update automatic normalization scale for pid_i.
        Uses EMA of |pid_i| with safety bounds.
        """
        abs_i = abs(pid_i)

        if self.pid_i_abs_ema is None:
            self.pid_i_abs_ema = abs_i
        else:
            alpha = self.dt / (self.norm_tau + self.dt)
            self.pid_i_abs_ema = (
                (1.0 - alpha) * self.pid_i_abs_ema
                + alpha * abs_i
            )

        # Clamp to safe bounds
        i_scale = min(
            max(self.pid_i_abs_ema, self.i_scale_min),
            self.i_scale_max,
        )
        return i_scale
