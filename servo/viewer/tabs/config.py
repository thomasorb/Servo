import tkinter as tk
from tkinter import ttk
import numpy as np

import logging

log = logging.getLogger(__name__)

class ConfigTab:
    """PID config panels."""

    def __init__(self, parent, viewer):
        self.viewer = viewer
        self.root = ttk.Frame(parent)
        self.root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._build_ui()

    def _read_float(self, key, default=0.0):
        try:
            return float(self.viewer.data[key][0])
        except Exception:
            return float(default)

    def _read_from_array(self, key, idx, default=0.0):
        try:
            return float(self.viewer.data[key][idx])
        except Exception:
            return float(default)

    def _build_ui(self):
        # PID OPD
        pid_opd = ttk.LabelFrame(self.root, text='PID OPD', padding=10)
        pid_opd.pack(side='top', anchor='nw', fill='x', pady=(0, 8))
        self.pid_opd_p = tk.DoubleVar(value=self._read_float('Servo.pid_P',
                                 self._read_from_array('Servo.PID_OPD', 0, 0.0)))
        self.pid_opd_i = tk.DoubleVar(value=self._read_float('Servo.pid_I',
                                 self._read_from_array('Servo.PID_OPD', 1, 0.0)))
        self.pid_opd_d = tk.DoubleVar(value=self._read_float('Servo.pid_D',
                                 self._read_from_array('Servo.PID_OPD', 2, 0.0)))
        for col, (label, var, key) in enumerate((
            ('P', self.pid_opd_p, 'Servo.pid_P'),
            ('I', self.pid_opd_i, 'Servo.pid_I'),
            ('D', self.pid_opd_d, 'Servo.pid_D'),
        )):
            ttk.Label(pid_opd, text=label).grid(row=0, column=col, sticky='w', padx=(0,8), pady=(0,4))
            e = ttk.Entry(pid_opd, textvariable=var, width=10)
            e.grid(row=1, column=col, sticky='w', padx=(0,16), pady=(0,6))
            def _mk_bind(entry, v, k):
                def _on_return(_evt=None):
                    try:
                        val = float(v.get())
                        if k in self.viewer.data:
                            self.viewer.data[k][0] = float(val)
                        else:
                            self.viewer.data[k] = [float(val)]
                    except Exception:
                        pass
                entry.bind('<Return>', _on_return)
            _mk_bind(e, var, key)
        ttk.Button(pid_opd, text='Apply OPD', command=self._apply_pid_opd).grid(row=1, column=3, sticky='w')

        # PID DA
        pid_da = ttk.LabelFrame(self.root, text='PID DA', padding=10)
        pid_da.pack(side='top', anchor='nw', fill='x')
        self.pid_da_p = tk.DoubleVar(value=self._read_from_array('Servo.PID_DA', 0, 0.0))
        self.pid_da_i = tk.DoubleVar(value=self._read_from_array('Servo.PID_DA', 1, 0.0))
        self.pid_da_d = tk.DoubleVar(value=self._read_from_array('Servo.PID_DA', 2, 0.0))
        for col, (label, var, idx) in enumerate((
            ('P', self.pid_da_p, 0),
            ('I', self.pid_da_i, 1),
            ('D', self.pid_da_d, 2),
        )):
            ttk.Label(pid_da, text=label).grid(row=0, column=col, sticky='w', padx=(0,8), pady=(0,4))
            e = ttk.Entry(pid_da, textvariable=var, width=10)
            e.grid(row=1, column=col, sticky='w', padx=(0,16), pady=(0,6))
            def _mk_bind_da(entry, v, index):
                def _on_return(_evt=None):
                    try:
                        try:
                            arr = self.viewer.data['Servo.PID_DA']
                        except Exception:
                            arr = None
                        if arr is None or len(arr) < 3:
                            self.viewer.data['Servo.PID_DA'] = np.array([0.0,0.0,0.0], dtype=self.viewer.config.DATA_DTYPE)
                        self.viewer.data['Servo.PID_DA'][index] = float(v.get())
                    except Exception:
                        pass
                entry.bind('<Return>', _on_return)
            _mk_bind_da(e, var, idx)
        ttk.Button(pid_da, text='Apply DA', command=self._apply_pid_da).grid(row=1, column=3, sticky='w')

        # PID NEX
        pid_nex = ttk.LabelFrame(self.root, text='PID NEXLINE', padding=10)
        pid_nex.pack(side='top', anchor='nw', fill='x')
        self.pid_nex_p = tk.DoubleVar(value=self._read_from_array('Servo.PID_NEXLINE', 0, 0.0))
        self.pid_nex_i = tk.DoubleVar(value=self._read_from_array('Servo.PID_NEXLINE', 1, 0.0))
        self.pid_nex_d = tk.DoubleVar(value=self._read_from_array('Servo.PID_NEXLINE', 2, 0.0))
        for col, (label, var, idx) in enumerate((
            ('P', self.pid_nex_p, 0),
            ('I', self.pid_nex_i, 1),
            ('D', self.pid_nex_d, 2),
        )):
            ttk.Label(pid_nex, text=label).grid(row=0, column=col, sticky='w', padx=(0,8), pady=(0,4))
            e = ttk.Entry(pid_nex, textvariable=var, width=10)
            e.grid(row=1, column=col, sticky='w', padx=(0,16), pady=(0,6))
            def _mk_bind_nex(entry, v, index):
                def _on_return(_evt=None):
                    try:
                        try:
                            arr = self.viewer.nexta['Servo.PID_NEXLINE']
                        except Exception:
                            arr = None
                        if arr is None or len(arr) < 3:
                            self.viewer.data['Servo.PID_NEXLINE'] = np.array([0.0,0.0,0.0], dtype=self.viewer.config.DATA_DTYPE)
                        self.viewer.data['Servo.PID_NEXLINE'][index] = float(v.get())
                    except Exception:
                        pass
                entry.bind('<Return>', _on_return)
            _mk_bind_nex(e, var, idx)
        ttk.Button(pid_nex, text='Apply NEXLINE', command=self._apply_pid_nex).grid(row=1, column=3, sticky='w')

        
        # Profile inputs
        prof = ttk.LabelFrame(self.root, text='Profile size', padding=10)
        prof.pack(side='top', anchor='nw', fill='x')
        top = ttk.Frame(prof); top.pack(side='top', anchor='nw', fill='x', pady=(0, 6))
        colL = ttk.Frame(top); colL.pack(side='left', anchor='nw', padx=(0,10))
        self.profile_len = tk.IntVar(value=self.viewer.data['IRCamera.profile_len'][0])
        ttk.Label(colL, text='length').pack(side='top', anchor='w')
        e = tk.Entry(colL, textvariable=self.profile_len, width=8)
        e.pack(side='top', anchor='w', pady=4)
        e.bind('<Return>', self.on_len_changed)
        colW = ttk.Frame(top); colW.pack(side='left', anchor='nw', padx=(0,10))
        self.profile_width = tk.IntVar(value=self.viewer.data['IRCamera.profile_width'][0])
        ttk.Label(colW, text='width').pack(side='top', anchor='w')
        e2 = tk.Entry(colW, textvariable=self.profile_width, width=8)
        e2.pack(anchor='w', pady=4)
        e2.bind('<Return>', self.on_width_changed)


    def _apply_pid_opd(self):
        try:
            self.viewer.data['Servo.PID_OPD'][:3] = np.array(
                [self.pid_opd_p.get(), self.pid_opd_i.get(), self.pid_opd_d.get()]
            ).astype(self.viewer.config.DATA_DTYPE)
        except Exception as e:
            log.error(f'error when applying OPD PID: {e}')

    def _apply_pid_da(self):
        try:
            self.viewer.data['Servo.PID_DA'][:3] = np.array(
                [self.pid_da_p.get(), self.pid_da_i.get(), self.pid_da_d.get()]
            ).astype(self.viewer.config.DATA_DTYPE)
        except Exception as e:
            log.error(f'error when applying OPD DA: {e}')
            
    def _apply_pid_nex(self):
        try:
            self.viewer.data['Servo.PID_NEXLINE'][:3] = np.array(
                [self.pid_nex_p.get(), self.pid_nex_i.get(), self.pid_nex_d.get()]
            ).astype(self.viewer.config.DATA_DTYPE)
        except Exception as e:
            log.error(f'error when applying OPD NEX: {e}')
        
    def on_len_changed(self, *_):
        if self.viewer.main_tab._roi_mode:
            self.profile_len.set(self.viewer.data['IRCamera.profile_len'][0])
            return
        n = int(self.profile_len.get())
        n = utils.validate_roi_len(n)
        self.viewer.data['IRCamera.profile_len'][0] = int(n)
        try:
            self.viewer.main_tab.hbar.set_count(n)
            self.viewer.main_tab.vbar.set_count(n)
            self.viewer.main_tab.update_profiles()
        except Exception:
            pass

    def on_width_changed(self, *_):
        w = int(self.profile_width.get())
        if w % 2:
            w += 1
            self.profile_width.set(w)
        self.viewer.data['IRCamera.profile_width'][0] = int(w)
