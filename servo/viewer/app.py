import json
from pathlib import Path
import logging
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from .tabs.main import MainTab
from .tabs.config import ConfigTab
from .tabs.debug import DebugTab
from .tabs.buffers import BuffersTab
from .utils.lut import build_lut
from .widgets.casebar import CaseBar  # convenience
from .. import core, config, utils
from ..fsm import ServoState, NexlineState, WorkerState


log = logging.getLogger(__name__)

class Viewer(core.Worker):
    """Tk-based scientific viewer (refactored)."""

    def __init__(self, data, events):
        super().__init__(data, events)
        self.root = tk.Tk()
        self.root.title('IRCamera Viewer')
        self.root.geometry('1280x900')
        self.root.minsize(900, 600)
        self.load_window_geometry()
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.root.option_add('*Font', ('Noto Sans', 14))
        self.root.tk.call('tk', 'scaling', 1.0)
        style = ttk.Style(self.root)
        style.configure('.', font=('Noto Sans', 14))
        style.theme_use('clam')

        # danger (red): STOP
        style.configure('Danger.TButton', foreground='white', background='#d32f2f')
        style.map('Danger.TButton',
                  background=[('disabled', '#ef9a9a'),
                              ('active',   '#b71c1c'),
                              ('!disabled','#d32f2f')])
        # warn (orange): NORMALIZE, Reset TIP-TILT
        style.configure('Warn.TButton', foreground='black', background='#f39c12')
        style.map('Warn.TButton',
                  background=[('disabled', '#f8c471'),
                              ('active',   '#d68910'),
                              ('!disabled','#f39c12')])
        
        # state
        self._close_loop = False
        self.show_normalized = True
        self.current_lut_name = 'magma'
        self.current_lut = build_lut(self.current_lut_name)

        # layout
        root_main = ttk.Frame(self.root)
        root_main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        left = ttk.Frame(root_main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._right_col = ttk.Frame(root_main)
        self._right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        # tabs
        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.tab_main = ttk.Frame(self.notebook)
        self.tab_config = ttk.Frame(self.notebook)
        self.tab_buffers = ttk.Frame(self.notebook)
        self.tab_debug = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text='Main')
        self.notebook.add(self.tab_config, text='Config')
        self.notebook.add(self.tab_buffers, text='Buffers')
        self.notebook.add(self.tab_debug, text='Debug')

        # build tabs
        self.main_tab = MainTab(self.tab_main, self)
        self.config_tab = ConfigTab(self.tab_config, self)
        self.buffers_tab = BuffersTab(self.tab_buffers, self)
        self.debug_tab = DebugTab(self.tab_debug, self)

        # right column
        self._build_right_column()

        # toolbar
        self._build_toolbar(root_main)

        # info bar
        self._build_infobar()

        # timers
        self.root.bind('<<Shutdown>>', lambda e: self._really_stop())
        self.root.after(100, self.refresh)

    # --- small LED widget helpers ---
    def _make_led(self, parent, diameter=12, on_color="#2ecc71", off_color="#c0392b"):
        """Return (frame, canvas, circle_id, on_color, off_color)."""
        frm = ttk.Frame(parent)
        cv = tk.Canvas(frm, width=diameter, height=diameter, highlightthickness=0, bd=0)
        cv.pack()
        cid = cv.create_oval(1, 1, diameter-1, diameter-1, outline="", fill=off_color)
        return (frm, cv, cid, on_color, off_color)

    def _set_led(self, led_tuple, is_on: bool):
        """Update LED fill color."""
        try:
            frm, cv, cid, on_color, off_color = led_tuple
            cv.itemconfig(cid, fill=(on_color if is_on else off_color))
        except Exception:
            pass
        
    # toolbar
    def _build_toolbar(self, root_main):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, pady=3, before=root_main)

        ttk.Label(toolbar, text='LUT:').pack(side=tk.LEFT, padx=5)
        self.lut_var = tk.StringVar(value=self.current_lut_name)
        choices = ['gray','viridis','inferno','magma','plasma','cividis','turbo','rainbow']
        cb = ttk.Combobox(toolbar, textvariable=self.lut_var, values=choices, state='readonly', width=10)
        cb.pack(side=tk.LEFT, padx=5)
        cb.bind('<<ComboboxSelected>>', self.on_lut_changed)

        self.shownorm_btn = ttk.Button(toolbar, text='Show Un-normalized', command=self.toggle_normalized)
        self.shownorm_btn.pack(side=tk.LEFT, padx=10)

        ttk.Button(toolbar, text='Reset Zoom', command=self.main_tab.reset_zoom).pack(side=tk.LEFT, padx=10)

        # STOP (red)
        self.stop_btn = ttk.Button(toolbar, text='STOP', style='Danger.TButton',
                                   command=lambda: self._set_event('Servo.stop'))
        self.stop_btn.pack(side=tk.RIGHT, padx=10)

        # CLOSE/OPEN LOOP (toggle by state)
        self.close_loop_btn = ttk.Button(toolbar, text='CLOSE LOOP', command=self.toggle_close_loop)
        self.close_loop_btn.pack(side=tk.RIGHT, padx=10)

        # NORMALIZE (orange)
        self.normalize_btn = ttk.Button(toolbar, text='NORMALIZE', style='Warn.TButton',
                                        command=lambda: self._set_event('Servo.normalize'))
        self.normalize_btn.pack(side=tk.RIGHT, padx=10)

        self.move_to_opd_btn = ttk.Button(toolbar, text='MOVE to OPD',
                                          command=lambda: self._set_event('Servo.move_to_opd'))
        self.move_to_opd_btn.pack(side=tk.RIGHT, padx=10)

        # Reset TIP-TILT (orange)
        self.reset_tiptilt_btn = ttk.Button(toolbar, text='Reset TIP-TILT', style='Warn.TButton',
                                            command=self._reset_tiptilt)
        self.reset_tiptilt_btn.pack(side=tk.RIGHT, padx=10)

        self.roi_mode_btn = ttk.Button(toolbar, text='ROI MODE', command=self.toggle_roi_mode)
        self.roi_mode_btn.pack(side=tk.RIGHT, padx=10)

        self.reset_zpd_btn = ttk.Button(toolbar, text='Reset ZPD',
                                        command=lambda: self._set_event('Servo.reset_zpd'))
        self.reset_zpd_btn.pack(side=tk.RIGHT, padx=10)

        # initial sync (enable/disable according to current state)
        self._update_commands_enabled()
        
    # right column
    def _build_right_column(self):
        status = ttk.LabelFrame(self._right_col, text='Status', padding=10)
        status.pack(fill=tk.X, expand=False, pady=15)
        self.status_var = tk.StringVar(value='Idle')
        ttk.Label(status, textvariable=self.status_var).pack(anchor='w')
        # --- LOST LED row ---
        lost_row = ttk.Frame(status)
        lost_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(lost_row, text='LOST:').pack(side=tk.LEFT)

        self.led_lost = self._make_led(
            lost_row, diameter=12, on_color="#c0392b", off_color="#2ecc71")
        self.led_lost[0].pack(side=tk.LEFT, padx=6)
        
        self._build_piezos(self._right_col)
        # Profile inputs
        prof = ttk.LabelFrame(self._right_col, text='Profile size', padding=10)
        prof.pack(fill=tk.X, expand=False, pady=15)
        top = ttk.Frame(prof); top.pack(side='top', anchor='nw', fill='x', pady=(0, 6))
        colL = ttk.Frame(top); colL.pack(side='left', anchor='nw', padx=(0,10))
        self.profile_len = tk.IntVar(value=self.data['IRCamera.profile_len'][0])
        ttk.Label(colL, text='length').pack(side='top', anchor='w')
        e = tk.Entry(colL, textvariable=self.profile_len, width=8)
        e.pack(side='top', anchor='w', pady=4)
        e.bind('<Return>', self.on_len_changed)
        colW = ttk.Frame(top); colW.pack(side='left', anchor='nw', padx=(0,10))
        self.profile_width = tk.IntVar(value=self.data['IRCamera.profile_width'][0])
        ttk.Label(colW, text='width').pack(side='top', anchor='w')
        e2 = tk.Entry(colW, textvariable=self.profile_width, width=8)
        e2.pack(anchor='w', pady=4)
        e2.bind('<Return>', self.on_width_changed)
        bot = ttk.Frame(prof); bot.pack(side='top', anchor='nw', fill='x')
        self.opd_target = tk.DoubleVar(value=self.data['Servo.opd_target'][0])
        ttk.Label(bot, text='OPD target').pack(side='left', anchor='w', padx=(0,6))
        e3 = tk.Entry(bot, textvariable=self.opd_target, width=8)
        e3.pack(side='left', anchor='w', pady=4)
        e3.bind('<Return>', self.on_opd_target_changed)

        # ROI preview
        self.roi_wrap = ttk.LabelFrame(self._right_col, text='ROI', padding=6)
        self.roi_wrap.pack(fill=tk.X, expand=False, pady=10)
        self.roi_wrap.configure(width=360, height=360)
        self.roi_wrap.pack_propagate(False)
        self.roi_canvas = tk.Canvas(self.roi_wrap, bg='black')
        self.roi_canvas.pack(fill=tk.BOTH, expand=True)
        self.roi_image_id = None
        # bootstrap
        try:
            n = int(self.data['IRCamera.profile_len'][0])
        except Exception:
            n = 32
        self.roi_shape = (n, n)
        self.roi_image = np.zeros(self.roi_shape, dtype=np.float32)
        self.roi_wrap.bind('<Configure>', lambda e: self._resize_roi())
        self.root.after_idle(self._resize_roi)

    def _resize_roi(self):
        if getattr(self, '_roi_resize_pending', False):
            return
        self._roi_resize_pending = True
        def do():
            try:
                target = max(160, min(self.roi_wrap.winfo_width(), self.roi_wrap.winfo_height()))
                cw, ch = self.roi_canvas.winfo_width(), self.roi_canvas.winfo_height()
                if target != cw or target != ch:
                    self.roi_canvas.config(width=target, height=target)
            finally:
                self._roi_resize_pending = False
        self.root.after_idle(do)

    # info bar
    def _build_infobar(self):
        info = ttk.Frame(self.root, height=30)
        info.pack(side=tk.BOTTOM, fill=tk.X, pady=3)
        info.pack_propagate(False)
        ttk.Label(info, text='Position:').pack(side=tk.LEFT)
        self.pos_var = tk.StringVar(value='x = —, y = —')
        ttk.Label(info, textvariable=self.pos_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(info, text='Value:').pack(side=tk.LEFT)
        self.val_var = tk.StringVar(value='—')
        ttk.Label(info, textvariable=self.val_var).pack(side=tk.LEFT, padx=5)

    # run
    def run(self):
        try:
            self.root.mainloop()
        except Exception:
            pass

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
        try:
            self.root.event_generate('<<Shutdown>>', when='tail')
        except Exception:
            pass

    def _really_stop(self):
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # events
    def _set_event(self, key: str):
        try:
            self.events[key].set()
        except Exception as e:
            log.error(f"event set '{key}': {e}")

    # toolbar actions
    def on_lut_changed(self, *_):
        name = self.lut_var.get()
        self.current_lut = build_lut(name)
        self.current_lut_name = name
        self.main_tab.render()

    def toggle_normalized(self):
        self.show_normalized = not self.show_normalized
        self.shownorm_btn.config(text='Show Un-normalized' if self.show_normalized else 'Show Normalized')

    def toggle_close_loop(self):
        """Send the only valid loop-transition from current state."""
        st = self._servo_state()
        if st == ServoState.RUNNING:
            self._set_event('Servo.close_loop')
        elif st == ServoState.STAY_AT_OPD:
            self._set_event('Servo.open_loop')
        # labels & enablement will be updated by next refresh/_update_status()
        
    def toggle_roi_mode(self):
        self.main_tab._roi_mode = not self.main_tab._roi_mode
        if self.main_tab._roi_mode:
            self._set_event('Servo.roi_mode')
        else:
            self._set_event('Servo.full_frame_mode')

    def _reset_tiptilt(self):
        try:
            self.data['Servo.tip_target'][0] = float(np.mean(self.data['IRCamera.tip_buffer'][:config.IRCAM_BUFFER_SIZE]))
            self.data['Servo.tilt_target'][0] = float(np.mean(self.data['IRCamera.tilt_buffer'][:config.IRCAM_BUFFER_SIZE]))
        except Exception:
            pass

    # handlers (right)
    def on_len_changed(self, *_):
        if self.main_tab._roi_mode:
            self.profile_len.set(self.data['IRCamera.profile_len'][0])
            return
        n = int(self.profile_len.get())
        n = utils.validate_roi_len(n)
        self.data['IRCamera.profile_len'][0] = int(n)
        try:
            self.main_tab.hbar.set_count(n)
            self.main_tab.vbar.set_count(n)
            self.main_tab.update_profiles()
        except Exception:
            pass

    def on_width_changed(self, *_):
        w = int(self.profile_width.get())
        if w % 2:
            w += 1
            self.profile_width.set(w)
        self.data['IRCamera.profile_width'][0] = int(w)

    def on_opd_target_changed(self, *_):
        try:
            self.data['Servo.opd_target'][0] = float(self.opd_target.get())
        except Exception:
            pass

    # window state
    def _state_path(self):
        try:
            base = Path.home() / '.config' / 'scientific_viewer'
            base.mkdir(parents=True, exist_ok=True)
            return base / 'viewer_state.json'
        except Exception:
            return Path('viewer_state.json')

    def load_window_geometry(self):
        p = self._state_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
        except Exception:
            return
        geom = data.get('geometry')
        state = data.get('state', 'normal')
        if isinstance(geom, str) and 'x' in geom and '+' in geom:
            sw = self.root.winfo_screenwidth(); sh = self.root.winfo_screenheight()
            try:
                size_part, pos_part = geom.split('+', 1)
                w_str, h_str = size_part.split('x', 1)
                x_str, y_str = pos_part.split('+', 1)
                w = max(200, min(int(w_str), sw))
                h = max(150, min(int(h_str), sh))
                x = max(0, min(int(x_str), sw-50))
                y = max(0, min(int(y_str), sh-50))
                self.root.geometry(f"{w}x{h}+{x}+{y}")
            except Exception:
                self.root.geometry(geom)
        def _apply_state():
            try:
                if state == 'zoomed':
                    self.root.state('zoomed')
                else:
                    self.root.state('normal')
            except Exception:
                pass
        self.root.after_idle(_apply_state)

    def save_window_geometry(self):
        try:
            state = self.root.state()
        except Exception:
            state = 'normal'
        geom = self.root.geometry()
        p = self._state_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({'geometry': geom, 'state': state}, indent=2))
        except Exception:
            Path('viewer_state.json').write_text(json.dumps({'geometry': geom, 'state': state}, indent=2))

    def on_close(self):
        try:
            if self.stop_event:
                self.stop_event.set()
        except Exception:
            pass
        try:
            self.save_window_geometry()
        except Exception:
            pass
        try:
            self.root.after(0, self._really_stop)
        except Exception:
            pass

    # status
    def _update_status(self):
        try:
            s_servo = ServoState(self.data['Servo.state'][0]).name
        except Exception:
            s_servo = '—'
        try:
            s_nx = NexlineState(self.data['Nexline.state'][0]).name
        except Exception:
            s_nx = '—'
        try:
            s_ir = WorkerState(self.data['IRCamera.state'][0]).name
        except Exception:
            s_ir = '—'
        try:
            s_daq = WorkerState(self.data['DAQ.state'][0]).name
        except Exception:
            s_daq = '—'
        try:
            s_sc = WorkerState(self.data['SerialComm.state'][0]).name
        except Exception:
            s_sc = '—'
        try:
            mean_opd = float(self.data['IRCamera.mean_opd'][0])
        except Exception:
            mean_opd = float('nan')
        # std from data if available (fallback to NaN)
        try:
            std_opd = float(self.data['IRCamera.std_opd'][0])
        except Exception:
            std_opd = float('nan')
        try:
            fps = 1.0/float(self.data['IRCamera.median_sampling_time'][0])
        except Exception:
            fps = float('nan')
        try:
            drops = int(self.data['IRCamera.lost_frames'][0])
        except Exception:
            drops = 0

        try:
            opd_target = float(self.data['Servo.opd_target'][0])
        except Exception:
            opd_target = float('nan')

            
        s = (f"mean OPD: {utils.fwformat(mean_opd, 9)} nm | (std): {utils.fwformat(std_opd, 5, decimals=1)} nm",
             f"OPD target: {utils.fwformat(opd_target, 9)} nm",
             f"fps: {utils.fwformat(fps/1e3, 6, decimals=1)} kHz | frame drops: {drops}/{config.IRCAM_BUFFER_SIZE}")
        self.status_var.set('\n'.join(s))

        # sync toolbar according to current state
        self._update_commands_enabled()

        try:
            is_lost_val = int(self.data['Servo.is_lost'][0])
        except Exception:
            is_lost_val = 0
        self._set_led(self.led_lost, bool(is_lost_val))

    # right: piezos
    def _build_piezos(self, parent):
        frame = ttk.LabelFrame(parent, text='Piezos', padding=10)
        frame.pack(fill=tk.Y, expand=False)
        row = ttk.Frame(frame); row.pack(fill=tk.Y, expand=False)
        self.var_opd = tk.DoubleVar(value=0.0)
        self.var_da1 = tk.DoubleVar(value=0.0)
        self.var_da2 = tk.DoubleVar(value=0.0)
        try:
            init = self.data['DAQ.piezos_level'][:3]
            if len(init) >= 3:
                self.var_opd.set(float(init[0]))
                self.var_da1.set(float(init[1]))
                self.var_da2.set(float(init[2]))
        except Exception:
            pass
        self.piezo_step = 0.01
        def _nudge(var: tk.DoubleVar, scl: tk.Scale, delta: float):
            try: v = float(var.get()) + float(delta)
            except Exception: v = 0.0
            vmin = min(float(scl['from']), float(scl['to']))
            vmax = max(float(scl['from']), float(scl['to']))
            v = max(vmin, min(v, vmax))
            var.set(v)
            self._write_piezos()
        def _col(label, var, col):
            colf = ttk.Frame(row); colf.grid(row=0, column=col, padx=10, pady=5)
            ttk.Label(colf, text=label).pack(pady=(0,6))
            scl = tk.Scale(colf, from_=10.0, to=0.0, variable=var, orient=tk.VERTICAL,
                           showvalue=True, resolution=0.01, length=220,
                           command=self._on_piezos_change)
            scl.pack()
            btns = ttk.Frame(colf); btns.pack(pady=(6,0))
            ttk.Button(btns, text='-', width=2,
                       command=lambda: _nudge(var, scl, -self.piezo_step)).pack(side=tk.LEFT, padx=(0,4))
            ttk.Button(btns, text='+', width=2,
                       command=lambda: _nudge(var, scl, +self.piezo_step)).pack(side=tk.LEFT)
            return scl
        self.scale_opd = _col('OPD', self.var_opd, 0)
        self.scale_da1 = _col('DA-1', self.var_da1, 1)
        self.scale_da2 = _col('DA-2', self.var_da2, 2)
        self._write_piezos()

    def _write_piezos(self):
        vals = [float(self.var_opd.get()), float(self.var_da1.get()), float(self.var_da2.get())]
        self.data['DAQ.piezos_level'][:] = vals

    def _on_piezos_change(self, *_):
        self._write_piezos()

    # --- state helpers ---
    def _servo_state(self):
        """Return ServoState or None."""
        try:
            return ServoState(self.data['Servo.state'][0])
        except Exception:
            return None

    def _sync_close_loop_button(self, state):
        """Set button label to the next valid action based on current state."""
        if state == ServoState.STAY_AT_OPD:
            self.close_loop_btn.config(text='OPEN LOOP')
        else:
            # default -> if RUNNING, we show 'CLOSE LOOP'; otherwise keep this label too
            self.close_loop_btn.config(text='CLOSE LOOP')

    def _set_enabled(self, widget, enabled: bool):
        try:
            if enabled:
                widget.state(['!disabled'])
            else:
                widget.state(['disabled'])
        except Exception:
            pass

    def _update_commands_enabled(self):
        """Enable/disable toolbar buttons according to Servo.state."""
        st = self._servo_state()
        
        # STOP: disabled when loop is closed (STAY_AT_OPD)
        self._set_enabled(self.stop_btn, st != ServoState.STAY_AT_OPD)
        
        # CLOSE/OPEN LOOP enabled only in RUNNING or STAY_AT_OPD
        can_loop_toggle = st in (ServoState.RUNNING, ServoState.STAY_AT_OPD)
        self._set_enabled(self.close_loop_btn, bool(can_loop_toggle))
        self._sync_close_loop_button(st)

        # NORMALIZE only in RUNNING or STAY_AT_OPD
        self._set_enabled(self.normalize_btn, st == (ServoState.RUNNING or ServoState.STAY_AT_OPD))

        # MOVE to OPD only in STAY_AT_OPD
        self._set_enabled(self.move_to_opd_btn, st == ServoState.STAY_AT_OPD)

        # Reset TIP-TILT in RUNNING or STAY_AT_OPD
        self._set_enabled(self.reset_tiptilt_btn, st in (ServoState.RUNNING, ServoState.STAY_AT_OPD))

        # ROI MODE (ROI/FF) transitions defined from RUNNING
        self._set_enabled(self.roi_mode_btn, st == ServoState.RUNNING)

        # Reset ZPD only in RUNNING
        self._set_enabled(self.reset_zpd_btn, st == ServoState.RUNNING)
        
    # refresh loop
    def refresh(self):
        if self.stop_event and self.stop_event.is_set():
            self.stop()
            return
        # frame
        try:
            prev_w, prev_h = getattr(self.main_tab, 'img_w', None), getattr(self.main_tab, 'img_h', None)
            self.main_tab.img_w = int(self.data['IRCamera.frame_dimx'][0])
            self.main_tab.img_h = int(self.data['IRCamera.frame_dimy'][0])
            raw = self.data['IRCamera.last_frame'][:self.data['IRCamera.frame_size'][0]]
            self.main_tab.frame = np.array(raw).reshape((self.main_tab.img_w, self.main_tab.img_h))
            if (prev_w, prev_h) != (self.main_tab.img_w, self.main_tab.img_h):
                self.main_tab.fitted = False
                self.main_tab.fit_image()
        except Exception as e:
            log.error(f'frame refresh: {e}')
        # roi
        try:
            n = int(self.data['IRCamera.profile_len'][0])
            self.roi_shape = (n, n)
            if self.main_tab._roi_mode:
                new_frame = self.main_tab.frame
            else:
                try:
                    raw = self.data['IRCamera.roi'][:n**2]
                    new_frame = np.array(raw).reshape(self.roi_shape).T
                except Exception:
                    new_frame = np.zeros(self.roi_shape).T
            if self.show_normalized:
                raw_min = np.array(self.data['Servo.roinorm_min'][:n**2]).reshape(self.roi_shape).T
                raw_max = np.array(self.data['Servo.roinorm_max'][:n**2]).reshape(self.roi_shape).T
                new_frame = np.clip((new_frame - raw_min) / (raw_max - raw_min), 0, 1)
            self.roi_image = new_frame
            # draw ROI thumbnail
            stretched = self.main_tab._stretch(self.roi_image.T)
            rgb = self.main_tab._apply_lut(stretched)
            w = max(1, self.roi_canvas.winfo_width()); h = max(1, self.roi_canvas.winfo_height())
            pil = Image.fromarray(rgb, 'RGB').resize((w, h), Image.NEAREST)
            self._tk_roi = ImageTk.PhotoImage(pil)
            if self.roi_image_id is None:
                self.roi_image_id = self.roi_canvas.create_image(0, 0, anchor='nw', image=self._tk_roi)
            else:
                self.roi_canvas.itemconfig(self.roi_image_id, image=self._tk_roi)
                self.roi_canvas.coords(self.roi_image_id, 0, 0)
        except Exception as e:
            log.error(f'roi refresh: {e}')
        # draw
        try:
            self.main_tab.render()
        except Exception:
            pass
        # profiles + status
        try:
            self.main_tab.update_profiles()
            self._update_status()
        except Exception:
            pass
        # debug inspector
        try:
            self.debug_tab.update()
        except Exception:
            pass
        # dynamic buffers
        try:
            self.buffers_tab.update()
        except Exception:
            pass
        # piezos (actual)
        try:
            levels = self.data['DAQ.piezos_level_actual'][:3]
            if len(levels) >= 3:
                self.var_opd.set(float(levels[0]))
                self.var_da1.set(float(levels[1]))
                self.var_da2.set(float(levels[2]))
        except Exception:
            pass
        # schedule
        if not (self.stop_event and self.stop_event.is_set()):
            self.root.after(100, self.refresh)
            
