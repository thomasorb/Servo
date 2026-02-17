import numpy as np
import time
import re
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import traceback
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from pathlib import Path
import logging
from collections import deque
from . import core
from . import config
from . import utils
from .fsm import ServoState, NexlineState, WorkerState
log = logging.getLogger(__name__)
import tkinter as tk
from tkinter import ttk


class CaseBar(tk.Frame):
    """
    Dynamic case bar.
    - States: 0, 1, 2 (none, side, center)
    - Left click: cycle state
    - Right click: context menu
    - on_change(index:int, state:str, all_states:list[str]) if provided
    """
    DEFAULT_COLORS = (
        {"fill": "#f6f6f6", "outline": "#bdbdbd"},
        {"fill": "#1677ff", "outline": "#0b59c8"},
        {"fill": "#2ecc71", "outline": "#239b56"},
    )
    ORDER = [0, 1, 2]

    def __init__(self, parent, count=8, height=28,
                 gap=2, padding=8, min_cell_w=6,
                 colors=None, on_change=None, states=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.count = int(count)
        self.height = int(height)
        self.gap = int(gap)
        self.padding = int(padding)
        self.min_cell_w = int(min_cell_w)
        self.colors = (colors or self.DEFAULT_COLORS)
        self.on_change = on_change

        if states is None:
            self.states = [0] * self.count
        else:
            self.set_states(states)

        self.canvas = tk.Canvas(self, height=self.height,
                                bg=self["bg"] if "bg" in self.keys() else "white",
                                highlightthickness=0)
        self.canvas.pack(fill="x", expand=True)

        self.menu = tk.Menu(self, tearoff=False)
        for st in self.ORDER:
            self.menu.add_command(label=f"Mettre: {st}", command=lambda s=st: self._ctx_set(s))

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", lambda e: self._hover_cursor(e))

        self._ctx_index = None
        self._layout = []  # (x0,y0,x1,y1)
        self._items = []   # ids
        self._redraw_all()

    # --- Public API ---
    def set_count(self, n: int):
        n = int(n)
        if n < 0: return
        if n > len(self.states):
            self.states.extend([0] * (n - len(self.states)))
        else:
            self.states = self.states[:n]
        self.count = n
        self._redraw_all()
        self._notify(-1, None)

    def set_state(self, index: int, state: int):
        if 0 <= index < self.count and state in self.ORDER:
            self.states[index] = state
            self._paint_cell(index)
            self._notify(index, state)

    def set_states(self, states):
        if len(states) > self.count:
            self.states = list(states)[:self.count]
        else:
            self.states = list(states) + [0] * (self.count - len(states))
        if hasattr(self, 'canvas'):
            self._redraw_all()
        self._notify(-1, None)

    def get_states(self):
        return list(self.states)

    # --- Drawing ---
    def _on_resize(self, event=None):
        self._redraw_all()

    def _compute_layout(self):
        W = max(1, self.canvas.winfo_width())
        H = self.height
        self.canvas.config(height=H)
        inner_w = max(1, W - 2*self.padding)
        if self.count <= 0:
            return [], 0
        cell_w = (inner_w - self.gap*(self.count-1)) / self.count
        cell_w = max(self.min_cell_w, int(cell_w))
        total_w = cell_w*self.count + self.gap*(self.count-1)
        x = self.padding  # left aligned (no centering)
        y0 = self.padding//2
        y1 = H - self.padding//2
        layout = []
        for i in range(self.count):
            x0 = x + i*(cell_w + self.gap)
            x1 = x0 + cell_w
            layout.append((x0, y0, x1, y1))
        return layout, total_w

    def _redraw_all(self):
        self.canvas.delete("all")
        self._items.clear()
        self._layout, _ = self._compute_layout()
        for i, bbox in enumerate(self._layout):
            item = self._draw_one(i, bbox, self.states[i])
            self._items.append(item)

    def _draw_one(self, idx, bbox, state):
        x0, y0, x1, y1 = bbox
        c = self.colors[state]
        return self.canvas.create_rectangle(
            x0, y0, x1, y1,
            fill=c["fill"], outline=c["outline"], width=1,
            tags=("cell", f"cell-{idx}")
        )

    def _paint_cell(self, idx):
        if not (0 <= idx < len(self._items)): return
        item = self._items[idx]
        st = self.states[idx]
        c = self.colors[st]
        self.canvas.itemconfig(item, fill=c["fill"], outline=c["outline"])

    # --- Interaction ---
    def _index_from_x(self, x):
        for i, (x0, y0, x1, y1) in enumerate(self._layout):
            if x0 <= x <= x1:
                return i
        return None

    def _hover_cursor(self, event):
        idx = self._index_from_x(event.x)
        self.canvas.config(cursor="hand2" if idx is not None else "")

    def _on_left_click(self, event):
        idx = self._index_from_x(event.x)
        if idx is None: return
        curr = self.states[idx]
        nxt = self.ORDER[(self.ORDER.index(curr) + 1) % len(self.ORDER)]
        self.states[idx] = nxt
        self._paint_cell(idx)
        self._notify(idx, nxt)

    def _on_right_click(self, event):
        idx = self._index_from_x(event.x)
        if idx is None: return
        self._ctx_index = idx
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _ctx_set(self, state):
        idx = self._ctx_index
        if idx is None: return
        self.set_state(idx, state)
        self._ctx_index = None

    def _notify(self, index, state):
        if callable(self.on_change):
            self.on_change(index, state, self.get_states())


class Viewer(core.Worker):
    """
    Tkinter-based scientific viewer.
    """

    # ---------------------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------------------
    def __init__(self, data, events):
        super().__init__(data, events)
        self.root = tk.Tk()
        self.root.title("IRCamera Viewer")
        self.root.geometry("1280x900")
        self.root.minsize(900, 600)

        # Restore old window geometry
        self.load_window_geometry()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root.option_add("*Font", ("Noto Sans", 14))
        self.root.tk.call("tk", "scaling", 1.0)
        style = ttk.Style(self.root)
        style.configure(".", font=("Noto Sans", 14))
        style.theme_use("clam")


        style.map("State.TButton",
                  background=[("disabled", "#bdc3c7"), ("active", "#27ae60"), ("!disabled", "#2ecc71")],
                  foreground=[("disabled", "#7f8c8d"), ("!disabled", "white")])
        #btn.state(["disabled"])  # pour désactiver
        #btn.state(["!disabled"]) # pour réactiver

        # -----------------------------------------------------------------
        # Root layout: left = Notebook (tabs), right = persistent controls
        # -----------------------------------------------------------------
        root_main = ttk.Frame(self.root)
        root_main.pack(fill=tk.BOTH, expand=True)

        # Left stack holds the Notebook; switching tabs only changes this area
        left_stack = ttk.Frame(root_main)
        left_stack.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- TABS (Notebook) ---------------------------------------------
        self.notebook = ttk.Notebook(left_stack)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Main tab (image + profiles live on the left area)
        self.tab_main = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="Main")

        # Config tab
        self.tab_config = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_config, text="Config")

        # Buffers tab
        self.tab_buffers = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_buffers, text="Buffers")

        # Debug tab
        self.tab_debug = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_debug, text="Debug")

        # Build the 'Buffers' tab content
        self._build_buffers_tab()

        # -----------------------------------------------------------------
        # Persistent right column: Status, Piezos, Profile size, ROI preview
        # -----------------------------------------------------------------
        self._right_col = ttk.Frame(root_main)
        self._right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        status_frame = ttk.LabelFrame(self._right_col, text="Status", padding=10)
        status_frame.pack(fill=tk.X, expand=False, pady=15)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w")

        # Piezos frame
        self._build_piezos_controls(self._right_col)

        # -----------------------------------------------------------------
        # Profile inputs (right side, under Piezos) - persistent
        # -----------------------------------------------------------------
        profile_len_frame = ttk.LabelFrame(self._right_col, text="Profile size", padding=10)
        profile_len_frame.pack(fill=tk.X, expand=False, pady=15)

        top_row = ttk.Frame(profile_len_frame)
        top_row.pack(side="top", anchor="nw", fill="x", pady=(0, 6))

        length_col = ttk.Frame(top_row)
        length_col.pack(side="left", anchor="nw", padx=(0, 10))
        self.profile_len = tk.IntVar(value=self.data['IRCamera.profile_len'][0])
        ttk.Label(length_col, text="length").pack(side='top', anchor="w")
        e = tk.Entry(length_col, textvariable=self.profile_len, width=8)
        e.pack(side='top', anchor="w", pady=4)
        e.bind("<Return>", self.on_len_changed)

        width_col = ttk.Frame(top_row)
        width_col.pack(side="left", anchor="nw", padx=(0, 10))
        self.profile_width = tk.IntVar(value=self.data['IRCamera.profile_width'][0])
        ttk.Label(width_col, text="width").pack(side='top', anchor='w')
        e = tk.Entry(width_col, textvariable=self.profile_width, width=8)
        e.pack(anchor="w", pady=4)
        e.bind("<Return>", self.on_width_changed)

        bottom_row = ttk.Frame(profile_len_frame)
        bottom_row.pack(side="top", anchor="nw", fill="x")
        self.opd_target = tk.DoubleVar(value=self.data['Servo.opd_target'][0])
        ttk.Label(bottom_row, text="OPD target").pack(side="left", anchor="w", padx=(0, 6))
        e = tk.Entry(bottom_row, textvariable=self.opd_target, width=8)
        e.pack(side="left", anchor="w", pady=4)
        e.bind("<Return>", self.on_opd_target_changed)

        # ROI preview canvas now lives in the persistent right column
        self.roi_canvas = tk.Canvas(self._right_col, bg="black")
        self.roi_canvas.pack(expand=False)
        self.roi_image_id = None

        # -----------------------------------------------------------------
        # Build "Main" tab LEFT content only (image + profiles)
        # -----------------------------------------------------------------
        left_col_main = ttk.Frame(self.tab_main)
        left_col_main.pack(fill=tk.BOTH, expand=True)

        # Image canvas at the top
        canvas_frame = ttk.Frame(left_col_main)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Profiles under the image
        profiles_frame = ttk.Frame(left_col_main)
        profiles_frame.pack(fill=tk.X, expand=False, pady=10)
        self._profiles_frame = profiles_frame

        # ====================== Config tab content ===========================
        cfg_wrap = ttk.Frame(self.tab_config)
        cfg_wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- PID OPD frame ---------------------------------------------------
        pid_opd_frame = ttk.LabelFrame(cfg_wrap, text="PID OPD", padding=10)
        pid_opd_frame.pack(side="top", anchor="nw", fill="x", pady=(0, 8))

        # Helpers to read defaults from data
        def _read_float(key, default=0.0):
            try:
                val = self.data[key][0]
                return float(val)
            except Exception:
                return float(default)

        def _read_from_array(key, index, default=0.0):
            try:
                return float(self.data[key][index])
            except Exception:
                return float(default)

        # PID OPD variables (first try per-element keys, fallback to Servo.PID array)
        self.pid_opd_p = tk.DoubleVar(value=_read_float('Servo.pid_P', _read_from_array('Servo.PID_OPD', 0, 0.0)))
        self.pid_opd_i = tk.DoubleVar(value=_read_float('Servo.pid_I', _read_from_array('Servo.PID_OPD', 1, 0.0)))
        self.pid_opd_d = tk.DoubleVar(value=_read_float('Servo.pid_D', _read_from_array('Servo.PID_OPD', 2, 0.0)))

        # Layout for PID OPD: labels + entries in a row (grid)
        for col, (label, var, key) in enumerate((
            ("P", self.pid_opd_p, 'Servo.pid_P'),
            ("I", self.pid_opd_i, 'Servo.pid_I'),
            ("D", self.pid_opd_d, 'Servo.pid_D'),
        )):
            ttk.Label(pid_opd_frame, text=label).grid(row=0, column=col, sticky="w", padx=(0, 8), pady=(0, 4))
            e = ttk.Entry(pid_opd_frame, textvariable=var, width=10)
            e.grid(row=1, column=col, sticky="w", padx=(0, 16), pady=(0, 6))

            # On Return: update individual OPD PID scalars in shared data
            def _mk_bind(entry, v, k):
                def _on_return(_evt=None):
                    try:
                        val = float(v.get())
                        if k in self.data:
                            self.data[k][0] = float(val)
                        else:
                            self.data[k] = [float(val)]
                        log.info(f"{k} updated: {val}")
                    except Exception as ex:
                        log.error(f"Error updating {k}: {ex}")
                entry.bind("<Return>", _on_return)
            _mk_bind(e, var, key)

        ttk.Button(pid_opd_frame, text="Apply OPD", command=self._apply_pid_opd).grid(
            row=1, column=3, sticky="w", padx=(0, 0)
        )

        # --- PID DA frame ----------------------------------------------------
        pid_da_frame = ttk.LabelFrame(cfg_wrap, text="PID DA", padding=10)
        pid_da_frame.pack(side="top", anchor="nw", fill="x")

        self.pid_da_p = tk.DoubleVar(value=_read_from_array('Servo.PID_DA', 0, 0.0))
        self.pid_da_i = tk.DoubleVar(value=_read_from_array('Servo.PID_DA', 1, 0.0))
        self.pid_da_d = tk.DoubleVar(value=_read_from_array('Servo.PID_DA', 2, 0.0))

        for col, (label, var, idx) in enumerate((
            ("P", self.pid_da_p, 0),
            ("I", self.pid_da_i, 1),
            ("D", self.pid_da_d, 2),
        )):
            ttk.Label(pid_da_frame, text=label).grid(row=0, column=col, sticky="w", padx=(0, 8), pady=(0, 4))
            e = ttk.Entry(pid_da_frame, textvariable=var, width=10)
            e.grid(row=1, column=col, sticky="w", padx=(0, 16), pady=(0, 6))

            # On Return: directly update self.data['Servo_PID_DA'][idx]
            def _mk_bind_da(entry, v, index):
                def _on_return(_evt=None):
                    try:
                        # Ensure array exists and has length >= 3
                        try:
                            arr = self.data['Servo.PID_DA']
                        except Exception:
                            arr = None
                        if arr is None or len(arr) < 3:
                            self.data['Servo.PID_DA'] = np.array([0.0, 0.0, 0.0], dtype=config.DATA_DTYPE)
                        # Write value
                        self.data['Servo.PID_DA'][index] = float(v.get())
                        log.info(f"Servo.PID_DA[{index}] updated: {float(v.get())}")
                    except Exception as ex:
                        log.error(f"Error updating Servo.PID_DA[{index}]: {ex}")
                entry.bind("<Return>", _on_return)
            _mk_bind_da(e, var, idx)

        ttk.Button(pid_da_frame, text="Apply DA", command=self._apply_pid_da).grid(
            row=1, column=3, sticky="w", padx=(0, 0)
        )

        # ====================== Debug tab content ============================
        debug_wrap = ttk.Frame(self.tab_debug)
        debug_wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        states_frame = ttk.LabelFrame(debug_wrap, text="States", padding=10)
        states_frame.pack(side="top", anchor="nw", fill="x")

        # Debug Data Inspector (choose keys from self.data and auto-refresh)
        self._debug_watch_keys = []  # list of keys currently watched
        self._debug_tree_items = {}  # map: key -> tree item id
        self._debug_last_update = 0.0  # throttle UI refresh
        self._build_debug_inspector(debug_wrap)

        self.state_servo = tk.IntVar(value=self.data['Servo.state'][0])
        self.state_nexline = tk.IntVar(value=self.data['Nexline.state'][0])
        self.state_ircam = tk.IntVar(value=self.data['IRCamera.state'][0])
        self.state_daq = tk.IntVar(value=self.data['DAQ.state'][0])
        self.state_serialcomm = tk.IntVar(value=self.data['DAQ.state'][0])
        
        for col, (label, var, key) in enumerate((
                ("Servo", self.state_servo, 'Servo.state'),
                ("Nexline", self.state_nexline, 'Nexline.state'),
                ("IRCamera", self.state_ircam, 'IRCamera.state'),
                ("DAQ", self.state_daq, 'DAQ.state'),
                ("SerialComm", self.state_serialcomm, 'SerialComm.state'),)):
            ttk.Label(states_frame, text=label).grid(row=0, column=col, sticky="w",
                                                     padx=(0, 8), pady=(0, 4))
            e = ttk.Label(states_frame, textvariable=var, width=10)
            e.grid(row=1, column=col, sticky="w", padx=(0, 16), pady=(0, 6))

        # -----------------------------------------------------------------
        # Image info / state
        # -----------------------------------------------------------------
        self.img_w = self.data['IRCamera.frame_dimx'][0]
        self.img_h = self.data['IRCamera.frame_dimy'][0]
        self.frame = self.generate_frame()

        # -----------------------------------------------------------------
        # Brightness / Contrast
        # -----------------------------------------------------------------
        self.brightness = 0.0
        self.contrast = 1.0
        self.last_bc_mouse = None

        # -----------------------------------------------------------------
        # LUT
        # -----------------------------------------------------------------
        self.current_lut_name = "magma"
        self.lut = self.build_lut_magma()

        # -----------------------------------------------------------------
        # Global toolbar (persistent across all tabs)
        # -----------------------------------------------------------------
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, pady=3)

        ttk.Label(toolbar, text="LUT:").pack(side=tk.LEFT, padx=5)
        self.lut_var = tk.StringVar(value="magma")
        lut_choices = ["gray", "viridis", "inferno", "magma", "plasma", "cividis", "turbo", "rainbow"]
        self.lut_menu = ttk.Combobox(toolbar, textvariable=self.lut_var,
                                     values=lut_choices, state="readonly", width=10)
        self.lut_menu.pack(side=tk.LEFT, padx=5)
        self.lut_menu.bind("<<ComboboxSelected>>", self.on_lut_changed)

        self.shownorm_btn = ttk.Button(toolbar, text="Show Un-normalized",
                                       command=self.toggle_normalized)
        self.shownorm_btn.pack(side=tk.LEFT, padx=10)
        self._show_normalized = True

        self.reset_btn = ttk.Button(toolbar, text="Reset Zoom", command=self.reset_zoom)
        self.reset_btn.pack(side=tk.LEFT, padx=10)

        # (Optional) Clear ROI button kept commented
        # self.clear_roi_btn = ttk.Button(toolbar, text="Clear ROI", command=self.clear_roi)
        # self.clear_roi_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = ttk.Button(toolbar, text="STOP", command=self.stop_servo)
        self.stop_btn.pack(side=tk.RIGHT, padx=10)

        self.close_loop_btn = ttk.Button(toolbar, text="CLOSE LOOP", command=self.toggle_close_loop)
        self.close_loop_btn.pack(side=tk.RIGHT, padx=10)
        self._close_loop = False

        self.normalize_btn = ttk.Button(toolbar, text="NORMALIZE", command=self.normalize)
        self.normalize_btn.pack(side=tk.RIGHT, padx=10)

        self.move_to_opd_btn = ttk.Button(toolbar, text="MOVE to OPD", command=self.move_to_opd)
        self.move_to_opd_btn.pack(side=tk.RIGHT, padx=10)

        self.reset_tiptilt_btn = ttk.Button(toolbar, text="Reset TIP-TILT",
                                            command=self.reset_tiptilt)
        self.reset_tiptilt_btn.pack(side=tk.RIGHT, padx=10)

        self.roi_mode_btn = ttk.Button(toolbar, text="ROI MODE", command=self.toggle_roi_mode)
        self.roi_mode_btn.pack(side=tk.RIGHT, padx=10)
        self._roi_mode = False

        self.reset_zpd_btn = ttk.Button(toolbar, text="Reset ZPD", command=self.reset_zpd)
        self.reset_zpd_btn.pack(side=tk.RIGHT, padx=10)

        # -----------------------------------------------------------------
        # Global info bar (persistent across all tabs)
        # -----------------------------------------------------------------
        info = ttk.Frame(self.root, height=30)
        info.pack(fill=tk.X, pady=3)
        info.pack_propagate(False)

        ttk.Label(info, text="Position:").pack(side=tk.LEFT)
        self.pos_var = tk.StringVar(value="x = —, y = —")
        ttk.Label(info, textvariable=self.pos_var).pack(side=tk.LEFT, padx=5)

        ttk.Label(info, text="Value:").pack(side=tk.LEFT)
        self.val_var = tk.StringVar(value="—")
        ttk.Label(info, textvariable=self.val_var).pack(side=tk.LEFT, padx=5)

        # -----------------------------------------------------------------
        # Profiles Panel (horizontal & vertical real-time profiles)
        # -----------------------------------------------------------------
        # ===== Profiles - H row (profile + casebar on the left, ellipse_x on the right)
        h_row = ttk.Frame(self._profiles_frame)
        h_row.pack(fill=tk.X, expand=False, pady=6)

        # left column (H profile + H casebar)
        h_left = ttk.Frame(h_row)
        h_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.fig_h = plt.Figure(figsize=(6, 1.8), dpi=90)
        self.ax_h = self.fig_h.add_subplot(111)
        self.ax_h.set_title("Horizontal profile")
        self.ax_h.set_ylim(0, 1)
        self.canvas_h = FigureCanvasTkAgg(self.fig_h, master=h_left)
        self.canvas_h.get_tk_widget().pack(fill=tk.X, padx=8, pady=4)

        self.hbar = CaseBar(
            h_left,
            count=self.profile_len.get(),
            height=28,
            states=self.data['Servo.pixels_x'][:].astype(int).tolist(),
            on_change=self._on_hbar_change
        )
        self.hbar.pack(fill=tk.X, padx=8, pady=(0, 8))

        # right column (ellipse_x)
        h_right = ttk.Frame(h_row, width=240)
        h_right.pack(side=tk.RIGHT, fill=tk.Y, padx=8)

        self.fig_ellx = plt.Figure(figsize=(3.0, 1.8), dpi=90)
        self.ax_ellx = self.fig_ellx.add_subplot(111)
        self.ax_ellx.set_title("ellipse_x")
        self.canvas_ellx = FigureCanvasTkAgg(self.fig_ellx, master=h_right)
        self.canvas_ellx.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ===== Profiles - V row (profile + casebar on the left, ellipse_y on the right)
        v_row = ttk.Frame(self._profiles_frame)
        v_row.pack(fill=tk.X, expand=False, pady=6)

        # left column (V profile + V casebar)
        v_left = ttk.Frame(v_row)
        v_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.fig_v = plt.Figure(figsize=(6, 1.8), dpi=90)
        self.ax_v = self.fig_v.add_subplot(111)
        self.ax_v.set_title("Vertical profile")
        self.ax_v.set_ylim(0, 1)
        self.canvas_v = FigureCanvasTkAgg(self.fig_v, master=v_left)
        self.canvas_v.get_tk_widget().pack(fill=tk.X, padx=8, pady=4)

        self.vbar = CaseBar(
            v_left,
            count=self.profile_len.get(),
            height=28,
            states=self.data['Servo.pixels_y'][:].astype(int).tolist(),
            on_change=self._on_vbar_change
        )
        self.vbar.pack(fill=tk.X, padx=8, pady=(0, 8))

        # right column (ellipse_y)
        v_right = ttk.Frame(v_row, width=240)
        v_right.pack(side=tk.RIGHT, fill=tk.Y, padx=8)

        self.fig_elly = plt.Figure(figsize=(3.0, 1.8), dpi=90)
        self.ax_elly = self.fig_elly.add_subplot(111)
        self.ax_elly.set_title("ellipse_y")
        self.canvas_elly = FigureCanvasTkAgg(self.fig_elly, master=v_right)
        self.canvas_elly.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.hlevels_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)
        self.vlevels_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)

        # Tk image holder
        self.root.after_idle(lambda: self.on_resize(None))
        self.tk_image = None
        self.image_id = None

        # -----------------------------------------------------------------
        # Zoom & Pan
        # -----------------------------------------------------------------
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.fitted = False

        # Overlays
        self.cross_h = self.canvas.create_line(0, 0, 0, 0, fill="yellow")
        self.cross_v = self.canvas.create_line(0, 0, 0, 0, fill="yellow")
        self.canvas.tag_raise(self.cross_h)
        self.canvas.tag_raise(self.cross_v)
        self.points = []

        # Mouse bindings
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonPress-1>", self.on_left_press_dispatch)
        self.canvas.bind("<B1-Motion>", self.on_left_motion_dispatch)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release_dispatch)
        self.canvas.bind("<ButtonPress-3>", self.bc_start)
        self.canvas.bind("<B3-Motion>", self.bc_drag)
        self.canvas.bind("<ButtonRelease-3>", self.bc_end)
        self.canvas.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, 1.1))
        self.canvas.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 0.9))

        # SHIFT tool (none)
        # CTRL tool (ROI)
        self.roi_rect_id = None
        self._normal_click_start = None

        self.root.bind("<<Shutdown>>", lambda e: self._really_stop())

        # init values
        self.add_marker(self.data['IRCamera.profile_x'][0], self.data['IRCamera.profile_y'][0])

        # Start periodic refresh
        self.root.after(100, self.refresh)

        # events-dependent buffers initialized in _build_buffers_tab()
        # piezo levels updated in refresh()

    # ---------------------------------------------------------------------
    # PUBLIC CONTROL
    # ---------------------------------------------------------------------
    def run(self):
        """Main viewer loop. Tk exits cleanly when stop() schedules quit()."""
        try:
            self.root.mainloop()
        except Exception:
            pass

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
        try:
            # Send a Tk event that will run in the Tk thread
            self.root.event_generate("<<Shutdown>>", when="tail")
        except Exception:
            pass

    def _really_stop(self):
        """Executed INSIDE the Tk event loop."""
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _set_event(self, key: str):
        """
        Triggers (set) the event identified by `key` in self.events.
        Logs a warning if the event is missing or has no .set().
        """
        try:
            self.events[key].set()
        except Exception as e:
            log.error(f"Error at event set '{key}': {e}")

    # ---------------------------------------------------------------------
    # FRAME ACQUISITION
    # ---------------------------------------------------------------------
    def generate_frame(self):
        """Dummy initial frame."""
        return np.random.rand(self.img_h, self.img_w).astype(np.float32)

    # ---------------------------------------------------------------------
    # LUT BUILDERS (gray, viridis, inferno, etc.)
    # ---------------------------------------------------------------------
    def build_lut_gray(self):
        lut = np.linspace(0, 1, 256)
        lut = np.stack([lut, lut, lut], axis=1)
        return lut

    def build_lut_viridis(self):
        from matplotlib import cm
        return cm.get_cmap("viridis", 256)(np.linspace(0, 1, 256))[:, :3]

    def build_lut_inferno(self):
        from matplotlib import cm
        return cm.get_cmap("inferno", 256)(np.linspace(0, 1, 256))[:, :3]

    def build_lut_magma(self):
        from matplotlib import cm
        return cm.get_cmap("magma", 256)(np.linspace(0, 1, 256))[:, :3]

    def build_lut_plasma(self):
        from matplotlib import cm
        return cm.get_cmap("plasma", 256)(np.linspace(0, 1, 256))[:, :3]

    def build_lut_cividis(self):
        from matplotlib import cm
        return cm.get_cmap("cividis", 256)(np.linspace(0, 1, 256))[:, :3]

    def build_lut_turbo(self):
        from matplotlib import cm
        return cm.get_cmap("turbo", 256)(np.linspace(0, 1, 256))[:, :3]

    def build_lut_rainbow(self):
        from matplotlib import cm
        return cm.get_cmap("rainbow", 256)(np.linspace(0, 1, 256))[:, :3]

    def on_lut_changed(self, *_):
        """Apply the selected LUT name to the rendering pipeline."""
        name = self.lut_var.get()
        mapping = {
            "gray": self.build_lut_gray,
            "viridis": self.build_lut_viridis,
            "inferno": self.build_lut_inferno,
            "magma": self.build_lut_magma,
            "plasma": self.build_lut_plasma,
            "cividis": self.build_lut_cividis,
            "turbo": self.build_lut_turbo,
            "rainbow": self.build_lut_rainbow,
        }
        func = mapping.get(name, self.build_lut_magma)
        self.lut = func()
        self.current_lut_name = name
        self.render()

    # ---------------------------------------------------------------------
    # BRIGHTNESS / CONTRAST
    # ---------------------------------------------------------------------
    def bc_start(self, event):
        self.last_bc_mouse = (event.x, event.y)

    def bc_drag(self, event):
        if not self.last_bc_mouse:
            return
        dx = event.x - self.last_bc_mouse[0]
        dy = event.y - self.last_bc_mouse[1]
        self.brightness += dy * 0.002
        self.contrast *= (1 + dx * 0.002)
        self.contrast = max(0.05, min(self.contrast, 5))
        self.last_bc_mouse = (event.x, event.y)
        self.render()

    def bc_end(self, *_):
        self.last_bc_mouse = None

    # ---------------------------------------------------------------------
    # NORMALIZATION + APPLY LUT
    # ---------------------------------------------------------------------
    def stretch_frame(self, frame):
        fmin = frame.min()
        fmax = frame.max()
        if fmax <= fmin:
            return np.zeros_like(frame)
        norm = (frame - fmin) / (fmax - fmin)
        norm = (norm - 0.5) * self.contrast + 0.5 + self.brightness
        return np.clip(norm, 0, 1)

    def apply_lut(self, stretched):
        idx = (stretched * 255).astype(np.uint8)
        rgb = self.lut[idx]
        return (rgb * 255).astype(np.uint8)

    # ---------------------------------------------------------------------
    # RENDERING
    # ---------------------------------------------------------------------
    def render(self):
        # main image
        stretched = self.stretch_frame(self.frame.T)
        rgb = self.apply_lut(stretched)
        disp_w = int(self.img_w * self.scale)
        disp_h = int(self.img_h * self.scale)
        pil_img = Image.fromarray(rgb, "RGB").resize((disp_w, disp_h), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        if self.image_id is None:
            self.image_id = self.canvas.create_image(self.offset_x, self.offset_y,
                                                     anchor="nw", image=self.tk_image)
        else:
            self.canvas.itemconfig(self.image_id, image=self.tk_image)
        self.canvas.coords(self.image_id, self.offset_x, self.offset_y)

        # roi image
        if hasattr(self, 'roi_image'):
            stretched = self.stretch_frame(self.roi_image.T)
            rgb = self.apply_lut(stretched)
            w = max(1, self.roi_canvas.winfo_width())
            h = max(1, self.roi_canvas.winfo_height())
            pil_img = Image.fromarray(rgb, "RGB").resize((w, h), Image.NEAREST)
            self.tk_roi = ImageTk.PhotoImage(pil_img)
            if self.roi_image_id is None:
                self.roi_image_id = self.roi_canvas.create_image(0, 0, anchor="nw",
                                                                 image=self.tk_roi)
            else:
                self.roi_canvas.itemconfig(self.roi_image_id, image=self.tk_roi)
            self.roi_canvas.coords(self.roi_image_id, 0, 0)

        # Markers
        r = 4
        for item_id, ix, iy in self.points:
            cx, cy = self.image_to_canvas(ix, iy)
            self.canvas.coords(item_id, cx-r, cy-r, cx+r, cy+r)
            self.canvas.tag_raise(item_id)

        # ROI overlay
        hw = self.data['IRCamera.profile_len'][0]//2
        ix = self.data['IRCamera.profile_x'][0]
        iy = self.data['IRCamera.profile_y'][0]
        x0, y0, x1, y1 = ix-hw, iy-hw, ix+hw, iy+hw
        cx0, cy0 = self.image_to_canvas(x0, y0)
        cx1, cy1 = self.image_to_canvas(x1, y1)
        if self.roi_rect_id is None:
            self.roi_rect_id = self.canvas.create_rectangle(
                cx0, cy0, cx1, cy1, outline="#5FA8FF", width=2)
        else:
            self.canvas.coords(self.roi_rect_id, cx0, cy0, cx1, cy1)
        self.canvas.tag_raise(self.roi_rect_id)
        self.canvas.tag_raise(self.cross_h)
        self.canvas.tag_raise(self.cross_v)

    # ---------------------------------------------------------------------
    # ZOOM / PAN
    # ---------------------------------------------------------------------
    def fit_image_to_canvas(self):
        """Fit the image to the canvas size."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1 or self.img_w <= 0 or self.img_h <= 0:
            return
        self.scale = min(cw / self.img_w, ch / self.img_h)
        self.offset_x = (cw - self.img_w * self.scale) / 2
        self.offset_y = (ch - self.img_h * self.scale) / 2

        # ROI preview square on the persistent right column (same logic as before)
        size = max(1, min(self._right_col.winfo_width(), self._right_col.winfo_height()))
        self.roi_canvas.config(width=size, height=size)
        self.fitted = True
        self.render()

    def on_resize(self, event):
        if not self.fitted:
            self.fit_image_to_canvas()

    def reset_zoom(self):
        self.fitted = False
        self.fit_image_to_canvas()

    def zoom_at(self, cx, cy, factor):
        old = self.scale
        new = max(0.1, min(50.0, old * factor))
        if new == old:
            return
        ix = (cx - self.offset_x) / old
        iy = (cy - self.offset_y) / old
        self.scale = new
        self.offset_x = cx - ix * new
        self.offset_y = cy - iy * new
        self.render()

    # ---------------------------------------------------------------------
    # COORDINATE TRANSFORMS
    # ---------------------------------------------------------------------
    def canvas_to_image(self, cx, cy):
        ix = int((cx - self.offset_x) / self.scale)
        iy = int((cy - self.offset_y) / self.scale)
        return ix, iy

    def image_to_canvas(self, ix, iy):
        return ix*self.scale + self.offset_x, iy*self.scale + self.offset_y

    # ---------------------------------------------------------------------
    # MOUSE MOVE: crosshair + pixel readout
    # ---------------------------------------------------------------------
    def on_mouse_move(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y)
        if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
            self.pos_var.set(f"x = {ix}, y = {iy}")
            self.val_var.set(f"{float(self.frame[ix, iy]):.3f}")
        else:
            self.pos_var.set("x = —, y = —")
            self.val_var.set("—")
        self.canvas.coords(self.cross_h, 0, event.y,
                           self.canvas.winfo_width(), event.y)
        self.canvas.coords(self.cross_v, event.x, 0,
                           event.x, self.canvas.winfo_height())

    # ---------------------------------------------------------------------
    # LEFT CLICK: marker (normal mode)
    # ---------------------------------------------------------------------
    def on_left_press_dispatch(self, event):
        if event.state & 0x0001:  # SHIFT
            pass
        elif event.state & 0x0004:  # CTRL
            # ROI tool disabled here
            pass
        else:
            self._normal_click_start = (event.x, event.y)

    def on_left_motion_dispatch(self, event):
        if event.state & 0x0001:
            pass
        elif event.state & 0x0004:
            # ROI tool disabled here
            pass

    def add_marker(self, ix, iy):
        if self._roi_mode: return
        # Remove old markers
        for item, _, _ in self.points:
            self.canvas.delete(item)
        self.points.clear()
        # Add new marker
        cx, cy = self.image_to_canvas(ix, iy)
        r = 4
        p = self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                    fill="red", outline="")
        self.points.append((p, ix, iy))

    def on_left_release_dispatch(self, event):
        # Normal click → marker + CSV
        if self._normal_click_start:
            if self._roi_mode: return
            ix, iy = self.canvas_to_image(event.x, event.y)
            ix, iy = utils.validate_roi_position((ix, iy))
            if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
                self.add_marker(ix, iy)
                self.data['IRCamera.profile_x'][0] = int(ix)
                self.data['IRCamera.profile_y'][0] = int(iy)
            self._normal_click_start = None

    def on_len_changed(self, *args):
        if self._roi_mode:
            self.profile_len.set(self.data['IRCamera.profile_len'][0])
            log.warning('length cannot be changed in ROI mode')
            return
        profile_len = self.profile_len.get()
        profile_len = utils.validate_roi_len(profile_len)
        log.info(f"Profile length updated: {profile_len}")
        self.data['IRCamera.profile_len'][0] = int(self.profile_len.get())
        try:
            self.update_profiles()
        except Exception as e:
            log.error(f"Error updating profiles after length change: {e}")
            return
        self.hbar.set_count(profile_len)
        self.vbar.set_count(profile_len)

    def on_width_changed(self, *args):
        profile_width = self.profile_width.get()
        if profile_width % 2:
            profile_width += 1
            self.profile_width.set(profile_width)
            log.warning(f'profile_width must be even, changed to {profile_width}')
        log.info(f"Profile width updated: {profile_width}")
        self.data['IRCamera.profile_width'][0] = int(self.profile_width.get())

    def on_opd_target_changed(self, *args):
        opd_target = self.opd_target.get()
        log.info(f"OPD target updated: {opd_target}")
        self.data['Servo.opd_target'][0] = float(self.opd_target.get())

    def _on_hbar_change(self, index, state, all_states):
        try:
            profile_len = self.data['IRCamera.profile_len'][0]
            self.data["Servo.pixels_x"][:profile_len] = [int(s) for s in all_states]
        except Exception as e:
            log.error(f"Error updating pixels_x: {e}")

    def _on_vbar_change(self, index, state, all_states):
        try:
            profile_len = self.data['IRCamera.profile_len'][0]
            self.data["Servo.pixels_y"][:profile_len] = [int(s) for s in all_states]
        except Exception as e:
            log.error(f"Error updating pixels_y: {e}")

    # ---------------------------------------------------------------------
    # APPLY BUTTONS FOR PID FRAMES
    # ---------------------------------------------------------------------
    def _apply_pid_opd(self):
        """Apply OPD PID triplet to self.data['Servo.PID']."""
        try:
            self.data['Servo.PID'][:3] = np.array(
                [self.pid_opd_p.get(), self.pid_opd_i.get(), self.pid_opd_d.get()]
            ).astype(config.DATA_DTYPE)
            log.info("PID OPD parameters applied")
        except Exception as ex:
            log.error(f"Error applying PID OPD params: {ex}")

    def _apply_pid_da(self):
        """Apply DA PID triplet to self.data['Servo_PID_DA']."""
        try:
            self.data['Servo.PID_DA'][:3] = np.array(
                [self.pid_da_p.get(), self.pid_da_i.get(), self.pid_da_d.get()]
            ).astype(config.DATA_DTYPE)
            log.info("PID DA parameters applied")
        except Exception as ex:
            try:
                # Create if missing
                self.data['Servo.PID_DA'] = np.array(
                    [self.pid_da_p.get(), self.pid_da_i.get(), self.pid_da_d.get()],
                    dtype=config.DATA_DTYPE
                )
                log.info("PID DA parameters created and applied")
            except Exception as ex2:
                log.error(f"Error applying PID DA params: {ex2}")

    def toggle_normalized(self):
        if self._show_normalized:
            self._show_normalized = False
            self.shownorm_btn.config(text="Show Normalized")
        else:
            self._show_normalized = True
            self.shownorm_btn.config(text="Show Un-normalized")

    def toggle_close_loop(self):
        if self._close_loop:
            self._close_loop = False
            self.close_loop_btn.config(text="CLOSE LOOP")
            self._set_event('Servo.open_loop')
        else:
            self._close_loop = True
            self.close_loop_btn.config(text="OPEN LOOP")
            self._set_event('Servo.close_loop')

    def toggle_roi_mode(self):
        if self._roi_mode:
            self._roi_mode = False
            self.roi_mode_btn.config(text="ROI MODE")
            self._set_event('Servo.full_frame_mode')
        else:
            self._roi_mode = True
            self.roi_mode_btn.config(text="FF MODE")
            self._set_event('Servo.roi_mode')

    # ---------------------------------------------------------------------
    # MOVE TO OPD
    # ---------------------------------------------------------------------
    def move_to_opd(self):
        self._set_event('Servo.move_to_opd')

    # ---------------------------------------------------------------------
    # WINDOW GEOMETRY SAVE/RESTORE
    # ---------------------------------------------------------------------
    def _state_path(self):
        try:
            base = Path.home() / ".config" / "scientific_viewer"
            base.mkdir(parents=True, exist_ok=True)
            return base / "viewer_state.json"
        except Exception:
            return Path("viewer_state.json")

    def load_window_geometry(self):
        p = self._state_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
        except Exception:
            return
        geom = data.get("geometry")
        state = data.get("state", "normal")
        if isinstance(geom, str) and "x" in geom and "+" in geom:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            try:
                size_part, pos_part = geom.split("+", 1)
                w_str, h_str = size_part.split("x", 1)
                x_str, y_str = pos_part.split("+", 1)
                w = max(200, min(int(w_str), sw))
                h = max(150, min(int(h_str), sh))
                x = max(0, min(int(x_str), sw-50))
                y = max(0, min(int(y_str), sh-50))
                self.root.geometry(f"{w}x{h}+{x}+{y}")
            except Exception:
                self.root.geometry(geom)

        def _apply_state():
            try:
                if state == "zoomed":
                    self.root.state("zoomed")
                else:
                    self.root.state("normal")
            except Exception:
                pass
        self.root.after_idle(_apply_state)

    def save_window_geometry(self):
        try:
            state = self.root.state()
        except Exception:
            state = "normal"
        geom = self.root.geometry()
        save = {"geometry": geom, "state": state}
        p = self._state_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(save, indent=2))
        except Exception:
            Path("viewer_state.json").write_text(json.dumps(save, indent=2))

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

    def update_status(self):
        try: self.state_servo.set(ServoState(self.data['Servo.state'][0]).name)
        except Exception: pass
        try: self.state_nexline.set(NexlineState(self.data['Nexline.state'][0]).name)
        except Exception: pass
        try: self.state_ircam.set(WorkerState(self.data['IRCamera.state'][0]).name)
        except Exception: pass
        try: self.state_daq.set(WorkerState(self.data['DAQ.state'][0]).name)
        except Exception: pass
        try: self.state_serialcomm.set(WorkerState(self.data['SerialComm.state'][0]).name)
        except Exception: pass

        mean_opd = self.data['IRCamera.mean_opd'][0]
        std_opd = self.opd_std_buf[0] if self.opd_std_buf else np.nan
        fps = 1./self.data['IRCamera.median_sampling_time'][0]
        drops = self.data['IRCamera.lost_frames'][0]
        status_str_list = (f"mean OPD: {mean_opd:.0f} nm",
                           f" (std): {std_opd:.1f} nm",
                           f"fps: {fps/1e3:.3f} kHz",
                           f"frame drops: {drops}/{config.IRCAM_BUFFER_SIZE}")
        self.status_var.set('\n'.join(status_str_list))

    # ---------------------------------------------------------------------
    # PERIODIC REFRESH (MAIN IMAGE UPDATE LOOP)
    # ---------------------------------------------------------------------
    def refresh(self):
        if self.stop_event and self.stop_event.is_set():
            self.stop()
            return
        try:
            # detect image size change
            prev_w, prev_h = getattr(self, "img_w", None), getattr(self, "img_h", None)
            self.img_w = self.data['IRCamera.frame_dimx'][0]
            self.img_h = self.data['IRCamera.frame_dimy'][0]
            raw = self.data['IRCamera.last_frame'][:self.data['IRCamera.frame_size'][0]]
            new_frame = np.array(raw).reshape((self.img_w, self.img_h))
            self.frame = new_frame
            if (prev_w, prev_h) != (self.img_w, self.img_h):
                self.fitted = False
                self.fit_image_to_canvas()
        except Exception as e:
            log.error(f'error at frame refresh: {e}')

        try:
            # profile_len may be different from profile_len set in viewer
            profile_len = self.data['IRCamera.profile_len'][0]
            self.roi_shape = (profile_len, profile_len)
            if self._roi_mode:
                new_frame = self.frame
            else:
                raw = self.data['IRCamera.roi'][:profile_len**2]
                new_frame = np.array(raw).reshape(self.roi_shape).T
            if self._show_normalized:
                raw_min = self.data['Servo.roinorm_min'][:profile_len**2]
                raw_min = np.array(raw_min).reshape(self.roi_shape).T
                raw_max = self.data['Servo.roinorm_max'][:profile_len**2]
                raw_max = np.array(raw_max).reshape(self.roi_shape).T
                new_frame = np.clip((new_frame - raw_min) / (raw_max - raw_min), 0, 1)
            self.roi_image = new_frame
        except Exception as e:
            log.error(f'error at frame refresh: {e}')

        try:
            self.render()
        except Exception as e:
            log.error(f'error at rendering: {traceback.format_exc()}')

        # Update profiles and all
        try:
            self.update_profiles()
            self.update_status()
        except Exception as e:
            print(e)

        # Update Debug data inspector (throttled)
        try:
            self._update_debug_inspector()
        except Exception:
            pass

        try:
            self._update_buffers_tab()
        except Exception as e:
            log.error(f"buffers update error: {e}")

        if not (self.stop_event and self.stop_event.is_set()):
            self.root.after(100, self.refresh)

        # update piezo levels
        levels = self.data["DAQ.piezos_level_actual"][:3]
        if len(levels) >= 3:
            self.var_opd.set(float(levels[0]))
            self.var_da1.set(float(levels[1]))
            self.var_da2.set(float(levels[2]))

    # ---------------------------------------------------------------------
    # HORIZONTAL AND VERTICAL PROFILES
    # ---------------------------------------------------------------------
    def update_profiles(self):
        """
        Update horizontal & vertical profiles
        """
        profile_len = self.data['IRCamera.profile_len'][0]
        hlevels = self.data['IRCamera.hprofile_levels'][:profile_len]
        vlevels = self.data['IRCamera.vprofile_levels'][:profile_len]
        hlevels_pos = self.data['IRCamera.hprofile_levels_pos'][:profile_len]
        vlevels_pos = self.data['IRCamera.vprofile_levels_pos'][:profile_len]

        if self._show_normalized:
            horiz = self.data['IRCamera.hprofile_normalized'][:profile_len]
            vert = self.data['IRCamera.vprofile_normalized'][:profile_len]
        else:
            horiz = self.data['IRCamera.hprofile'][:profile_len]
            vert = self.data['IRCamera.vprofile'][:profile_len]

        def draw(canvas, fig, ax, dat, title, levels, levels_pos):
            ax.clear()
            if self._show_normalized:
                # Plot levels markers (with horizontal x-error)
                ax.errorbar(levels_pos, levels, xerr=5, yerr=0, color='0.7', label='levels',
                            marker='o', ls='None', capsize=0, elinewidth=4, markersize=0)
            # Profile line
            ax.plot(dat, color="tab:orange", label=title, marker='s')
            ax.axhline(0.75, color='tab:gray')
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.set_xlim(-0.75, len(dat)-0.25)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_position([0, 0, 1, 1])
            ax.legend(loc='upper right')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            canvas.draw()

        def draw_ellipse(canvas, fig, ax, levels):
            """
            Draws the level scatter history (center vs left/right) with faded tail.
            """
            ax.clear()

            def get_levels(index):
                return [lvl[index] for lvl in levels]

            levels_center = get_levels(1)
            levels_left = get_levels(0)
            levels_right = get_levels(2)
            n = min(config.VIEWER_ELLIPSE_DRAW_BUFFER_SIZE, len(levels_center))

            ax.scatter(levels_center[0], levels_left[0], color="tab:blue")
            ax.scatter(levels_center[0], levels_right[0], color="tab:orange")
            ax.scatter(levels_center[:n], levels_left[:n], color="tab:blue", alpha=0.2)
            ax.scatter(levels_center[:n], levels_right[:n], color="tab:orange", alpha=0.2)

            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_position([0, 0, 1, 1])
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.axvline(0, c='0.8')
            ax.axhline(0, c='0.8')
            ax.axvline(1, c='0.8')
            ax.axhline(1, c='0.8')
            ax.axvline(0.5, c='0.5')
            ax.axhline(0.5, c='0.5')
            canvas.draw()

        draw(self.canvas_h, self.fig_h, self.ax_h, horiz, 'h profile', hlevels, hlevels_pos)
        draw(self.canvas_v, self.fig_v, self.ax_v, vert, 'v profile', vlevels, vlevels_pos)

        # Draw ellipses
        self.hlevels_buf.appendleft(np.copy(hlevels))
        self.vlevels_buf.appendleft(np.copy(vlevels))
        draw_ellipse(self.canvas_ellx, self.fig_ellx, self.ax_ellx, self.hlevels_buf)
        draw_ellipse(self.canvas_elly, self.fig_elly, self.ax_elly, self.vlevels_buf)

        # feed buffers
        self.time_buf.appendleft(time.time())
        all_opd = self.data['IRCamera.mean_opd_buffer'][:config.SERVO_BUFFER_SIZE]
        self.opd_mean_buf.append(np.nanmean(all_opd))
        self.opd_std_buf.append(np.nanstd(all_opd))
        self.piezo_opd_buf.append(self.data['DAQ.piezos_level_actual'][0])
        self.piezo_da1_buf.append(self.data['DAQ.piezos_level_actual'][1])
        self.piezo_da2_buf.append(self.data['DAQ.piezos_level_actual'][2])

        self.index_longbuf += 1
        if self.index_longbuf > config.VIEWER_BUFFER_SIZE:
            self.time_longbuf.append(np.mean(self.time_buf))
            self.opd_std_longbuf.append(np.mean(self.opd_std_buf))
            self.index_longbuf = 0

    # ---------------------------------------------------------------------
    # BUFFERS TAB
    # ---------------------------------------------------------------------
    def _build_buffers_tab(self):
        """
        Creates 'Buffers' tab with 3 plots:
        - mean_opd (single line)
        - std_opd (single line)
        - piezos (3 lines: OPD, DA-1, DA-2)
        The Line2D objects are kept for ultra-fast updates.
        """
        self.time_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)
        self.opd_mean_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)
        self.opd_std_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)
        self.piezo_opd_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)
        self.piezo_da1_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)
        self.piezo_da2_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)

        self.index_longbuf = 0
        self.time_longbuf = deque()  # for full observing time
        self.opd_mean_longbuf = deque()
        self.opd_std_longbuf = deque()
        self.piezo_opd_longbuf = deque()
        self.piezo_da1_longbuf = deque()
        self.piezo_da2_longbuf = deque()

        # ---- Tab layout: 3 LabelFrames stacked vertically ----------------
        wrap = ttk.Frame(self.tab_buffers)
        wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ========== mean_opd ==============================================
        lf_mean = ttk.LabelFrame(wrap, text="mean OPD", padding=6)
        lf_mean.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.fig_mean = plt.Figure(figsize=(5, 1.8), dpi=100)
        self.ax_mean = self.fig_mean.add_subplot(111)
        self.ax_mean.set_title("mean OPD")
        self.ax_mean.set_ylabel("nm")
        self.ax_mean.grid(alpha=0.3)
        self.ax_mean.set_xticks([])
        for spine in self.ax_mean.spines.values():
            spine.set_visible(False)
        self.ax_mean.set_position([0.03, 0.03, 0.97, 0.97])
        self.canvas_mean = FigureCanvasTkAgg(self.fig_mean, master=lf_mean)
        self.canvas_mean.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        (self.line_mean,) = self.ax_mean.plot([], [], color="tab:blue", lw=1.5, label="mean_opd")
        self.ax_mean.legend(loc="upper right")
        self.ax_mean.set_xlim(0, config.VIEWER_BUFFER_SIZE)

        # ========== std_opd ===============================================
        lf_std = ttk.LabelFrame(wrap, text="std OPD", padding=6)
        lf_std.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.fig_std = plt.Figure(figsize=(5, 1.8), dpi=100)
        self.ax_std = self.fig_std.add_subplot(111)
        self.ax_std.set_title("std OPD")
        self.ax_std.set_ylabel("nm")
        self.ax_std.grid(alpha=0.3)
        self.canvas_std = FigureCanvasTkAgg(self.fig_std, master=lf_std)
        self.canvas_std.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        (self.line_std,) = self.ax_std.plot([], [], color="tab:orange", lw=1.5, label="std_opd")
        (self.line_std_long,) = self.ax_std.plot([], [], color="tab:red", lw=1.5, label="std_opd (long)")
        self.ax_std.legend(loc="upper right")
        self.ax_std.set_xticks([])
        for spine in self.ax_std.spines.values():
            spine.set_visible(False)
        self.ax_std.set_position([0.03, 0.03, 0.97, 0.97])
        self.ax_std.set_xlim(0, config.VIEWER_BUFFER_SIZE)

        # ========== piezos ================================================
        lf_pz = ttk.LabelFrame(wrap, text="Piezos", padding=6)
        lf_pz.pack(fill=tk.BOTH, expand=True)
        self.fig_pz = plt.Figure(figsize=(5, 1.8), dpi=100)
        self.ax_pz = self.fig_pz.add_subplot(111)
        self.ax_pz.set_title("Piezos levels")
        self.ax_pz.set_ylabel("V")
        self.ax_pz.grid(alpha=0.3)
        self.canvas_pz = FigureCanvasTkAgg(self.fig_pz, master=lf_pz)
        self.canvas_pz.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        (self.line_pz_opd,) = self.ax_pz.plot([], [], color="tab:green", lw=1.4, label="OPD")
        (self.line_pz_da1,) = self.ax_pz.plot([], [], color="tab:red", lw=1.2, label="DA-1")
        (self.line_pz_da2,) = self.ax_pz.plot([], [], color="tab:purple", lw=1.2, label="DA-2")
        self.ax_pz.legend(loc="upper right")
        self.ax_pz.set_xticks([])
        for spine in self.ax_pz.spines.values():
            spine.set_visible(False)
        self.ax_pz.set_position([0.03, 0.03, 0.97, 0.97])
        self.ax_pz.set_xlim(0, config.VIEWER_BUFFER_SIZE)

        # throttle (optional)
        self._buffers_last_draw = 0

    def _np_buffer(self, buf):
        """
        Safely converts buffer to 1D np.array (filters NaN).
        Returns np.array([]) if key is missing or empty.
        """
        try:
            arr = np.array(buf, dtype=float).ravel()
            if arr.size == 0:
                return np.array([], dtype=float)
            return arr[np.isfinite(arr)]
        except Exception:
            return np.array([], dtype=float)

    def _autoscale_1d(self, ax, x, y, pad_ratio=0.01):
        """
        Simple auto-scaling helper.
        """
        if x.size > 0 and float(np.min(x)) != float(np.max(x)):
            ax.set_xlim(float(np.min(x)), float(np.max(x)))
        if y is not None and y.size > 0:
            ymin, ymax = float(np.min(y)), float(np.max(y))
            if ymin == ymax:
                eps = 1e-6 if ymin == 0 else abs(ymin) * 0.05
                ymin -= eps
                ymax += eps
            pad = (ymax - ymin) * pad_ratio
            ax.set_ylim(ymin - pad, ymax + pad)

    def _update_buffers_tab(self):
        """
        Updates the 3 plots of the 'Buffers' tab.
        Called from refresh() to remain in sync with frame_update.
        """
        def _set_data(line, x, y):
            size = min(len(x), len(y))
            line.set_data(x[:size], y[:size])

        now = time.perf_counter()
        if now - self._buffers_last_draw < 0.1:
            return

        xtime = np.array(self.time_buf)
        xtime_long = np.array(self.time_longbuf)  # reserved

        # --- mean_opd
        y_mean = self._np_buffer(self.opd_mean_buf)
        _set_data(self.line_mean, xtime, y_mean)
        self._autoscale_1d(self.ax_mean, xtime, y_mean)
        self.canvas_mean.draw_idle()

        # --- std_opd
        y_std = self._np_buffer(self.opd_std_buf)
        _set_data(self.line_std, xtime, y_std)
        self._autoscale_1d(self.ax_std, xtime, y_std)
        self.canvas_std.draw_idle()

        # --- piezos (3 curves)
        y_pz_opd = self._np_buffer(self.piezo_opd_buf)
        y_pz_da1 = self._np_buffer(self.piezo_da1_buf)
        y_pz_da2 = self._np_buffer(self.piezo_da2_buf)
        _set_data(self.line_pz_opd, xtime, y_pz_opd)
        _set_data(self.line_pz_da1, xtime, y_pz_da1)
        _set_data(self.line_pz_da2, xtime, y_pz_da2)
        self._autoscale_1d(self.ax_pz, xtime, None)
        self.ax_pz.set_ylim(0, 10)
        self.canvas_pz.draw_idle()

        self._buffers_last_draw = now

    # ---------------------------------------------------------------------
    # PIEZOS PANEL
    # ---------------------------------------------------------------------
    def _build_piezos_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Piezos", padding=10)
        frame.pack(fill=tk.Y, expand=False)

        row = ttk.Frame(frame)
        row.pack(fill=tk.Y, expand=False)

        self.var_opd = tk.DoubleVar(value=0.0)
        self.var_da1 = tk.DoubleVar(value=0.0)
        self.var_da2 = tk.DoubleVar(value=0.0)

        # Read initial values if present
        try:
            init = self.data["DAQ.piezos_level"][:3]
            if len(init) >= 3:
                self.var_opd.set(float(init[0]))
                self.var_da1.set(float(init[1]))
                self.var_da2.set(float(init[2]))
        except Exception as e:
            log.error(f'Error when reading piezos initial values {e}')

        # Fine step for +/- buttons (Volts)
        self.piezo_step = 0.01  # V

        # Helper to nudge a DoubleVar and clamp to scale range
        def _nudge(var: tk.DoubleVar, scl: tk.Scale, delta: float):
            try:
                v = float(var.get()) + float(delta)
            except Exception:
                v = 0.0
            # Clamp to the scale's actual numeric range (visual scale is inverted)
            vmin = min(float(scl['from']), float(scl['to']))
            vmax = max(float(scl['from']), float(scl['to']))
            v = max(vmin, min(v, vmax))
            var.set(v)
            self._write_piezos_to_shared()  # propagate into self.data

        def _col(label, var, col):
            col_frame = ttk.Frame(row)
            col_frame.grid(row=0, column=col, padx=10, pady=5)
            ttk.Label(col_frame, text=label).pack(pady=(0, 6))

            scl = tk.Scale(
                col_frame, from_=10.0, to=0.0,
                variable=var, orient=tk.VERTICAL,
                showvalue=True,
                resolution=0.01,
                length=220, command=self._on_piezos_change
            )
            scl.pack()

            # Buttons row under the scale
            btns = ttk.Frame(col_frame)
            btns.pack(pady=(6, 0))

            # Minus
            ttk.Button(
                btns, text="–",
                width=3,
                command=lambda: _nudge(var, scl, -self.piezo_step)
            ).pack(side=tk.LEFT, padx=(0, 4))

            # Plus
            ttk.Button(
                btns, text="+",
                width=3,
                command=lambda: _nudge(var, scl, +self.piezo_step)
            ).pack(side=tk.LEFT)

            return scl

        self.scale_opd = _col("OPD", self.var_opd, 0)
        self.scale_da1 = _col("DA-1", self.var_da1, 1)
        self.scale_da2 = _col("DA-2", self.var_da2, 2)

        # Initial push to shared buffer
        self._write_piezos_to_shared()

    def _write_piezos_to_shared(self):
        vals = [float(self.var_opd.get()), float(self.var_da1.get()), float(self.var_da2.get())]
        self.data["DAQ.piezos_level"][:] = vals

    def _on_piezos_change(self, *_):
        self._write_piezos_to_shared()

    # ---------------------------------------------------------------------
    # DEBUG INSPECTOR
    # ---------------------------------------------------------------------
    def _build_debug_inspector(self, parent):
        """
        Build the 'Data inspector' UI:
        - Left: filter + list of available keys (from self.data)
        - Middle buttons: Add / Remove
        - Right: live table with watched keys (type, shape, preview)
        """
        lf = ttk.LabelFrame(parent, text="Data inspector", padding=10)
        lf.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        container = ttk.Frame(lf)
        container.pack(fill=tk.BOTH, expand=True)

        # --- Left column: filter + available keys list -------------------
        left = ttk.Frame(container)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ttk.Label(left, text="Filter").pack(anchor="w")
        self._dbg_filter_var = tk.StringVar(value="")
        ent = ttk.Entry(left, textvariable=self._dbg_filter_var, width=24)
        ent.pack(fill=tk.X, pady=(2, 6))
        ent.bind("<KeyRelease>", lambda e: self._dbg_refresh_keylist())

        # Listbox with available keys
        left_list_wrap = ttk.Frame(left)
        left_list_wrap.pack(fill=tk.BOTH, expand=True)
        self._dbg_keys_listbox = tk.Listbox(
            left_list_wrap, selectmode=tk.EXTENDED, height=12, exportselection=False
        )
        sb = ttk.Scrollbar(left_list_wrap, orient=tk.VERTICAL, command=self._dbg_keys_listbox.yview)
        self._dbg_keys_listbox.configure(yscrollcommand=sb.set)
        self._dbg_keys_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._dbg_keys_listbox.bind("<Double-Button-1>", lambda e: self._dbg_add_selected_keys())

        # Populate with all current keys
        self._dbg_all_keys_cache = []  # full sorted list of keys for filtering
        self._dbg_refresh_keylist(full=True)

        # --- Middle: buttons --------------------------------------------
        mid = ttk.Frame(container)
        mid.pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(mid, text="Add →", command=self._dbg_add_selected_keys).pack(pady=(30, 6))
        ttk.Button(mid, text="← Remove", command=self._dbg_remove_selected_rows).pack()

        # --- Right: watched table (Treeview) -----------------------------
        right = ttk.Frame(container)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cols = ("key", "dtype", "shape", "preview")
        self._dbg_tree = ttk.Treeview(
            right, columns=cols, show="headings", height=12, selectmode="extended"
        )
        for col, w in zip(cols, (220, 110, 110, 600)):
            self._dbg_tree.heading(col, text=col)
            self._dbg_tree.column(col, width=w, anchor="w", stretch=(col == "preview"))
        sbx = ttk.Scrollbar(right, orient=tk.HORIZONTAL, command=self._dbg_tree.xview)
        sby = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self._dbg_tree.yview)
        self._dbg_tree.configure(xscrollcommand=sbx.set, yscrollcommand=sby.set)
        self._dbg_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sby.pack(side=tk.LEFT, fill=tk.Y)
        sbx.pack(fill=tk.X)

        # Key bindings
        self._dbg_tree.bind("<Delete>", lambda e: self._dbg_remove_selected_rows())
        self._dbg_tree.bind("<BackSpace>", lambda e: self._dbg_remove_selected_rows())
        self._dbg_tree.bind("<Control-c>", lambda e: self._dbg_copy_selected_to_clipboard())

        # Context menu for copy/remove
        self._dbg_ctx = tk.Menu(self._dbg_tree, tearoff=False)
        self._dbg_ctx.add_command(label="Copy", command=self._dbg_copy_selected_to_clipboard)
        self._dbg_ctx.add_command(label="Remove", command=self._dbg_remove_selected_rows)
        self._dbg_tree.bind(
            "<Button-3>",
            lambda e: (self._dbg_tree.focus(self._dbg_tree.identify_row(e.y)),
                       self._dbg_ctx.tk_popup(e.x_root, e.y_root))
        )

    def _dbg_refresh_keylist(self, full=False):
        """Refresh the left listbox of available keys according to the filter."""
        try:
            keys = self.data.keys()
        except Exception as e:
            log.error(f'error retrieving data keys: {e}')
            keys = []
        keys = [str(k) for k in keys]
        keys.sort()

        # Cache the full list once (or when full=True)
        if full or not hasattr(self, "_dbg_all_keys_cache") or not self._dbg_all_keys_cache:
            self._dbg_all_keys_cache = keys

        flt = (self._dbg_filter_var.get() or "").strip().lower()
        if flt:
            keys = [k for k in self._dbg_all_keys_cache if flt in k.lower()]
        else:
            keys = self._dbg_all_keys_cache

        self._dbg_keys_listbox.delete(0, tk.END)
        for k in keys:
            self._dbg_keys_listbox.insert(tk.END, k)

    def _dbg_add_selected_keys(self):
        """Add selected keys from the left listbox into the watched set."""
        sel = self._dbg_keys_listbox.curselection()
        if not sel:
            return
        for idx in sel:
            key = self._dbg_keys_listbox.get(idx)
            if key not in self._debug_watch_keys:
                self._debug_watch_keys.append(key)
                # Insert a new row; item id stored for fast updates
                iid = self._dbg_tree.insert("", tk.END, values=(key, "", "", ""))
                self._debug_tree_items[key] = iid
        # Force immediate refresh of values
        self._update_debug_inspector(force=True)

    def _dbg_remove_selected_rows(self):
        """Remove selected rows from the watched table."""
        items = self._dbg_tree.selection()
        if not items:
            # If nothing selected, also try the focused row
            focus = self._dbg_tree.focus()
            if focus:
                items = (focus,)

        keys_to_remove = []
        for iid in items:
            vals = self._dbg_tree.item(iid, "values")
            if vals:
                keys_to_remove.append(vals[0])

        for k in keys_to_remove:
            if k in self._debug_watch_keys:
                self._debug_watch_keys.remove(k)
            iid = self._debug_tree_items.pop(k, None)
            if iid:
                try:
                    self._dbg_tree.delete(iid)
                except Exception:
                    pass

    def _dbg_copy_selected_to_clipboard(self):
        """Copy selected watched rows (full previews) to clipboard as text."""
        items = self._dbg_tree.selection()
        lines = []
        for iid in items:
            key, dtype, shape, preview = self._dbg_tree.item(iid, "values")
            lines.append(f"{key} [{dtype} {shape}]: {preview}")
        if lines:
            text = "\n".join(lines)
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
            except Exception:
                pass

    def _format_preview(self, value, max_elems=12, head=6, tail=6, precision=3):
        """
        Create a numpy-like preview string.
        - Scalars -> 'value'
        - Arrays -> '[x0, x1, ..., xN-2, xN-1]' (flattened if needed)
        """

        def _fmt_number_sig(x, sig=4):
            """Return a number with `sig` significant digits,
            using scientific notation when appropriate."""
            s = f"{float(x):.{sig}g}"
            # Optional: collapse exponent like e-05 -> e-5 (purely cosmetic)
            return re.sub(r"e([+-])0+(\d+)$", r"e\1\2", s)

        def _is_scalar(x):
            try:
                if np.isscalar(x):
                    return True
            except Exception:
                pass
            try:
                arr = np.array(x)
                return arr.size == 1
            except Exception:
                return True  # fallback: treat as scalar

        # Scalar?
        if _is_scalar(value):
            try:
                arr = np.array(value).reshape(-1)
                v = arr[0]
                return _fmt_number_sig(v, sig=4) if _np.issubdtype(arr.dtype, _np.number) else str(v)
            except Exception:
                return str(value)

        # Array-like
        try:
            arr = np.array(value)
        except Exception:
            # unknown container: just stringify
            return str(value)

        flat = arr.ravel()
        n = flat.size

        def _fmt(x):
            try:
                return _fmt_number_sig(x, sig=4) if _np.issubdtype(flat.dtype, _np.number) else str(x)
            except Exception:
                return str(x)

        if n <= max_elems:
            return "[" + ", ".join(_fmt(x) for x in flat.tolist()) + "]"
        else:
            head_vals = ", ".join(_fmt(x) for x in flat[:head].tolist())
            tail_vals = ", ".join(_fmt(x) for x in flat[-tail:].tolist())
            return f"[{head_vals}, ..., {tail_vals}]"

    def _describe_value(self, value):
        """
        Return (dtype_str, shape_str) for the given value.
        Scalars -> dtype, shape='()' ; Arrays -> dtype, shape='(d0,d1,...)'
        """
        try:
            arr = np.array(value)
            dt = str(arr.dtype)
            shp = "()" if arr.shape == () or arr.size == 1 else str(tuple(arr.shape))
            return dt, shp
        except Exception:
            return (type(value).__name__, "")

    def _update_debug_inspector(self, force=False):
        """
        Periodically refresh the watched rows with latest values from self.data.
        Throttled to avoid redrawing too often.
        """
        now = time.perf_counter()
        if not force and (now - getattr(self, "_debug_last_update", 0.0) < 0.20):
            return

        for key in list(self._debug_watch_keys):  # copy since we may remove missing keys
            iid = self._debug_tree_items.get(key)
            try:
                val = self.data[key]
            except Exception:
                # key removed from self.data
                if iid:
                    self._dbg_tree.set(iid, "dtype", "—")
                    self._dbg_tree.set(iid, "shape", "—")
                    self._dbg_tree.set(iid, "preview", "(missing)")
                continue

            dtype, shape = self._describe_value(val)
            preview = self._format_preview(val)
            if iid:
                self._dbg_tree.set(iid, "dtype", dtype)
                self._dbg_tree.set(iid, "shape", shape)
                self._dbg_tree.set(iid, "preview", preview)
            else:
                # unexpected: row missing, recreate
                iid = self._dbg_tree.insert("", tk.END, values=(key, dtype, shape, preview))
                self._debug_tree_items[key] = iid

        self._debug_last_update = now

    def reset_tiptilt(self):
        """Set target angles equal to current (zero error)."""
        self.data['Servo.tip_target'][0] = float(np.mean(self.data['IRCamera.tip_buffer'][:100]))
        self.data['Servo.tilt_target'][0] = float(np.mean(self.data['IRCamera.tilt_buffer'][:100]))

    # events
    def stop_servo(self):
        self._set_event('Servo.stop')

    def normalize(self):
        self._set_event('Servo.normalize')

    def reset_zpd(self):
        self._set_event('Servo.reset_zpd')
