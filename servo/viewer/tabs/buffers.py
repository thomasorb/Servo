import time
from collections import deque
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ... import config


class BuffersTab:
    """Dynamic time-series tab (1 subplot per series)."""

    COLORS = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    REDUCERS = ('value', 'mean', 'min', 'max', 'std')

    def __init__(self, parent, viewer):
        self.viewer = viewer
        self.root = ttk.Frame(parent)
        self.root.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # time base + x buffer (seconds)
        self.t0 = time.perf_counter()
        self.time_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)

        # series defs [{'key','reducer','color','index','label'}]
        self.series_defs = []
        # y buffers per series ident=(key,index)
        self.series_bufs = {}

        # matplotlib objects
        self.fig = None
        self.canvas = None
        self.axes_by_ident = {}   # ident -> Axes
        self.lines_by_ident = {}  # ident -> Line2D

        self._last_draw = 0.0
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        ttk.Label(left, text='Filter').pack(anchor='w')
        self.filter_var = tk.StringVar(value='')
        ent = ttk.Entry(left, textvariable=self.filter_var, width=26)
        ent.pack(fill=tk.X, pady=(2, 6))
        ent.bind('<KeyRelease>', lambda e: self._refresh_keylist())

        self.lb = tk.Listbox(left, selectmode=tk.EXTENDED, height=16, exportselection=False)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.lb.yview)
        self.lb.configure(yscrollcommand=sb.set)
        self.lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._all_keys_cache = []
        self._refresh_keylist(full=True)

        mid = ttk.Frame(self.root)
        mid.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(mid, text='Add →', command=self._add_selected).pack(pady=(40, 6))
        ttk.Button(mid, text='← Remove', command=self._remove_selected).pack()

        right = ttk.Frame(self.root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # selected table
        cols = ('label', 'key', 'index', 'reducer', 'color')
        self.tree = ttk.Treeview(right, columns=cols, show='headings', height=6, selectmode='extended')
        for c, w in zip(cols, (220, 360, 80, 120, 80)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor='w')
        self.tree.pack(fill=tk.X, padx=4, pady=(0, 6))

        # edit row
        edit = ttk.Frame(right)
        edit.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(edit, text='Reducer').pack(side=tk.LEFT)
        self.reducer_var = tk.StringVar(value=self.REDUCERS[0])
        cb_red = ttk.Combobox(edit, textvariable=self.reducer_var,
                              values=self.REDUCERS, state='readonly', width=10)
        cb_red.pack(side=tk.LEFT, padx=(6, 14))
        ttk.Button(edit, text='Apply to selection', command=self._apply_reducer).pack(side=tk.LEFT, padx=(0, 8))
        # NEW: clear buffers button
        ttk.Button(edit, text='Clear buffers', command=self._clear_buffers).pack(side=tk.LEFT)

        # presets
        presets = ttk.LabelFrame(right, text='Presets', padding=6)
        presets.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(presets, text='OPD mean', command=self._preset_opd_mean).pack(side=tk.LEFT, padx=4)
        ttk.Button(presets, text='OPD std', command=self._preset_opd_std).pack(side=tk.LEFT, padx=4)
        ttk.Button(presets, text='Piezos', command=self._preset_piezos).pack(side=tk.LEFT, padx=4)

        # figure
        fig_wrap = ttk.LabelFrame(right, text='Time series', padding=6)
        fig_wrap.pack(fill=tk.BOTH, expand=True)
        self.fig = plt.Figure(figsize=(6, 2.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_wrap)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- Keys ----------
    def _refresh_keylist(self, full=False):
        try:
            keys = list(self.viewer.data.keys())
        except Exception:
            keys = []
        keys = [str(k) for k in keys]
        keys.sort()
        if full or not self._all_keys_cache:
            self._all_keys_cache = keys
        flt = (self.filter_var.get() or '').strip().lower()
        keys = [k for k in self._all_keys_cache if flt in k.lower()] if flt else self._all_keys_cache
        self.lb.delete(0, tk.END)
        for k in keys:
            self.lb.insert(tk.END, k)

    # ---------- Series mgmt ----------
    def _color_next(self):
        return self.COLORS[len(self.series_defs) % len(self.COLORS)]

    def _add_series(self, key, reducer='value', index=None, label=None, color=None):
        ident = (key, index)
        if any((d['key'], d['index']) == ident for d in self.series_defs):
            return
        color = color or self._color_next()
        label = label or (f"{key}[{index}]" if index is not None else key)
        d = {'key': key, 'reducer': reducer, 'color': color, 'index': index, 'label': label}
        self.series_defs.append(d)
        self.series_bufs[ident] = deque(maxlen=config.VIEWER_BUFFER_SIZE)
        # mirror row
        self.tree.insert('', tk.END, values=(label, key, '' if index is None else index, reducer, color))
        # rebuild plots
        self._rebuild_axes()

    def _add_selected(self):
        sels = self.lb.curselection()
        if not sels:
            return
        for i in sels:
            key = self.lb.get(i)
            self._add_series(key, reducer=self.REDUCERS[0], index=None, label=key)
        self.canvas.draw_idle()

    def _remove_selected(self):
        items = self.tree.selection()
        if not items:
            focus = self.tree.focus()
            if focus:
                items = (focus,)
        to_rm = []
        for iid in items:
            vals = self.tree.item(iid, 'values')
            if not vals:
                continue
            label, key, index, reducer, color = vals
            idx = None if index in (None, '', 'None') else int(index)
            to_rm.append((key, idx, iid))
        for key, idx, iid in to_rm:
            self.tree.delete(iid)
            self.series_defs = [d for d in self.series_defs if not (d['key'] == key and d['index'] == idx)]
            ident = (key, idx)
            self.series_bufs.pop(ident, None)
        self._rebuild_axes()

    def _apply_reducer(self):
        reducer = self.reducer_var.get()
        items = self.tree.selection()
        if not items:
            return
        for iid in items:
            vals = list(self.tree.item(iid, 'values'))
            if not vals:
                continue
            label, key, index, _old, color = vals
            vals[3] = reducer
            self.tree.item(iid, values=tuple(vals))
            idx = None if index in (None, '', 'None') else int(index)
            for d in self.series_defs:
                if d['key'] == key and d['index'] == idx:
                    d['reducer'] = reducer
                    break

    # ---------- Clear buffers ----------
    def _clear_buffers(self):
        """Clear all buffers and reset time origin."""
        # clear time axis + reset t0 so time restarts at 0.0s
        self.t0 = time.perf_counter()
        self.time_buf.clear()
        # clear data series
        for dq in self.series_bufs.values():
            dq.clear()
        # wipe lines on screen
        for line in self.lines_by_ident.values():
            line.set_data([], [])
        # reset axes limits
        for ax in self.axes_by_ident.values():
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()

    # ---------- Figure ----------
    def _rebuild_axes(self):
        """Recreate stacked subplots to match series_defs."""
        self.fig.clear()
        self.axes_by_ident.clear()
        self.lines_by_ident.clear()

        n = max(1, len(self.series_defs))
        # dynamic figure height
        base_h = 2.2
        self.fig.set_size_inches(6, max(base_h, min(2.2 * n, 2.2 * 6)), forward=True)

        axes = []
        for i, d in enumerate(self.series_defs):
            sharex = axes[0] if axes else None
            ax = self.fig.add_subplot(n, 1, i + 1, sharex=sharex)
            ax.grid(alpha=0.3)
            ax.set_ylabel('value')
            ax.set_title(d['label'])
            if i < n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('t [s]')  # explicit seconds on the last subplot
            ident = (d['key'], d['index'])
            line, = ax.plot([], [], color=d['color'], lw=1.6)
            self.axes_by_ident[ident] = ax
            self.lines_by_ident[ident] = line
            axes.append(ax)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---------- Reducers ----------
    def _reduce_value(self, v, how: str, index=None):
        try:
            arr = np.array(v)
        except Exception:
            return np.nan
        if arr.size == 0:
            return np.nan
        if index is not None:
            try:
                arr = arr.reshape(-1)
                if index < 0 or index >= arr.size:
                    return np.nan
                arr = arr[index]
            except Exception:
                return np.nan
        if np.isscalar(arr):
            return float(arr)
        if how == 'value':
            return float(arr.reshape(-1)[0])
        if how == 'mean':
            return float(np.nanmean(arr))
        if how == 'min':
            return float(np.nanmin(arr))
        if how == 'max':
            return float(np.nanmax(arr))
        if how == 'std':
            return float(np.nanstd(arr))
        return float(arr.reshape(-1)[0])

    # ---------- Presets ----------
    def _preset_opd_mean(self):
        self._add_series('IRCamera.mean_opd', reducer='value', label='OPD mean')

    def _preset_opd_std(self):
        self._add_series('IRCamera.mean_opd_buffer', reducer='std', label='OPD std')

    def _preset_piezos(self):
        labels = ('OPD', 'DA-1', 'DA-2')
        for i, lab in enumerate(labels):
            self._add_series('DAQ.piezos_level_actual', reducer='value', index=i, label=f'Piezo {lab}')

    # ---------- Update loop ----------
    def update(self):
        """Push samples + refresh plots (throttled, shared X in seconds)."""
        # time (seconds since last clear)
        t = time.perf_counter() - self.t0
        self.time_buf.append(t)  # bounded by config.VIEWER_BUFFER_SIZE

        # y samples per series
        for d in list(self.series_defs):
            key, idx, reducer = d['key'], d['index'], d['reducer']
            try:
                val = self.viewer.data[key]
            except Exception:
                val = np.nan
            y = self._reduce_value(val, reducer, index=idx)
            self.series_bufs[(key, idx)].append(y)  # bounded by config.VIEWER_BUFFER_SIZE

        # draw (throttle ~10 fps)
        now = time.perf_counter()
        if self._last_draw and (now - self._last_draw) < 0.10:
            return
        self._last_draw = now

        tx = np.array(self.time_buf)  # shared X (seconds)
        for ident, line in list(self.lines_by_ident.items()):
            y = np.array(self.series_bufs.get(ident, []), dtype=float)
            n = min(tx.size, y.size)
            line.set_data(tx[:n], y[:n])
            self._autoscale_axis(self.axes_by_ident.get(ident), tx[:n], y[:n])

        self.canvas.draw_idle()

    def _autoscale_axis(self, ax, x, y):
        if ax is None or x.size < 2:
            return
        # x range shared (seconds)
        ax.set_xlim(float(x.min()), float(x.max()))
        # y range with pad
        if y.size == 0 or not np.isfinite(y).any():
            return
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        if ymin == ymax:
            eps = 1e-6 if ymin == 0 else abs(ymin) * 0.05
            ymin -= eps
            ymax += eps
        pad = (ymax - ymin) * 0.05
        ax.set_ylim(ymin - pad, ymax + pad)
