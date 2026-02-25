import tkinter as tk
from tkinter import ttk
import numpy as np
import time
from ...fsm import ServoState, NexlineState, WorkerState  # enums (parent of viewer)

class DebugTab:
    """Data inspector + status/states."""

    def __init__(self, parent, viewer):
        self.viewer = viewer
        self.root = ttk.Frame(parent)
        self.root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Status (mirrors right column) ---
        # status = ttk.LabelFrame(self.root, text='Status', padding=10)
        # status.pack(fill=tk.X, expand=False, pady=(0, 8))
        # ttk.Label(status, textvariable=self.viewer.status_var, justify='left').pack(anchor='w')

        # --- States (names) ---
        states = ttk.LabelFrame(self.root, text='States', padding=10)
        states.pack(fill=tk.X, expand=False, pady=(0, 8))
        grid = ttk.Frame(states); grid.pack(fill=tk.X)

        self._lbl_servo = tk.StringVar(value='Servo: —')
        self._lbl_nx    = tk.StringVar(value='Nexline: —')
        self._lbl_ir    = tk.StringVar(value='IRCamera: —')
        self._lbl_daq   = tk.StringVar(value='DAQ: —')
        self._lbl_sc    = tk.StringVar(value='SerialComm: —')

        for col, v in enumerate((self._lbl_servo, self._lbl_nx, self._lbl_ir, self._lbl_daq, self._lbl_sc)):
            ttk.Label(grid, textvariable=v).grid(row=0, column=col, sticky='w', padx=(0, 14))

        # --- Inspector (keys) ---
        self._build_inspector()

        self._last_update = 0.0

    # inspector UI
    def _build_inspector(self):
        lf = ttk.LabelFrame(self.root, text='Data inspector', padding=10)
        lf.pack(fill=tk.BOTH, expand=True)
        container = ttk.Frame(lf)
        container.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(container)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        ttk.Label(left, text='Filter').pack(anchor='w')
        self.filter_var = tk.StringVar(value='')
        ent = ttk.Entry(left, textvariable=self.filter_var, width=24)
        ent.pack(fill=tk.X, pady=(2, 6))
        ent.bind('<KeyRelease>', lambda e: self._refresh_keylist())
        self.lb = tk.Listbox(left, selectmode=tk.EXTENDED, height=12, exportselection=False)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.lb.yview)
        self.lb.configure(yscrollcommand=sb.set)
        self.lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._all_keys_cache = []
        self._refresh_keylist(full=True)

        mid = ttk.Frame(container)
        mid.pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(mid, text='Add →', command=self._add_selected).pack(pady=(30, 6))
        ttk.Button(mid, text='← Remove', command=self._remove_selected).pack()

        right = ttk.Frame(container)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cols = ('key', 'dtype', 'shape', 'preview')
        self.tree = ttk.Treeview(right, columns=cols, show='headings', height=12, selectmode='extended')
        for col, w in zip(cols, (220, 110, 110, 600)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor='w', stretch=(col == 'preview'))
        sbx = ttk.Scrollbar(right, orient=tk.HORIZONTAL, command=self.tree.xview)
        sby = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(xscrollcommand=sbx.set, yscrollcommand=sby.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sby.pack(side=tk.LEFT, fill=tk.Y)
        sbx.pack(fill=tk.X)

        self.tree.bind('<Delete>', lambda e: self._remove_selected())
        self.tree.bind('<BackSpace>', lambda e: self._remove_selected())

        self.watch_keys = []
        self.tree_items = {}

    # key list
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

    def _add_selected(self):
        sel = self.lb.curselection()
        if not sel: return
        for idx in sel:
            key = self.lb.get(idx)
            if key not in self.watch_keys:
                self.watch_keys.append(key)
                iid = self.tree.insert('', tk.END, values=(key, '', '', ''))
                self.tree_items[key] = iid
        self.update(force=True)

    def _remove_selected(self):
        items = self.tree.selection()
        if not items:
            focus = self.tree.focus()
            if focus: items = (focus,)
        keys_to_remove = []
        for iid in items:
            vals = self.tree.item(iid, 'values')
            if vals: keys_to_remove.append(vals[0])
        for k in keys_to_remove:
            if k in self.watch_keys: self.watch_keys.remove(k)
            iid = self.tree_items.pop(k, None)
            if iid:
                try: self.tree.delete(iid)
                except Exception: pass

    # formatting
    def _describe(self, v):
        try:
            arr = np.array(v)
            dt = str(arr.dtype)
            shp = '()' if arr.shape == () or arr.size == 1 else str(tuple(arr.shape))
            return dt, shp
        except Exception:
            return (type(v).__name__, '')

    def _preview(self, v, max_elems=12, head=6, tail=6):
        try:
            arr = np.array(v)
        except Exception:
            return str(v)
        if arr.size <= 1:
            try: return f"{float(arr.reshape(-1)[0]):.4g}"
            except Exception: return str(arr)
        flat = arr.ravel()
        if flat.size <= max_elems:
            return '[' + ', '.join(f"{float(x):.4g}" if np.issubdtype(flat.dtype, np.number) else str(x)
                                   for x in flat.tolist()) + ']'
        head_vals = ', '.join(f"{float(x):.4g}" for x in flat[:head].tolist())
        tail_vals = ', '.join(f"{float(x):.4g}" for x in flat[-tail:].tolist())
        return f"[{head_vals}, ..., {tail_vals}]"

    def _update_states_block(self):
        # map numeric -> enum names (fallback '—')
        def _name(enum_cls, key):
            try: return enum_cls(self.viewer.data[key][0]).name
            except Exception: return '—'
        self._lbl_servo.set(f"Servo: {_name(ServoState, 'Servo.state')}")
        self._lbl_nx.set(f"Nexline: {_name(NexlineState, 'Nexline.state')}")
        self._lbl_ir.set(f"IRCamera: {_name(WorkerState, 'IRCamera.state')}")
        self._lbl_daq.set(f"DAQ: {_name(WorkerState, 'DAQ.state')}")
        self._lbl_sc.set(f"SerialComm: {_name(WorkerState, 'SerialComm.state')}")

    def update(self, force=False):
        now = time.perf_counter()
        if not force and (now - self._last_update) < 0.20:
            return
        self._last_update = now

        # update “States”
        self._update_states_block()

        # update inspector rows
        for key in list(self.watch_keys):
            iid = self.tree_items.get(key)
            try:
                val = self.viewer.data[key]
                dt, shp = self._describe(val)
                pv = self._preview(val)
            except Exception:
                dt, shp, pv = '—', '—', '(missing)'
            if iid:
                self.tree.set(iid, 'dtype', dt)
                self.tree.set(iid, 'shape', shp)
                self.tree.set(iid, 'preview', pv)
                
