# tabs/config.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

class ConfigTab:
    """
    Config tab:
      - Lists all self.data keys starting with 'params.'
      - Renders one row per key: label + editor(s)
      - Supports scalars and small 1D arrays
      - Single "Apply" button to push all changes, plus "Reload"
    """
    MAX_INLINE_ARRAY = 8  # "small arrays": render one Entry per element

    def __init__(self, parent, viewer):
        self.viewer = viewer
        self.root = parent
        self.data = viewer.data

        # key -> meta: {'widgets': [Entry...], 'dtype': np.dtype, 'len': int}
        self._rows = {}

        # --- Scrollable area + bottom bar ---
        outer = ttk.Frame(parent)
        outer.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self._inner = ttk.Frame(self._canvas)
        self._inner_id = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")

        # Keep inner width in sync with canvas width
        self._inner.bind("<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(self._inner_id, width=e.width))

        # Natural mouse scrolling
        self._inner.bind_all("<MouseWheel>", self._on_mousewheel)

        # Build the dynamic form
        self._build_form()

        # Bottom action bar
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(bar, text="Reload", command=self.reload).pack(side=tk.RIGHT, padx=6, pady=6)
        ttk.Button(bar, text="Apply", style='Green.TButton', command=self.apply).pack(side=tk.RIGHT, padx=6, pady=6)

    # ---------- structure helpers ----------

    def _on_mousewheel(self, event):
        """Scroll canvas content with mouse wheel."""
        try:
            delta = int(-1*(event.delta/120))
        except Exception:
            delta = -1 if event.delta < 0 else 1
        self._canvas.yview_scroll(delta, "units")

    def _discover_config_keys(self):
        """
        Return sorted list of 'config.*' keys.
        Assumes self.data behaves like a Mapping (keys()/__iter__).
        """
        keys = []
        try:
            keys = [k for k in self.data.keys() if isinstance(k, str) and k.startswith("params.")]
        except Exception:
            try:
                keys = [k for k in list(self.data) if isinstance(k, str) and k.startswith("params.")]
            except Exception:
                keys = []
        keys.sort()
        return keys

    def _read_array(self, key):
        """
        Read the raw value for 'key' and return (1D np.ndarray, dtype).
        - Scalars stored in a buffer (length 1) become shape (1,)
        - Arrays are flattened to 1D
        """
        try:
            buf = self.data[key]
            arr = np.array(buf)           # safe copy, supports memoryview-like objects
            dtype = arr.dtype
            arr = arr.ravel()
        except Exception:
            dtype = np.float64
            arr = np.array([], dtype=dtype)
        return arr, dtype

    def _build_form(self):
        """(Re)build the whole form from current self.data."""
        # Clear previous content
        for child in self._inner.winfo_children():
            child.destroy()
        self._rows.clear()

        keys = self._discover_config_keys()
        if not keys:
            ttk.Label(self._inner, text="No 'params.' keys found in self.data").pack(anchor="w", padx=8, pady=8)
            return

        # Header
        header = ttk.Frame(self._inner)
        header.pack(fill=tk.X, padx=8, pady=(8, 0))
        ttk.Label(header, text="Parameter", width=36).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Value(s)").grid(row=0, column=1, sticky="w")

        sep = ttk.Separator(self._inner, orient="horizontal")
        sep.pack(fill=tk.X, padx=6, pady=6)

        # Rows
        for i, key in enumerate(keys, start=1):
            row = ttk.Frame(self._inner)
            row.pack(fill=tk.X, padx=8, pady=4)

            # Short label without the 'params.' prefix
            pretty = key[len("params."):] if key.startswith("params.") else key
            ttk.Label(row, text=pretty).grid(row=0, column=0, sticky="w")

            values, dtype = self._read_array(key)

            # Editor area
            editor = ttk.Frame(row)
            editor.grid(row=0, column=1, sticky="w")

            widgets = []

            if values.size == 0:
                # Unreadable/unsupported → disabled field
                e = tk.Entry(editor, width=12)
                e.insert(0, "N/A")
                e.configure(state="disabled")
                e.pack(side=tk.LEFT)
                widgets.append(e)

            elif values.size == 1:
                # Scalar
                e = tk.Entry(editor, width=16)
                e.insert(0, self._format_number(values[0], dtype))
                e.pack(side=tk.LEFT, padx=(0, 4))
                widgets.append(e)

            elif values.size <= self.MAX_INLINE_ARRAY:
                # Small array → one Entry per element
                for k in range(values.size):
                    e = tk.Entry(editor, width=10)
                    e.insert(0, self._format_number(values[k], dtype))
                    e.pack(side=tk.LEFT, padx=(0, 4))
                    widgets.append(e)
            else:
                # Larger array → single CSV field
                e = tk.Entry(editor, width=48)
                e.insert(0, ", ".join(self._format_number(v, dtype) for v in values))
                e.pack(side=tk.LEFT, padx=(0, 4))
                widgets.append(e)

            # Store row metadata
            self._rows[key] = {
                "widgets": widgets,
                "dtype": dtype,
                "len": int(values.size)
            }

    # ---------- formatting & parsing ----------

    @staticmethod
    def _format_number(v, dtype):
        """Human-friendly formatting; preserve integers as integers."""
        if np.issubdtype(dtype, np.integer):
            return str(int(v))
        if np.issubdtype(dtype, np.bool_):
            return "1" if bool(v) else "0"
        # Floats: compact yet precise enough for typical nm/us parameters
        return f"{float(v):.6g}"

    @staticmethod
    def _parse_token_to_float(token: str):
        """Parse a single token into float (supports bool-like and hex)."""
        s = token.strip()
        if s.lower() in ("true", "false"):
            return 1.0 if s.lower() == "true" else 0.0
        if s.startswith("0x") or s.startswith("0X"):
            return float(int(s, 16))
        return float(s)  # raises ValueError on invalid input

    @staticmethod
    def _cast_to_dtype(vals, dtype):
        """Cast numpy array to the original dtype (no unnecessary copy)."""
        arr = np.asarray(vals)
        return arr.astype(dtype, copy=False)

    def _read_widgets(self, key, meta):
        """
        Collect editors' content and return a 1D np.ndarray in the original dtype.
        Handles CSV (long array) and multiple entries (small arrays).
        Raises ValueError if any cell is invalid.
        """
        widgets = meta["widgets"]
        n = meta["len"]
        dtype = meta["dtype"]

        if n <= 1:
            # scalar or empty
            raw = widgets[0].get()
            x = self._parse_token_to_float(raw)
            out = np.array([x], dtype=float)

        elif n <= self.MAX_INLINE_ARRAY:
            vals = []
            for w in widgets:
                vals.append(self._parse_token_to_float(w.get()))
            out = np.array(vals, dtype=float)

        else:
            # CSV
            raw = widgets[0].get()
            tokens = [t for t in raw.replace(";", ",").split(",") if t.strip() != ""]
            vals = [self._parse_token_to_float(t) for t in tokens]
            out = np.array(vals[:n], dtype=float)
            if out.size != n:
                raise ValueError(f"{key}: {out.size} value(s) provided, {n} expected")

        return self._cast_to_dtype(out, dtype)

    # ---------- user actions ----------

    def reload(self):
        """Reload current values from self.data and reset editors' content."""
        for key, meta in self._rows.items():
            values, dtype = self._read_array(key)
            meta["dtype"] = dtype  # in case producer changed the dtype
            meta["len"] = int(values.size)
            widgets = meta["widgets"]

            # Reset background color (clear previous validation errors)
            for w in widgets:
                try:
                    w.configure(background="white")
                except Exception:
                    pass

            if values.size <= 1:
                widgets[0].delete(0, tk.END)
                widgets[0].insert(0, self._format_number(values[0] if values.size else 0, dtype))

            elif values.size <= self.MAX_INLINE_ARRAY and len(widgets) == values.size:
                for i, w in enumerate(widgets):
                    w.delete(0, tk.END)
                    w.insert(0, self._format_number(values[i], dtype))

            else:
                # CSV
                widgets[0].delete(0, tk.END)
                widgets[0].insert(0, ", ".join(self._format_number(v, dtype) for v in values))

    def apply(self):
        """
        Validate and write **all** edited values back to self.data.
        - Scalars: self.data[key][0] = value
        - Arrays:  self.data[key][:n] = values (slice write)
        Colors invalid cells in red and shows an error dialog; shows success otherwise.
        """
        errors = []

        # Reset colors before validation
        for meta in self._rows.values():
            for w in meta["widgets"]:
                try:
                    w.configure(background="white")
                except Exception:
                    pass

        for key, meta in self._rows.items():
            try:
                arr = self._read_widgets(key, meta)

                # Write back to the shared buffer
                if arr.size <= 1:
                    self.data[key][0] = arr.astype(meta["dtype"])[0]
                else:
                    # Respect the actual buffer length if needed
                    try:
                        buf = self.data[key]
                        m = min(len(buf), arr.size)
                    except Exception:
                        m = arr.size
                    self.data[key][:m] = arr[:m]

            except Exception as e:
                errors.append((key, str(e)))
                # Highlight all editors in the row
                for w in meta["widgets"]:
                    try:
                        w.configure(background="#f2b6b6")  # pastel red
                    except Exception:
                        pass

        if errors:
            lines = [f"• {k} — {msg}" for k, msg in errors]
            messagebox.showerror("Params — validation errors",
                                 "Some values are invalid:\n\n" + "\n".join(lines))
        else:
            messagebox.showinfo("Params", "Parameters have been applied.")
