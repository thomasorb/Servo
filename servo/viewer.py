
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from pathlib import Path

from . import core
from . import config


class Viewer(core.Worker):
    """
    Tkinter-based scientific viewer with:
    - Zoom/pan
    - ROI tool
    - Line profile tool
    - Adjustable LUTs
    - Histogram window
    - Export to PNG
    - Piezos panel (OPD / DA-1 / DA-2)
    - Graceful shutdown via stop_event
    """

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------
    def __init__(self, data, stop_event=None):
        self.data = data
        self.stop_event = stop_event

        self.root = tk.Tk()
        self.root.title("IRCamera Viewer")
        self.root.geometry("1280x800")
        self.root.minsize(900, 600)

        # Restore old window geometry
        self.load_window_geometry()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root.option_add("*Font", ("Noto Sans", 14))
        self.root.tk.call("tk", "scaling", 1.0)

        style = ttk.Style(self.root)
        style.configure(".", font=("Noto Sans", 14))
        style.theme_use("clam")

        # ------------------------------------------------------------------
        # Image info
        # ------------------------------------------------------------------
        self.img_w = self.data['IRCamera.frame_dimx'][0]
        self.img_h = self.data['IRCamera.frame_dimy'][0]
        self.frame = self.generate_frame()

        # ------------------------------------------------------------------
        # Brightness / Contrast
        # ------------------------------------------------------------------
        self.brightness = 0.0
        self.contrast = 1.0
        self.last_bc_mouse = None

        # ------------------------------------------------------------------
        # LUT
        # ------------------------------------------------------------------
        self.current_lut_name = "gray"
        self.lut = self.build_lut_gray()

        # ------------------------------------------------------------------
        # MAIN LAYOUT : canvas (left) + right panel
        # ------------------------------------------------------------------
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas
        self.canvas = tk.Canvas(main_frame, bg="black")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right-side panel
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        # Piezos frame
        self._build_piezos_controls(right_panel)

        # Tk image holder
        self.root.after_idle(lambda: self.on_resize(None))
        self.tk_image = None
        self.image_id = None

        # ------------------------------------------------------------------
        # Zoom & Pan
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Toolbar
        # ------------------------------------------------------------------
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, pady=3)

        ttk.Label(toolbar, text="LUT:").pack(side=tk.LEFT, padx=5)
        self.lut_var = tk.StringVar(value="gray")
        lut_choices = ["gray", "viridis", "inferno", "magma",
                       "plasma", "cividis", "turbo", "rainbow"]
        self.lut_menu = ttk.Combobox(toolbar, textvariable=self.lut_var,
                                     values=lut_choices, state="readonly", width=10)
        self.lut_menu.pack(side=tk.LEFT, padx=5)
        self.lut_menu.bind("<<ComboboxSelected>>", self.on_lut_changed)

        self.hist_btn = ttk.Button(toolbar, text="Show Histogram",
                                   command=self.toggle_histogram)
        self.hist_btn.pack(side=tk.RIGHT, padx=10)

        self.reset_btn = ttk.Button(toolbar, text="Reset Zoom",
                                    command=self.reset_zoom)
        self.reset_btn.pack(side=tk.RIGHT, padx=10)

        self.clear_roi_btn = ttk.Button(toolbar, text="Clear ROI",
                                        command=self.clear_roi)
        self.clear_roi_btn.pack(side=tk.RIGHT, padx=10)

        self.export_btn = ttk.Button(toolbar, text="Export PNG",
                                     command=self.export_png)
        self.export_btn.pack(side=tk.RIGHT, padx=10)

        # ------------------------------------------------------------------
        # Info panel
        # ------------------------------------------------------------------
        info = ttk.Frame(self.root, height=30)
        info.pack(fill=tk.X, pady=3)
        info.pack_propagate(False)

        ttk.Label(info, text="Position:").pack(side=tk.LEFT)
        self.pos_var = tk.StringVar(value="x = —, y = —")
        ttk.Label(info, textvariable=self.pos_var).pack(side=tk.LEFT, padx=5)

        ttk.Label(info, text="Value:").pack(side=tk.LEFT)
        self.val_var = tk.StringVar(value="—")
        ttk.Label(info, textvariable=self.val_var).pack(side=tk.LEFT, padx=5)

        self.roi_stats_var = tk.StringVar(value="ROI: none")
        ttk.Label(info, textvariable=self.roi_stats_var,
                  foreground="#5FA8FF").pack(side=tk.RIGHT, padx=10)

        # ------------------------------------------------------------------
        # CSV logging
        # ------------------------------------------------------------------
        self.csv_file = "clicks.csv"
        self.ensure_csv_header()

        # ------------------------------------------------------------------
        # Histogram window
        # ------------------------------------------------------------------
        self.hist_window = None
        self.hist_canvas = None
        self.hist_fig = None
        self.hist_ax = None

        # ------------------------------------------------------------------
        # Mouse bindings
        # ------------------------------------------------------------------
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

        # SHIFT tool (line profile)
        self.profile_active = False
        self.profile_start = None
        self.profile_line_id = None

        # CTRL tool (ROI)
        self.roi_active = False
        self.roi_start_canvas = None
        self.roi_rect_id = None
        self.roi_bounds_img = None

        self._normal_click_start = None

        self.root.bind("<<Shutdown>>", lambda e: self._really_stop())
        
        # Start periodic refresh
        self.root.after(100, self.refresh)

    # ----------------------------------------------------------------------
    # PUBLIC CONTROL
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # FRAME ACQUISITION
    # ----------------------------------------------------------------------
    def generate_frame(self):
        """Dummy initial frame."""
        return np.random.rand(self.img_h, self.img_w).astype(np.float32)

    # ----------------------------------------------------------------------
    # LUT BUILDERS (gray, viridis, inferno, etc.)
    # ----------------------------------------------------------------------
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
        self.lut = mappingname
        self.render()
        self.update_histogram()

    # ----------------------------------------------------------------------
    # BRIGHTNESS / CONTRAST
    # ----------------------------------------------------------------------
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
        self.update_histogram()

    def bc_end(self, *_):
        self.last_bc_mouse = None

    # ----------------------------------------------------------------------
    # NORMALIZATION + APPLY LUT
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # RENDERING
    # ----------------------------------------------------------------------
    def render(self):
        stretched = self.stretch_frame(self.frame)
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

        # ROI overlay
        if self.roi_bounds_img:
            x0, y0, x1, y1 = self.roi_bounds_img
            cx0, cy0 = self.image_to_canvas(x0, y0)
            cx1, cy1 = self.image_to_canvas(x1, y1)
            if self.roi_rect_id is None:
                self.roi_rect_id = self.canvas.create_rectangle(
                    cx0, cy0, cx1, cy1, outline="#5FA8FF", width=2)
            else:
                self.canvas.coords(self.roi_rect_id, cx0, cy0, cx1, cy1)

        # Markers
        r = 4
        for item_id, ix, iy in self.points:
            cx, cy = self.image_to_canvas(ix, iy)
            self.canvas.coords(item_id, cx-r, cy-r, cx+r, cy+r)

        self.canvas.tag_raise(self.cross_h)
        self.canvas.tag_raise(self.cross_v)

    # ----------------------------------------------------------------------
    # ZOOM / PAN
    # ----------------------------------------------------------------------
    def on_resize(self, event):
        if not self.fitted:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw <= 1 or ch <= 1:
                return
            self.scale = min(cw/self.img_w, ch/self.img_h)
            self.offset_x = (cw - self.img_w*self.scale) / 2
            self.offset_y = (ch - self.img_h*self.scale) / 2
            self.fitted = True
            self.render()

    def reset_zoom(self):
        self.fitted = False
        self.on_resize(None)

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

    # ----------------------------------------------------------------------
    # COORDINATE TRANSFORMS
    # ----------------------------------------------------------------------
    def canvas_to_image(self, cx, cy):
        ix = int((cx - self.offset_x) / self.scale)
        iy = int((cy - self.offset_y) / self.scale)
        return ix, iy

    def image_to_canvas(self, ix, iy):
        return ix*self.scale + self.offset_x, iy*self.scale + self.offset_y

    # ----------------------------------------------------------------------
    # MOUSE MOVE: crosshair + pixel readout
    # ----------------------------------------------------------------------
    def on_mouse_move(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y)
        if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
            self.pos_var.set(f"x = {ix}, y = {iy}")
            self.val_var.set(f"{float(self.frame[iy, ix]):.3f}")
        else:
            self.pos_var.set("x = —, y = —")
            self.val_var.set("—")

        self.canvas.coords(self.cross_h, 0, event.y,
                           self.canvas.winfo_width(), event.y)
        self.canvas.coords(self.cross_v, event.x, 0,
                           event.x, self.canvas.winfo_height())

    # ----------------------------------------------------------------------
    # LEFT CLICK: marker / profile / ROI
    # ----------------------------------------------------------------------
    def on_left_press_dispatch(self, event):
        if event.state & 0x0001:   # SHIFT
            self.profile_start_event(event)
        elif event.state & 0x0004: # CTRL
            self.roi_start_event(event)
        else:
            self._normal_click_start = (event.x, event.y)

    def on_left_motion_dispatch(self, event):
        if event.state & 0x0001:
            self.profile_drag_event(event)
        elif event.state & 0x0004:
            self.roi_drag_event(event)

    def on_left_release_dispatch(self, event):
        if self.profile_active:
            self.profile_end_event(event)
            return
        if self.roi_active:
            self.roi_end_event(event)
            return

        # Normal click → marker + CSV
        if self._normal_click_start:
            ix, iy = self.canvas_to_image(event.x, event.y)
            if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
                val = float(self.frame[iy, ix])
                self.save_click(ix, iy, val)

                # Remove old markers
                for item, _, _ in self.points:
                    self.canvas.delete(item)
                self.points.clear()

                # Add new marker
                r = 4
                cx, cy = event.x, event.y
                p = self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                            fill="red", outline="")
                self.points.append((p, ix, iy))

            self._normal_click_start = None

    # ----------------------------------------------------------------------
    # PROFILE TOOL
    # ----------------------------------------------------------------------
    def profile_start_event(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y)
        if not (0 <= ix < self.img_w and 0 <= iy < self.img_h):
            return
        self.profile_active = True
        self.profile_start = (event.x, event.y)
        if self.profile_line_id:
            self.canvas.delete(self.profile_line_id)
        self.profile_line_id = None

    def profile_drag_event(self, event):
        if not self.profile_active:
            return
        if self.profile_line_id:
            self.canvas.delete(self.profile_line_id)
        self.profile_line_id = self.canvas.create_line(
            self.profile_start[0], self.profile_start[1],
            event.x, event.y, fill="cyan", width=2)

    def profile_end_event(self, event):
        if not self.profile_active:
            return
        self.profile_active = False
        if not self.profile_line_id:
            return

        x0, y0 = self.profile_start
        x1, y1 = event.x, event.y
        ix0, iy0 = self.canvas_to_image(x0, y0)
        ix1, iy1 = self.canvas_to_image(x1, y1)

        if not (0 <= ix0 < self.img_w and 0 <= iy0 < self.img_h):
            return
        if not (0 <= ix1 < self.img_w and 0 <= iy1 < self.img_h):
            return

        profile = self.compute_line_profile(ix0, iy0, ix1, iy1)
        self.show_profile_window(profile)

        self.canvas.delete(self.profile_line_id)
        self.profile_line_id = None

    def compute_line_profile(self, x0, y0, x1, y1):
        length = int(np.hypot(x1 - x0, y1 - y0))
        if length < 2:
            return np.array([])

        xs = np.linspace(x0, x1, length)
        ys = np.linspace(y0, y1, length)

        xs = np.clip(xs, 0, self.img_w-1)
        ys = np.clip(ys, 0, self.img_h-1)

        vals = []
        for x, y in zip(xs, ys):
            x0i = int(np.floor(x))
            x1i = min(x0i + 1, self.img_w - 1)
            y0i = int(np.floor(y))
            y1i = min(y0i + 1, self.img_h - 1)

            dx = x - x0i
            dy = y - y0i

            v00 = self.frame[y0i, x0i]
            v10 = self.frame[y0i, x1i]
            v01 = self.frame[y1i, x0i]
            v11 = self.frame[y1i, x1i]

            top = v00 * (1-dx) + v10 * dx
            bottom = v01 * (1-dx) + v11 * dx
            vals.append(top * (1-dy) + bottom * dy)

        return np.array(vals, dtype=np.float32)

    def show_profile_window(self, profile):
        if profile.size == 0:
            return
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.plot(profile, color="cyan")
        ax.set_title("Line Profile")
        ax.set_xlabel("Distance (pixels)")
        ax.set_ylabel("Intensity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.show()

    # ----------------------------------------------------------------------
    # ROI TOOL
    # ----------------------------------------------------------------------
    def roi_start_event(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y)
        if not (0 <= ix < self.img_w and 0 <= iy < self.img_h):
            return
        self.roi_active = True
        self.roi_start_canvas = (event.x, event.y)
        if self.roi_rect_id:
            self.canvas.delete(self.roi_rect_id)
        self.roi_rect_id = None
        self.roi_bounds_img = None
        self.roi_stats_var.set("ROI: drawing…")

    def roi_drag_event(self, event):
        if not self.roi_active:
            return
        x0, y0 = self.roi_start_canvas
        x1, y1 = event.x, event.y
        if self.roi_rect_id:
            self.canvas.coords(self.roi_rect_id, x0, y0, x1, y1)
        else:
            self.roi_rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="#5FA8FF", width=2)

    def roi_end_event(self, event):
        if not self.roi_active:
            return
        self.roi_active = False
        if not self.roi_rect_id:
            self.roi_stats_var.set("ROI: none")
            return

        cx0, cy0 = self.roi_start_canvas
        cx1, cy1 = event.x, event.y
        ix0, iy0 = self.canvas_to_image(cx0, cy0)
        ix1, iy1 = self.canvas_to_image(cx1, cy1)

        x0, x1 = sorted((max(0, ix0), min(self.img_w-1, ix1)))
        y0, y1 = sorted((max(0, iy0), min(self.img_h-1, iy1)))

        if x1 <= x0 or y1 <= y0:
            self.clear_roi()
            return

        self.roi_bounds_img = (x0, y0, x1, y1)
        self.render()

        roi = self.frame[y0:y1, x0:x1]
        count = roi.size
        vmin = float(np.min(roi))
        vmax = float(np.max(roi))
        vmean = float(np.mean(roi))
        vstd = float(np.std(roi))

        self.roi_stats_var.set(
            f"ROI: {x1-x0}×{y1-y0} "
            f"min={vmin:.3f} max={vmax:.3f} "
            f"mean={vmean:.3f} std={vstd:.3f} n={count}"
        )
        self.update_histogram()

    def clear_roi(self):
        if self.roi_rect_id:
            self.canvas.delete(self.roi_rect_id)
        self.roi_rect_id = None
        self.roi_bounds_img = None
        self.roi_stats_var.set("ROI: none")
        self.update_histogram()

    # ----------------------------------------------------------------------
    # CSV LOGGING
    # ----------------------------------------------------------------------
    def ensure_csv_header(self):
        try:
            with open(self.csv_file, "x", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["x", "y", "value"])
        except FileExistsError:
            pass

    def save_click(self, x, y, val):
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([x, y, val])

    # ----------------------------------------------------------------------
    # HISTOGRAM WINDOW
    # ----------------------------------------------------------------------
    def toggle_histogram(self):
        if self.hist_window:
            self.close_histogram()
            self.hist_btn.config(text="Show Histogram")
        else:
            self.open_histogram()
            self.hist_btn.config(text="Hide Histogram")

    def open_histogram(self):
        self.hist_window = tk.Toplevel(self.root)
        self.hist_window.title("Histogram")

        self.hist_fig = plt.Figure(figsize=(4.5, 3.2), dpi=100)
        self.hist_ax = self.hist_fig.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.hist_window)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_histogram()

    def close_histogram(self):
        if self.hist_window:
            self.hist_window.destroy()
        self.hist_window = None

    def update_histogram(self):
        if not self.hist_window:
            return
        self.hist_ax.clear()

        data = self.frame.flatten()
        self.hist_ax.hist(data, bins=100, color="white",
                          alpha=0.9, label="Image", density=True)

        if self.roi_bounds_img:
            x0, y0, x1, y1 = self.roi_bounds_img
            roi = self.frame[y0:y1, x0:x1].flatten()
            if roi.size > 0:
                self.hist_ax.hist(roi, bins=100, color="#5FA8FF",
                                  alpha=0.6, label="ROI", density=True)

        self.hist_ax.set_title("Histogram")
        self.hist_ax.set_xlabel("Pixel Value")
        self.hist_ax.set_ylabel("Count")
        self.hist_ax.grid(alpha=0.3)
        self.hist_ax.legend(loc="upper right")
        self.hist_canvas.draw()

    # ----------------------------------------------------------------------
    # EXPORT PNG
    # ----------------------------------------------------------------------
    def export_png(self):
        self.canvas.update()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        ps = self.canvas.postscript(colormode='color')
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.crop((0, 0, w, h))

        filename = "export_view.png"
        img.save(filename)
        print(f"Exported view to {filename}")

    # ----------------------------------------------------------------------
    # WINDOW GEOMETRY SAVE/RESTORE
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # PERIODIC REFRESH (MAIN IMAGE UPDATE LOOP)
    # ----------------------------------------------------------------------
    def refresh(self):
        if self.stop_event and self.stop_event.is_set():
            self.stop()
            return

        try:
            raw = self.data['IRCamera.last_frame'][
                :self.data['IRCamera.frame_size'][0]
            ]
            new_frame = np.array(raw).reshape((self.img_w, self.img_h)).T
            self.frame = new_frame
            self.render()
            self.update_histogram()
        except Exception as e:
            log.error(f'error at frame refresh {e}')

        if not (self.stop_event and self.stop_event.is_set()):
            self.root.after(100, self.refresh)

    # ----------------------------------------------------------------------
    # PIEZOS PANEL
    # ----------------------------------------------------------------------
    def _build_piezos_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Piezos", padding=10)
        frame.pack(fill=tk.Y, expand=False)

        row = ttk.Frame(frame)
        row.pack(fill=tk.Y, expand=False)

        self.var_opd = tk.DoubleVar(value=0.0)
        self.var_da1 = tk.DoubleVar(value=0.0)
        self.var_da2 = tk.DoubleVar(value=0.0)

        try:
            init = self.data.get("DAQ.piezos_level", None)
            if init and len(init) >= 3:
                self.var_opd.set(float(init[0]))
                self.var_da1.set(float(init[1]))
                self.var_da2.set(float(init[2]))
        except Exception:
            pass

        def _col(label, var, col):
            col_frame = ttk.Frame(row)
            col_frame.grid(row=0, column=col, padx=10, pady=5)
            ttk.Label(col_frame, text=label).pack(pady=(0, 6))
            scl = ttk.Scale(col_frame, from_=10.0, to=0.0,
                            variable=var, orient=tk.VERTICAL,
                            length=220, command=self._on_piezos_change)
            scl.pack()
            return scl

        self.scale_opd = _col("OPD", self.var_opd, 0)
        self.scale_da1 = _col("DA-1", self.var_da1, 1)
        self.scale_da2 = _col("DA-2", self.var_da2, 2)

        self._write_piezos_to_shared()

    def _write_piezos_to_shared(self):
        vals = [
            float(self.var_opd.get()),
            float(self.var_da1.get()),
            float(self.var_da2.get())
        ]
        self.data["DAQ.piezos_level"][:] = vals

    def _on_piezos_change(self, *_):
        self._write_piezos_to_shared()
