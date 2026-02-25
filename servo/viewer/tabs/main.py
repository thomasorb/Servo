import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from ..widgets.casebar import CaseBar
from ... import config

class MainTab:
    """Main image + profiles + ROI preview."""

    def __init__(self, parent, viewer):
        self.viewer = viewer
        self.root = ttk.Frame(parent)
        self.root.pack(fill=tk.BOTH, expand=True)

        # left column
        left = ttk.Frame(self.root)
        left.pack(fill=tk.BOTH, expand=True)

        # image canvas
        canvas_frame = ttk.Frame(left)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # profiles
        profiles = ttk.Frame(left)
        profiles.pack(fill=tk.X, expand=False, pady=10)
        self._build_profiles(profiles)

        # image state + mouse
        self._init_image_state()
        self._bind_mouse()

    # profiles
    def _build_profiles(self, parent):
        self.hlevels_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)
        self.vlevels_buf = deque(maxlen=config.VIEWER_BUFFER_SIZE)

        # H row
        h_row = ttk.Frame(parent)
        h_row.pack(fill=tk.X, expand=False, pady=6)
        h_left = ttk.Frame(h_row)
        h_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.fig_h = plt.Figure(figsize=(6, 1.8), dpi=90)
        self.ax_h = self.fig_h.add_subplot(111)
        self.ax_h.set_title('Horizontal profile')
        self.ax_h.set_ylim(0, 1)
        self.canvas_h = FigureCanvasTkAgg(self.fig_h, master=h_left)
        self.canvas_h.get_tk_widget().pack(fill=tk.X, padx=8, pady=4)
        self.hbar = CaseBar(
            h_left,
            count=int(self.viewer.data['IRCamera.profile_len'][0]),
            height=28,
            states=self.viewer.data['Servo.pixels_x'][:].astype(int).tolist(),
            on_change=self._on_hbar_change,
        )
        self.hbar.pack(fill=tk.X, padx=8, pady=(0, 8))
        h_right = ttk.Frame(h_row, width=240)
        h_right.pack(side=tk.RIGHT, fill=tk.Y, padx=8)
        self.fig_ellx = plt.Figure(figsize=(3.0, 1.8), dpi=90)
        self.ax_ellx = self.fig_ellx.add_subplot(111)
        self.ax_ellx.set_title('ellipse_x')
        self.canvas_ellx = FigureCanvasTkAgg(self.fig_ellx, master=h_right)
        self.canvas_ellx.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # V row
        v_row = ttk.Frame(parent)
        v_row.pack(fill=tk.X, expand=False, pady=6)
        v_left = ttk.Frame(v_row)
        v_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.fig_v = plt.Figure(figsize=(6, 1.8), dpi=90)
        self.ax_v = self.fig_v.add_subplot(111)
        self.ax_v.set_title('Vertical profile')
        self.ax_v.set_ylim(0, 1)
        self.canvas_v = FigureCanvasTkAgg(self.fig_v, master=v_left)
        self.canvas_v.get_tk_widget().pack(fill=tk.X, padx=8, pady=4)
        self.vbar = CaseBar(
            v_left,
            count=int(self.viewer.data['IRCamera.profile_len'][0]),
            height=28,
            states=self.viewer.data['Servo.pixels_y'][:].astype(int).tolist(),
            on_change=self._on_vbar_change,
        )
        self.vbar.pack(fill=tk.X, padx=8, pady=(0, 8))
        v_right = ttk.Frame(v_row, width=240)
        v_right.pack(side=tk.RIGHT, fill=tk.Y, padx=8)
        self.fig_elly = plt.Figure(figsize=(3.0, 1.8), dpi=90)
        self.ax_elly = self.fig_elly.add_subplot(111)
        self.ax_elly.set_title('ellipse_y')
        self.canvas_elly = FigureCanvasTkAgg(self.fig_elly, master=v_right)
        self.canvas_elly.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_hbar_change(self, index, state, all_states):
        try:
            n = int(self.viewer.data['IRCamera.profile_len'][0])
            self.viewer.data['Servo.pixels_x'][:n] = [int(s) for s in all_states]
        except Exception:
            pass

    def _on_vbar_change(self, index, state, all_states):
        try:
            n = int(self.viewer.data['IRCamera.profile_len'][0])
            self.viewer.data['Servo.pixels_y'][:n] = [int(s) for s in all_states]
        except Exception:
            pass

    # image
    def _init_image_state(self):
        self.img_w = int(self.viewer.data['IRCamera.frame_dimx'][0])
        self.img_h = int(self.viewer.data['IRCamera.frame_dimy'][0])
        self.frame = np.random.rand(self.img_h, self.img_w).astype(np.float32)
        self.tk_image = None
        self.image_id = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.fitted = False
        self.cross_h = self.canvas.create_line(0, 0, 0, 0, fill='yellow')
        self.cross_v = self.canvas.create_line(0, 0, 0, 0, fill='yellow')
        self.canvas.tag_raise(self.cross_h)
        self.canvas.tag_raise(self.cross_v)
        self.points = []
        self._roi_mode = False
        self._show_normalized = True
        self.brightness = 0.0
        self.contrast = 1.0
        self._last_bc_pt = None

        # initial marker
        try:
            self.add_marker(int(self.viewer.data['IRCamera.profile_x'][0]),
                            int(self.viewer.data['IRCamera.profile_y'][0]))
        except Exception:
            pass

    def _bind_mouse(self):
        self.canvas.bind('<Configure>', self.on_resize)
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonPress-1>', self._on_left_press)
        self.canvas.bind('<B1-Motion>', self._on_left_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_left_release)
        self.canvas.bind('<Button-4>', lambda e: self.zoom_at(e.x, e.y, 1.1))
        self.canvas.bind('<Button-5>', lambda e: self.zoom_at(e.x, e.y, 0.9))
        self.canvas.bind('<ButtonPress-3>', self._bc_start)
        self.canvas.bind('<B3-Motion>', self._bc_drag)
        self.canvas.bind('<ButtonRelease-3>', self._bc_end)
        self._normal_click_start = None

    # render helpers
    def _stretch(self, frame):
        fmin, fmax = frame.min(), frame.max()
        if fmax <= fmin:
            return np.zeros_like(frame)
        norm = (frame - fmin) / (fmax - fmin)
        norm = (norm - 0.5) * self.contrast + 0.5 + self.brightness
        return np.clip(norm, 0, 1)

    def _apply_lut(self, stretched):
        idx = (stretched * 255).astype(np.uint8)
        rgb = self.viewer.current_lut[idx]
        return (rgb * 255).astype(np.uint8)

    def render(self):
        # main image
        stretched = self._stretch(self.frame.T)
        rgb = self._apply_lut(stretched)
        dw, dh = int(self.img_w * self.scale), int(self.img_h * self.scale)
        pil = Image.fromarray(rgb, 'RGB').resize((dw, dh), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(pil)
        if self.image_id is None:
            self.image_id = self.canvas.create_image(self.offset_x, self.offset_y, anchor='nw', image=self.tk_image)
        else:
            self.canvas.itemconfig(self.image_id, image=self.tk_image)
            self.canvas.coords(self.image_id, self.offset_x, self.offset_y)
        # markers
        r = 4
        for item_id, ix, iy in self.points:
            cx, cy = self.image_to_canvas(ix, iy)
            self.canvas.coords(item_id, cx - r, cy - r, cx + r, cy + r)
            self.canvas.tag_raise(item_id)

    # zoom/pan
    def fit_image(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1 or ch <= 1 or self.img_w <= 0 or self.img_h <= 0:
            return
        self.scale = min(cw / self.img_w, ch / self.img_h)
        self.offset_x = (cw - self.img_w * self.scale) / 2
        self.offset_y = (ch - self.img_h * self.scale) / 2
        self.fitted = True
        self.render()

    def on_resize(self, _e=None):
        if not self.fitted:
            self.fit_image()

    def reset_zoom(self):
        self.fitted = False
        self.fit_image()

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

    # coords
    def canvas_to_image(self, cx, cy):
        ix = int((cx - self.offset_x) / self.scale)
        iy = int((cy - self.offset_y) / self.scale)
        return ix, iy

    def image_to_canvas(self, ix, iy):
        return ix * self.scale + self.offset_x, iy * self.scale + self.offset_y

    # mouse
    def on_mouse_move(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y)
        if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
            self.viewer.pos_var.set(f"x = {ix}, y = {iy}")
            try:
                self.viewer.val_var.set(f"{float(self.frame[ix, iy]):.3f}")
            except Exception:
                self.viewer.val_var.set('—')
        else:
            self.viewer.pos_var.set('x = —, y = —')
            self.viewer.val_var.set('—')
        self.canvas.coords(self.cross_h, 0, event.y, self.canvas.winfo_width(), event.y)
        self.canvas.coords(self.cross_v, event.x, 0, event.x, self.canvas.winfo_height())

    def _on_left_press(self, event):
        self._normal_click_start = (event.x, event.y)

    def _on_left_drag(self, event):
        pass

    def add_marker(self, ix, iy):
        if self._roi_mode: return
        for item, _, _ in self.points:
            self.canvas.delete(item)
        self.points.clear()
        cx, cy = self.image_to_canvas(ix, iy)
        r = 4
        p = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill='red', outline='')
        self.points.append((p, ix, iy))

    def _on_left_release(self, event):
        if self._roi_mode:
            return
        ix, iy = self.canvas_to_image(event.x, event.y)
        if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
            self.add_marker(ix, iy)
            try:
                self.viewer.data['IRCamera.profile_x'][0] = int(ix)
                self.viewer.data['IRCamera.profile_y'][0] = int(iy)
            except Exception:
                pass
        self._normal_click_start = None

    # brightness/contrast
    def _bc_start(self, event):
        self._last_bc_pt = (event.x, event.y)

    def _bc_drag(self, event):
        if not self._last_bc_pt:
            return
        dx = event.x - self._last_bc_pt[0]
        dy = event.y - self._last_bc_pt[1]
        self.brightness += dy * 0.002
        self.contrast *= (1 + dx * 0.002)
        self.contrast = max(0.05, min(self.contrast, 5))
        self._last_bc_pt = (event.x, event.y)
        self.render()

    def _bc_end(self, _e=None):
        self._last_bc_pt = None

    # profiles update
    def update_profiles(self):
        n = int(self.viewer.data['IRCamera.profile_len'][0])
        hlevels = self.viewer.data['IRCamera.hprofile_levels'][:n]
        vlevels = self.viewer.data['IRCamera.vprofile_levels'][:n]
        hpos = self.viewer.data['IRCamera.hprofile_levels_pos'][:n]
        vpos = self.viewer.data['IRCamera.vprofile_levels_pos'][:n]
        if self.viewer.show_normalized:
            horiz = self.viewer.data['IRCamera.hprofile_normalized'][:n]
            vert  = self.viewer.data['IRCamera.vprofile_normalized'][:n]
        else:
            horiz = self.viewer.data['IRCamera.hprofile'][:n]
            vert  = self.viewer.data['IRCamera.vprofile'][:n]

        def draw(fig, ax, canvas, dat, title, levels, levels_pos):
            ax.clear()
            if self.viewer.show_normalized:
                ax.errorbar(levels_pos, levels, xerr=5, yerr=0, color='0.7',
                            marker='o', ls='None', capsize=0, elinewidth=4, markersize=0)
            ax.plot(dat, color='tab:orange', label=title, marker='s')
            ax.axhline(0.75, color='tab:gray')
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.set_xlim(-0.75, len(dat) - 0.25)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_position([0, 0, 1, 1])
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.legend(loc='upper right')
            canvas.draw()

        def draw_ellipse(fig, ax, canvas, levels):
            ax.clear()
            def get_levels(index):
                return [lvl[index] for lvl in levels]
            cen = get_levels(1)
            left = get_levels(0)
            right = get_levels(2)
            nbuf = min(config.VIEWER_ELLIPSE_DRAW_BUFFER_SIZE, len(cen))
            if len(cen):
                ax.scatter(cen[0], left[0], color='tab:blue')
                ax.scatter(cen[0], right[0], color='tab:orange')
                ax.scatter(cen[:nbuf], left[:nbuf], color='tab:blue', alpha=0.2)
                ax.scatter(cen[:nbuf], right[:nbuf], color='tab:orange', alpha=0.2)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_position([0,0,1,1])
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            for v in (0, 0.5, 1):
                ax.axvline(v, c='0.8' if v in (0,1) else '0.5')
                ax.axhline(v, c='0.8' if v in (0,1) else '0.5')
            canvas.draw()

        draw(self.fig_h, self.ax_h, self.canvas_h, horiz, 'h profile', hlevels, hpos)
        draw(self.fig_v, self.ax_v, self.canvas_v, vert,  'v profile', vlevels, vpos)
        self.hlevels_buf.appendleft(np.copy(hlevels))
        self.vlevels_buf.appendleft(np.copy(vlevels))
        draw_ellipse(self.fig_ellx, self.ax_ellx, self.canvas_ellx, self.hlevels_buf)
        draw_ellipse(self.fig_elly, self.ax_elly, self.canvas_elly, self.vlevels_buf)
