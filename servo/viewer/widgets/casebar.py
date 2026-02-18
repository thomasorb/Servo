import tkinter as tk

class CaseBar(tk.Frame):
    """Clickable state bar (0/1/2 per cell)."""
    DEFAULT_COLORS = (
        {"fill": "#f6f6f6", "outline": "#bdbdbd"},
        {"fill": "#1677ff", "outline": "#0b59c8"},
        {"fill": "#2ecc71", "outline": "#239b56"},
    )
    ORDER = [0, 1, 2]

    def __init__(self, parent, count=8, height=28, gap=2, padding=8, min_cell_w=6,
                 colors=None, on_change=None, states=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.count = int(count)
        self.height = int(height)
        self.gap = int(gap)
        self.padding = int(padding)
        self.min_cell_w = int(min_cell_w)
        self.colors = (colors or self.DEFAULT_COLORS)
        self.on_change = on_change
        self.states = [0] * self.count if states is None else list(states)[:self.count]
        self.canvas = tk.Canvas(self, height=self.height, bg=self['bg'] if 'bg' in self.keys() else 'white', highlightthickness=0)
        self.canvas.pack(fill='x', expand=True)
        self.menu = tk.Menu(self, tearoff=False)
        for st in self.ORDER:
            self.menu.add_command(label=f"Set: {st}", command=lambda s=st: self._ctx_set(s))
        self.canvas.bind('<Configure>', self._on_resize)
        self.canvas.bind('<Button-1>', self._on_left_click)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Motion>', lambda e: self._hover_cursor(e))
        self._ctx_index = None
        self._layout = []
        self._items = []
        self._redraw_all()

    # public
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
        states = list(states)
        if len(states) > self.count:
            self.states = states[:self.count]
        else:
            self.states = states + [0] * (self.count - len(states))
        if hasattr(self, 'canvas'):
            self._redraw_all()
            self._notify(-1, None)

    def get_states(self):
        return list(self.states)

    # draw
    def _on_resize(self, _=None):
        self._redraw_all()

    def _compute_layout(self):
        W = max(1, self.canvas.winfo_width())
        H = self.height
        self.canvas.config(height=H)
        inner_w = max(1, W - 2 * self.padding)
        if self.count <= 0: return [], 0
        cell_w = (inner_w - self.gap * (self.count - 1)) / self.count
        cell_w = max(self.min_cell_w, int(cell_w))
        total_w = cell_w * self.count + self.gap * (self.count - 1)
        x = self.padding
        y0 = self.padding // 2
        y1 = H - self.padding // 2
        layout = []
        for i in range(self.count):
            x0 = x + i * (cell_w + self.gap)
            x1 = x0 + cell_w
            layout.append((x0, y0, x1, y1))
        return layout, total_w

    def _redraw_all(self):
        self.canvas.delete('all')
        self._items.clear()
        self._layout, _ = self._compute_layout()
        for i, bbox in enumerate(self._layout):
            item = self._draw_one(i, bbox, self.states[i])
            self._items.append(item)

    def _draw_one(self, idx, bbox, state):
        x0, y0, x1, y1 = bbox
        c = self.colors[state]
        return self.canvas.create_rectangle(x0, y0, x1, y1, fill=c['fill'], outline=c['outline'], width=1,
                                            tags=('cell', f'cell-{idx}'))

    def _paint_cell(self, idx):
        if not (0 <= idx < len(self._items)): return
        item = self._items[idx]
        st = self.states[idx]
        c = self.colors[st]
        self.canvas.itemconfig(item, fill=c['fill'], outline=c['outline'])

    # ui
    def _index_from_x(self, x):
        for i, (x0, _y0, x1, _y1) in enumerate(self._layout):
            if x0 <= x <= x1: return i
        return None

    def _hover_cursor(self, event):
        idx = self._index_from_x(event.x)
        self.canvas.config(cursor='hand2' if idx is not None else '')

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
