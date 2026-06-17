import numpy as np
import matplotlib.pyplot as plt

# simple LUT builders
def build_lut(name: str):
    name = (name or 'magma').lower()
    if name == 'gray':
        lut = np.linspace(0, 1, 256)
        return np.stack([lut, lut, lut], axis=1)
    try:
        return plt.get_cmap(name, 256)(np.linspace(0, 1, 256))[:, :3]
    except Exception:
        return plt.get_cmap('magma', 256)(np.linspace(0, 1, 256))[:, :3]
    
