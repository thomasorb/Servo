import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import numba as nb

from . import config

def simulate_sin(n):
    x = np.linspace(0, np.random.uniform(3,5)*np.pi, n)
    y = np.sin(x + np.random.uniform(0,1)*np.pi) + 1 + np.random.uniform(0, 2)
    y *= 10**np.random.uniform(2, 5)
    y = np.random.poisson(y)
    return y / np.max(y) * np.random.uniform(0.01, 1)

def sinus(t, A, f, phi, C):
    return A * np.sin(2*np.pi*f*t + phi) + C


def estimate_frequency_fft(y, zp_ratio=10):
    
    y_detrended = y - np.mean(y)
    fft_vals = np.fft.rfft(y_detrended, n=zp_ratio*len(y))
    freqs = np.fft.rfftfreq(len(y), zp_ratio)
    idx = np.argmax(np.abs(fft_vals))
    return freqs[idx]


def fit_sinus(y):
    f0 = estimate_frequency_fft(y)
    A0 = np.diff(np.percentile(y, [2, 98]))[0] / 2
    C0 = np.median(y)
    phi0 = 0

    p0 = [A0, f0, phi0, C0]

    popt, pcov = scipy.optimize.curve_fit(sinus, np.arange(len(y)), y, p0=p0)
    return popt, pcov

def get_min_max(y, plot=False, sinfit=False):
    if sinfit:
        try:
            popt, _ = fit_sinus(y)
            ymax = np.abs(popt[3]) + np.abs(popt[0])
            ymin = np.abs(popt[3]) - np.abs(popt[0])
            if plot:
                plt.figure()
                plt.plot(y)
                plt.plot(sinus(np.arange(len(y)), *popt))
                plt.axhline(ymin, c='red')
                plt.axhline(ymax, c='red')
            
        except Exception as e:
            print(f'error at fft sinus fitting : {e}')
            sinfit = False
            
    if not sinfit: # must be kept like that to recover from exception
        ymin, ymax = np.percentile(y, [4, 96])
        
        if plot:
            plt.figure()
            plt.plot(y)
            plt.axhline(ymin, c='red')
            plt.axhline(ymax, c='red')
    return ymin, ymax

def get_normalization_coeffs(profiles):
    coeffs = list()
    for i in range(profiles.shape[1]):
        isin = profiles[int(len(profiles)*config.NORMALIZATION_LEN_RATIO):,i]
        coeffs.append(get_min_max(isin, plot=False))
    return np.array(coeffs)

def get_roi_normalization_coeffs(rois):
    vmin, vmax = np.nanpercentile(rois[:int(len(rois)*config.NORMALIZATION_LEN_RATIO),::], [4,96], axis=0)
    return vmin, vmax

# def compute_profiles(a, x, y, w, l, sanity_check=True, get_roi=True):
#     assert not(w%2), f'w should be even {w}'
#     assert not(l%2), f'l should be even {l}'
#     hw = w//2
#     hl = l//2
#     hbar = a[max(0, x - hl): min(x + hl, a.shape[0]), 
#              max(0, y - hw): min(y + hw, a.shape[1])]
#     hprofile = np.mean(hbar, axis=1, dtype=a.dtype)
#     vbar = a[max(0, x - hw): min(x + hw, a.shape[0]),
#              max(0, y - hl): min(y + hl, a.shape[1])]
#     vprofile = np.mean(vbar, axis=0, dtype=a.dtype)
    
#     if hprofile.shape[0] != l:
#         _zprofile = np.full(l, np.nan, dtype=a.dtype)
#         if (x - hl) < 0:
#             _zprofile[:hprofile.shape[0]] = hprofile
#         else:
#             _zprofile[-hprofile.shape[0]:] = hprofile
#         hprofile = _zprofile

#     if vprofile.shape[0] != l:
#         _zprofile = np.full(l, np.nan, dtype=a.dtype)
#         if (y - hl) < 0:
#             _zprofile[:vprofile.shape[0]] = vprofile
#         else:
#             _zprofile[-vprofile.shape[0]:] = vprofile
#         vprofile = _zprofile
            
#     if sanity_check:
#         assert len(hprofile) == l, 'profile is taken too near the border given w and l'
#         assert len(vprofile) == l, 'profile is taken too near the border given w and l'

#     if get_roi:
#         roi = a[max(0, x - hl): min(x + hl, a.shape[0]),
#                 max(0, y - hl): min(y + hl, a.shape[1])]
#         roi = np.ascontiguousarray(roi, dtype=config.DATA_DTYPE)
#     else: roi = None

#     hprofile = np.ascontiguousarray(hprofile, dtype=config.DATA_DTYPE)
#     vprofile = np.ascontiguousarray(vprofile, dtype=config.DATA_DTYPE)
#     return hprofile, vprofile, roi
    

import numpy as np
from numba import njit

# ---- Numba core: computes horizontal & vertical profiles only (no ROI) ----
@njit(cache=True, fastmath=False)
def _compute_profiles_core_numba_f32(a, x, y, w, l):
    """
    a: 2D float32 array
    x, y: ints (center)
    w, l: even ints (width of averaging bar, length of profile)
    Returns:
        hprofile, vprofile (both float32, shape (l,))
    """
    # Preconditions equivalent to your asserts
    if (w & 1) or (l & 1):
        raise ValueError("w and l must be even.")

    hw = w // 2
    hl = l // 2
    nrows, ncols = a.shape

    # ----- Horizontal profile -----
    # Horizontal bar spans rows [x-hl, x+hl), columns [y-hw, y+hw)
    row_start = 0 if x - hl < 0 else (x - hl)
    row_end   = nrows if x + hl > nrows else (x + hl)
    col_start_h = 0 if y - hw < 0 else (y - hw)
    col_end_h   = ncols if y + hw > ncols else (y + hw)

    hlen = row_end - row_start
    # temp store of means before padding
    htmp = np.empty(hlen, dtype=np.float32)

    width_h = col_end_h - col_start_h
    for i in range(hlen):
        r = row_start + i
        s = 0.0  # accumulate in float64 inside Numba for accuracy
        for c in range(col_start_h, col_end_h):
            s += float(a[r, c])
        # width_h > 0 by construction unless image is degenerate
        mean_val = s / max(1, width_h)
        htmp[i] = np.float32(mean_val)

    hprofile = np.empty(l, dtype=np.float32)
    # init with NaNs (float32 NaN)
    for i in range(l):
        hprofile[i] = np.float32(np.nan)

    if (x - hl) < 0:
        # too close to top edge -> place from the beginning
        for i in range(hlen):
            hprofile[i] = htmp[i]
    else:
        # too close to bottom edge -> right-aligned
        start_out = l - hlen
        for i in range(hlen):
            hprofile[start_out + i] = htmp[i]

    # ----- Vertical profile -----
    # Vertical bar spans rows [x-hw, x+hw), columns [y-hl, y+hl)
    row_start_v = 0 if x - hw < 0 else (x - hw)
    row_end_v   = nrows if x + hw > nrows else (x + hw)
    col_start = 0 if y - hl < 0 else (y - hl)
    col_end   = ncols if y + hl > ncols else (y + hl)

    vlen = col_end - col_start
    vtmp = np.empty(vlen, dtype=np.float32)

    height_v = row_end_v - row_start_v
    for j in range(vlen):
        c = col_start + j
        s = 0.0
        for r in range(row_start_v, row_end_v):
            s += float(a[r, c])
        mean_val = s / max(1, height_v)
        vtmp[j] = np.float32(mean_val)

    vprofile = np.empty(l, dtype=np.float32)
    for i in range(l):
        vprofile[i] = np.float32(np.nan)

    if (y - hl) < 0:
        # too close to left edge -> place from the beginning
        for j in range(vlen):
            vprofile[j] = vtmp[j]
    else:
        # too close to right edge -> right-aligned
        start_out = l - vlen
        for j in range(vlen):
            vprofile[start_out + j] = vtmp[j]

    return hprofile, vprofile


# ---- Drop-in wrapper: same signature & behavior as your original ----
def compute_profiles(a, x, y, w, l, sanity_check=True, get_roi=True):
    """
    Drop-in accelerated version.
    Assumptions:
      - a is a 2D array of dtype float32 (will be made contiguous if needed)
      - Returns (hprofile, vprofile, roi) all float32 (roi can be None)
    """
    # Keep your even-size checks (will also be checked inside the JIT core)
    assert not (w % 2), f"w should be even {w}"
    assert not (l % 2), f"l should be even {l}"

    # Ensure a is float32 & contiguous for best throughput
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    a_contig = np.ascontiguousarray(a, dtype=np.float32)

    # Call Numba-compiled core (first call triggers JIT compilation)
    hprofile, vprofile = _compute_profiles_core_numba_f32(a_contig, int(x), int(y), int(w), int(l))

    # Sanity check preserved (your original intent)
    if sanity_check:
        assert len(hprofile) == l, "profile is taken too near the border given w and l"
        assert len(vprofile) == l, "profile is taken too near the border given w and l"

    # ROI handling (float32, contiguous), identical spatial logic
    if get_roi:
        hl = l // 2
        r0 = max(0, x - hl)
        r1 = min(x + hl, a.shape[0])
        c0 = max(0, y - hl)
        c1 = min(y + hl, a.shape[1])
        roi = np.ascontiguousarray(a_contig[r0:r1, c0:c1], dtype=np.float32)
    else:
        roi = None

    # Ensure float32 & contiguous for outputs (already the case)
    hprofile = np.ascontiguousarray(hprofile, dtype=np.float32)
    vprofile = np.ascontiguousarray(vprofile, dtype=np.float32)
    return hprofile, vprofile, roi

def validate_roi_length(length):
    length = int(length) // 8 * 8
    length = np.clip(length, config.MIN_ROI_SHAPE, min(config.FULL_FRAME_SHAPE))
    return length
    
def validate_roi_shape(shape):
    shape = np.copy(shape)
    shape[0] = shape[0] // 8 * 8
    shape[1] = shape[1] // 8 * 8
    shape[0] = np.clip(shape[0], config.MIN_ROI_SHAPE, config.FULL_FRAME_SHAPE[0])
    shape[1] = np.clip(shape[1], config.MIN_ROI_SHAPE, config.FULL_FRAME_SHAPE[1])
    return shape

def validate_roi_position(position):
    position = np.copy(position)
    position[0] = position[0] // 4 * 4
    position[1] = position[1] // 4 * 4
    position[0] = np.clip(position[0], 0, config.FULL_FRAME_SHAPE[0] - 1)
    position[1] = np.clip(position[1], 0, config.FULL_FRAME_SHAPE[1] - 1)
    return position
    

def get_pixels_lists(states):
    left = list()
    center = list()
    right = list()
    center_passed = False
    for i in range(len(states)):
        if states[i] == 1:
            if not center_passed:
                left.append(i)
            else:
                right.append(i)
                
        elif states[i] == 2:
            center.append(i)
            if not center_passed:
                center_passed = True

    left = np.ascontiguousarray(left, dtype=np.int32)
    center = np.ascontiguousarray(center, dtype=np.int32)
    right = np.ascontiguousarray(right, dtype=np.int32)
    return left, center, right

def get_mean_pixels_positions(pixels_lists):
    pos = list()
    for ilist in pixels_lists:
        pos.append((np.max(ilist) + np.min(ilist)) / 2)
    return pos
        

@nb.njit(fastmath=True)
def normalize_profile(a, vmin, vmax, inplace=False):
    """
    a, vmin, vmax: tableaux 1D (N,) de même taille et dtype float32/float64.
    Retourne un nouveau tableau normé et clampé à [0, 1].
    """
    n = a.size
    if inplace: out = a
    else: out = np.empty_like(a)
    for i in range(n):
        denom = vmax[i] - vmin[i]
        if denom <= 0.0:
            out[i] = 0.0
        else:
            x = (a[i] - vmin[i]) / denom
            # clamp manuel : plus rapide qu'un appel à np.clip dans la boucle
            if x < 0.0:
                x = 0.0
            elif x > 1.0:
                x = 1.0
            out[i] = x
    return out

# @nb.njit(fastmath=True)
# def normalize_and_compute_profile_level(a, vmin, vmax, pixels_list):
#     """
#     """
#     n = a.size
#     m = pixels_list.size
#     b = 0
#     for j in range(m):
#         i = pixels_list[j]
#         denom = vmax[i] - vmin[i]
#         if denom <= 0.0:
#             a[i] = 0.0
#         else:
#             x = (a[i] - vmin[i]) / denom
#             if x < 0.0:
#                 x = 0.0
#             elif x > 1.0:
#                 x = 1.0
#             b += x
#     return b / float(m)



@nb.njit(fastmath=True, cache=True)
def normalize_and_compute_profile_level(a, vmin, vmax, pixels_list):
    """
    Calcule la moyenne des valeurs normalisées et clampées sur les indices de pixels_list.
    Hypothèses:
      - a, vmin, vmax: float32 1D contigus (même taille)
      - pixels_list: int32 1D contigu
    Retour: float32 (moyenne sur pixels_list)
    """
    m = pixels_list.size
    if m == 0:
        return np.float32(0.0)

    acc = 0.0  # double interne pour sommation plus stable, retourné en float32
    for j in range(m):
        i = pixels_list[j]
        # Accès local -> variables locales pour limiter les lectures RAM
        ai   = a[i]
        vmin_i = vmin[i]
        vmax_i = vmax[i]
        denom = vmax_i - vmin_i

        if denom <= 0.0:
            x = 0.0
        else:
            x = (ai - vmin_i) / denom
            # clamp manuel (plus rapide qu'un appel fonction dans la boucle)
            if x < 0.0:
                x = 0.0
            elif x > 1.0:
                x = 1.0

        acc += x

    return np.float32(acc / m)


def compute_profile_levels(profile, pixels_lists, mean=True):
    left_level = profile[pixels_lists[0]]
    center_level = profile[pixels_lists[1]]
    right_level = profile[pixels_lists[2]]

    if mean:
        return np.mean(left_level), np.mean(center_level), np.mean(right_level)
    else:
        return np.median(left_level), np.median(center_level), np.median(right_level)


def compute_profiles_levels(profiles, normalization_coeffs, pixels_lists):
    """Return an array of levels for each profile in profiles."""
    levels = list()
    for i in range(profiles.shape[0]):
        inorm = normalize_profile(profiles[i],
                                  normalization_coeffs[:,0],
                                  normalization_coeffs[:,1])
        levels.append(compute_profile_levels(inorm, pixels_lists, mean=True))
    return np.array(levels)

def get_ellipse_normalization_coeffs(profiles, normalization_coeffs, pixels_lists):
    levels = compute_profiles_levels(profiles, normalization_coeffs, pixels_lists)
    levels = levels[int(config.NORMALIZATION_LEN_RATIO*len(levels)):,:]
    low_center_level, high_center_level = np.nanpercentile(levels[:,1], [1, 99])
    
    left_norm = (np.nanmean(levels[np.nonzero(levels[:,1] <= low_center_level),0]),
                 np.nanmean(levels[np.nonzero(levels[:,1] >= high_center_level),0]))

    right_norm = (np.nanmean(levels[np.nonzero(levels[:,1] <= low_center_level),2]),
                  np.nanmean(levels[np.nonzero(levels[:,1] >= high_center_level),2]))

    return np.array(list(left_norm) + list(right_norm))

@njit(cache=True, fastmath=False)
def unwrap_scalar_2pi(angle, last_angle):
    # Numba n'aime pas toujours np.float32(np.pi), on force en double puis on cast
    two_pi = 2.0 * np.pi
    pi = np.pi

    # On fait le calcul en float64 pour la robustesse, puis on cast en float32 à la fin
    a = float(angle)
    la = float(last_angle)

    delta = a - la
    delta = (delta + pi) % two_pi - pi

    return np.float32(la + delta)

@njit(cache=True, fastmath=True)
def compute_angles(levels, ellipse_norm_coeffs, last_angles):
    l0 = levels[0] - levels[1] * (ellipse_norm_coeffs[1] - ellipse_norm_coeffs[0]) - ellipse_norm_coeffs[0]
    l2 = levels[2] - levels[1] * (ellipse_norm_coeffs[3] - ellipse_norm_coeffs[2]) - ellipse_norm_coeffs[2]
    l1 = levels[1] - 0.5

    return (unwrap_scalar_2pi(np.arctan2(l0, l1), last_angles[0]),
            unwrap_scalar_2pi(np.arctan2(l2, l1), last_angles[1]))


@njit(cache=True, fastmath=True)
def compute_opds(unwrapped_angles):
    return -config.CALIBRATION_LASER_WAVELENGTH * unwrapped_angles / (2*np.pi)

@njit(cache=True, fastmath=True)
def mean(a):
    n = a.size

    b = np.float32(0)
    for i in range(n):
        b += a[i]
        
    return np.float32(b/n)
