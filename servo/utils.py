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


@nb.njit(cache=True, fastmath=True)
def _integral_image_f32(a):
    """
    Build 2D integral image (inclusive prefix sums) for float32 input.
    out[i, j] = sum of a[0:i, 0:j] including row i-1, col j-1 in 1-based sense.
    We use (nrows+1, ncols+1) to avoid bounds checks on rectangle queries.
    """
    nrows, ncols = a.shape
    I = np.zeros((nrows + 1, ncols + 1), dtype=np.float32)
    for r in range(1, nrows + 1):
        row_sum = 0.0
        for c in range(1, ncols + 1):
            row_sum += a[r - 1, c - 1]
            I[r, c] = I[r - 1, c] + row_sum
    return I

@nb.njit(inline='always')
def _rect_sum(I, r0, c0, r1, c1):
    """
    Sum over a[r0:r1, c0:c1] using integral image I (with +1 padding).
    r1 and c1 are exclusive bounds.
    """
    # shift by +1 due to integral image shape
    r0p = r0 + 1
    c0p = c0 + 1
    r1p = r1 + 1
    c1p = c1 + 1
    return I[r1p, c1p] - I[r0p, c1p] - I[r1p, c0p] + I[r0p, c0p]

# @nb.njit(cache=True, fastmath=True)
# def _compute_profiles_integral_f32(a, x, y, w, l):
#     """
#     Integral-image-based profiles:
#     - Horizontal profile: mean of vertical strip width=w centered at y, for l rows around x.
#     - Vertical profile: mean of horizontal strip width=w centered at x, for l cols around y.
#     Handles borders by clamping and padding with NaNs to keep length l.
#     """
#     if (w & 1) or (l & 1):
#         raise ValueError("w and l must be even.")

#     nrows, ncols = a.shape
#     hw = w // 2
#     hl = l // 2

#     I = _integral_image_f32(a)

#     # Horizontal profile (vary rows, fixed vertical strip over columns)
#     # rows [x-hl, x+hl), cols [y-hw, y+hw)
#     r0_h = max(0, x - hl)
#     r1_h = min(nrows, x + hl)
#     c0_h = max(0, y - hw)
#     c1_h = min(ncols, y + hw)
#     hlen = r1_h - r0_h
#     htmp = np.empty(hlen, dtype=np.float32)

#     width_h = max(1, c1_h - c0_h)
#     for i in range(hlen):
#         r = r0_h + i
#         s = _rect_sum(I, r, c0_h, r + 1, c1_h)  # 1-row tall strip
#         htmp[i] = np.float32(s / width_h)

#     hprofile = np.empty(l, dtype=np.float32)
#     # fill with NaN
#     for i in range(l):
#         hprofile[i] = np.float32(np.nan)
#     if (x - hl) < 0:
#         for i in range(hlen):
#             hprofile[i] = htmp[i]
#     else:
#         start = l - hlen
#         for i in range(hlen):
#             hprofile[start + i] = htmp[i]

#     # Vertical profile (vary cols, fixed horizontal strip over rows)
#     # rows [x-hw, x+hw), cols [y-hl, y+hl)
#     r0_v = max(0, x - hw)
#     r1_v = min(nrows, x + hw)
#     c0_v = max(0, y - hl)
#     c1_v = min(ncols, y + hl)
#     vlen = c1_v - c0_v
#     vtmp = np.empty(vlen, dtype=np.float32)

#     height_v = max(1, r1_v - r0_v)
#     for j in range(vlen):
#         c = c0_v + j
#         s = _rect_sum(I, r0_v, c, r1_v, c + 1)  # 1-col wide strip
#         vtmp[j] = np.float32(s / height_v)

#     vprofile = np.empty(l, dtype=np.float32)
#     for i in range(l):
#         vprofile[i] = np.float32(np.nan)
#     if (y - hl) < 0:
#         for j in range(vlen):
#             vprofile[j] = vtmp[j]
#     else:
#         start = l - vlen
#         for j in range(vlen):
#             vprofile[start + j] = vtmp[j]

#     return hprofile, vprofile

# def compute_profiles(a, x, y, w, l, sanity_check=True, get_roi=True):
#     """
#     Drop-in accelerated version using integral images.
#     Assumptions:
#       - a is a 2D array, will be promoted to float32 contiguous for speed.
#       - x, y, w, l as in the original implementation.
#     Returns:
#       - hprofile, vprofile (float32, shape=(l,))
#       - roi: contiguous float32 subregion if requested (same convention as before).
#     """
#     assert not (w % 2), f"w should be even {w}"
#     assert not (l % 2), f"l should be even {l}"

#     if a.dtype != np.float32:
#         a = a.astype(np.float32, copy=False)
#     a_contig = np.ascontiguousarray(a, dtype=np.float32)

#     hprofile, vprofile = _compute_profiles_integral_f32(a_contig, int(x), int(y), int(w), int(l))

#     if sanity_check:
#         assert len(hprofile) == l, "profile is taken too near the border given w and l"
#         assert len(vprofile) == l, "profile is taken too near the border given w and l"

#     roi = None
#     if get_roi:
#         hl = l // 2
#         r0 = max(0, x - hl)
#         r1 = min(x + hl, a_contig.shape[0])
#         c0 = max(0, y - hl)
#         c1 = min(y + hl, a_contig.shape[1])
#         roi = np.ascontiguousarray(a_contig[r0:r1, c0:c1], dtype=np.float32)

#     return hprofile, vprofile, roi

@nb.njit(cache=True, fastmath=True)
def _compute_profiles_local_f32(a, x, y, w, l):
    """
    Profils par calcul local (O(l*w)), sans image intégrale :
      - hprofile[i] = moyenne sur la "bande verticale" (colonnes y-hw:y+hw-1) à la ligne r = x - hl + i
      - vprofile[i] = moyenne sur la "bande horizontale" (lignes  x-hw:x+hw-1) à la colonne c = y - hl + i
    Les positions hors image sont remplies par NaN pour conserver une longueur l.
    """
    nrows, ncols = a.shape
    hw = w // 2
    hl = l // 2
    hprofile = np.empty(l, dtype=np.float32)
    vprofile = np.empty(l, dtype=np.float32)

    # Horizontal profile
    c0 = y - hw
    c1 = y + hw
    for i in range(l):
        r = x - hl + i
        if 0 <= r < nrows:
            cc0 = 0 if c0 < 0 else c0
            cc1 = ncols if c1 > ncols else c1
            width = cc1 - cc0
            if width <= 0:
                hprofile[i] = np.float32(np.nan)
            else:
                s = 0.0
                for c in range(cc0, cc1):
                    s += a[r, c]
                hprofile[i] = np.float32(s / width)
        else:
            hprofile[i] = np.float32(np.nan)

    # Vertical profile
    r0 = x - hw
    r1 = x + hw
    for i in range(l):
        c = y - hl + i
        if 0 <= c < ncols:
            rr0 = 0 if r0 < 0 else r0
            rr1 = nrows if r1 > nrows else r1
            height = rr1 - rr0
            if height <= 0:
                vprofile[i] = np.float32(np.nan)
            else:
                s = 0.0
                for r in range(rr0, rr1):
                    s += a[r, c]
                vprofile[i] = np.float32(s / height)
        else:
            vprofile[i] = np.float32(np.nan)

    return hprofile, vprofile

def compute_profiles(a, x, y, w, l, get_roi=True):
    """
    Drop-in replacement de compute_profiles() basé sur calcul local.
    - a est 2D ; converti/contigu en float32 si besoin (sans copie si déjà bon).
    - x, y, w, l identiques à l’API actuelle.
    Retourne (hprofile, vprofile, roi) pour compat.
    """
    assert not (w % 2), f"w should be even {w}"
    assert not (l % 2), f"l should be even {l}"

    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    a_contig = np.ascontiguousarray(a, dtype=np.float32)

    hprofile, vprofile = _compute_profiles_local_f32(a_contig, int(x), int(y), int(w), int(l))

    roi = None
    if get_roi:
        hl = l // 2
        r0 = max(0, x - hl); r1 = min(x + hl, a_contig.shape[0])
        c0 = max(0, y - hl); c1 = min(y + hl, a_contig.shape[1])
        roi = np.ascontiguousarray(a_contig[r0:r1, c0:c1], dtype=np.float32)
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

@nb.njit(cache=True, fastmath=False)
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

@nb.njit(cache=True, fastmath=True)
def compute_angles(levels, ellipse_norm_coeffs, last_angles):
    l0 = levels[0] - levels[1] * (ellipse_norm_coeffs[1] - ellipse_norm_coeffs[0]) - ellipse_norm_coeffs[0]
    l2 = levels[2] - levels[1] * (ellipse_norm_coeffs[3] - ellipse_norm_coeffs[2]) - ellipse_norm_coeffs[2]
    l1 = levels[1] - 0.5

    return (unwrap_scalar_2pi(np.arctan2(l0, l1), last_angles[0]),
            unwrap_scalar_2pi(np.arctan2(l2, l1), last_angles[1]))


@nb.njit(cache=True, fastmath=True)
def compute_opds(unwrapped_angles):
    return -config.CALIBRATION_LASER_WAVELENGTH * unwrapped_angles / (2*np.pi)

@nb.njit(cache=True, fastmath=True)
def mean(a):
    n = a.size

    b = np.float32(0)
    for i in range(n):
        b += a[i]
        
    return np.float32(b/n)

@nb.njit(cache=True, fastmath=True)
def copy_transpose_to_1d(src_2d, dst_1d):
    """
    Écrit src_2d.T aplati (ordre C) dans dst_1d sans allouer de temporaire.
    Équivalent à dst_1d[:] = src_2d.T.ravel() mais sans créer l'intermédiaire.
    """
    nrows, ncols = src_2d.shape
    k = 0
    # Parcours colonne-par-colonne de src (équiv. à a.T.ravel(order='C'))
    for c in range(ncols):
        for r in range(nrows):
            dst_1d[k] = src_2d[r, c]
            k += 1

@nb.njit(cache=True, fastmath=False)
def publish_timers_ns(timers_ns, timers_version, v0, v1, v2, v3, v4):
    """
    Atomically publish 5 int64 nanosecond durations into shared memory using a seqlock-like protocol.
    Writer: increment version to odd, write, increment to even.
    Readers: only trust even versions with same value before/after read.
    """
    # bump to odd
    timers_version[0] += 1
    if (timers_version[0] & 1) == 0:
        timers_version[0] += 1

    timers_ns[0] = v0
    timers_ns[1] = v1
    timers_ns[2] = v2
    timers_ns[3] = v3
    timers_ns[4] = v4

    # bump to next even
    timers_version[0] += 1
    if (timers_version[0] & 1) == 1:
        timers_version[0] += 1
