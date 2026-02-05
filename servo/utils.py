import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

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
    vmin, vmax = np.percentile(rois[:int(len(rois)*config.NORMALIZATION_LEN_RATIO),::], [4,96], axis=0)
    return vmin, vmax

def compute_profiles(a, x, y, w, l, sanity_check=True, get_roi=True):
    assert not(w%2), f'w should be even {w}'
    assert not(l%2), f'l should be even {l}'
    hw = w//2
    hl = l//2
    hbar = a[max(0, x - hl): min(x + hl, a.shape[0]), 
             max(0, y - hw): min(y + hw, a.shape[1])]
    hprofile = np.mean(hbar, axis=1)
    vbar = a[max(0, x - hw): min(x + hw, a.shape[0]),
             max(0, y - hl): min(y + hl, a.shape[1])]
    vprofile = np.mean(vbar, axis=0)
    if sanity_check:
        assert hbar.shape == (l, w), 'profile is taken too near the border given w and l'
        assert vbar.shape == (w, l), 'profile is taken too near the border given w and l'

    if get_roi:
        roi = a[max(0, x - hl): min(x + hl, a.shape[0]),
                max(0, y - hl): min(y + hl, a.shape[1])]
        
        return hprofile, vprofile, roi
    else: return hprofile, vprofile

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

    return left, center, right

def get_mean_pixels_positions(pixels_lists):
    pos = list()
    for ilist in pixels_lists:
        pos.append((np.max(ilist) + np.min(ilist)) / 2)
    return pos
        
        
        
def compute_profile_levels(profile, pixels_lists, mean=True):
    left_level = profile[pixels_lists[0]]
    center_level = profile[pixels_lists[1]]
    right_level = profile[pixels_lists[2]]

    if mean:
        return np.mean(left_level), np.mean(center_level), np.mean(right_level)
    else:
        return np.median(left_level), np.median(center_level), np.median(right_level)

def normalize_profile(profile, vmin, vmax):
    prof_norm = np.clip((profile - vmin) / (vmax - vmin), 0,1)
    return prof_norm

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
    low_center_level, high_center_level = np.percentile(levels[:,1], [1, 99])
    left_norm = (np.mean(levels[np.nonzero(levels[:,1] < low_center_level),0]),
                 np.mean(levels[np.nonzero(levels[:,1] > high_center_level),0]))

    right_norm = (np.mean(levels[np.nonzero(levels[:,1] < low_center_level),2]),
                  np.mean(levels[np.nonzero(levels[:,1] > high_center_level),2]))

    return np.array(list(left_norm) + list(right_norm))

def compute_angles(levels, ellipse_norm_coeffs):
    levels = np.copy(levels)
    levels[0] = (levels[0] - levels[1] * (ellipse_norm_coeffs[1] - ellipse_norm_coeffs[0])
                 - ellipse_norm_coeffs[0])
    levels[2] = (levels[2] - levels[1] * (ellipse_norm_coeffs[3] - ellipse_norm_coeffs[2])
                 - ellipse_norm_coeffs[2])
    levels[1] = levels[1] - 0.5
    left_angle = np.arctan2(levels[0], levels[1])
    right_angle = np.arctan2(levels[2], levels[1])
    return np.array((left_angle, right_angle))
    
def unwrap_angles(angles, last_angles):
    unwrapped = np.unwrap(np.array((last_angles, angles)), axis=0, period=2*np.pi)[1]
    return unwrapped

def compute_opds(unwrapped_angles):
    return -config.CALIBRATION_LASER_WAVELENGTH * unwrapped_angles / (2*np.pi)
