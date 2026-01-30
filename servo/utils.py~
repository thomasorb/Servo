import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

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
        isin = profiles[int(len(profiles)*0.7):,i]
        coeffs.append(get_min_max(isin, plot=False))
    return np.array(coeffs)
