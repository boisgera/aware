#!/usr/bin/env python
# coding: utf-8
"""
Aware -- Perceptual Audio Coder
"""

# Python 2.7 Standard Library
import doctest
import pickle
import sys
import time

# Third-Party Libraries
import sh
from pylab import *; seterr(all="ignore")

# Digital Audio Coding
from filters import MPEG
from frames import split
import psychoacoustics
from psychoacoustics import ATH, bark, hertz, Mask
from quantizers import Quantizer, ScaleFactor, Uniform
import wave

#
# Metadata
# ------------------------------------------------------------------------------
#
__author__ = u"Sébastien Boisgérault <Sebastien.Boisgerault@mines-paristech.fr>"
__license__ = "MIT License"
__version__ = None

#
# TODO
# ------------------------------------------------------------------------------
#
# Make a `demo` that can manage signals with time-varying masks.
#
#   - the psychoacoustics "Mask" abstraction is costly and should probably not 
#     be used here. Still, if I take into account only the mask computation, 
#     the perceptual coder could run at approximately 1/2 - 1/3 real-time.
#


#
# Constants
# ------------------------------------------------------------------------------
#

N = 512
df = 44100.0
dt = 1.0 / df

#
# Signal Generators
# -------------------------------------------------------------------------------
#

def tone(f=440.0, N=512, phi=0.0):
    t = arange(N) * dt
    return cos(2*pi*f*t + phi)

def white_noise(sigma=1.0, N=512):
    return normal(0.0, sigma, N)

def square(f=440.0, N=512):
    n = int(round_(0.5 * (df / f)))
    period = reshape(r_[ones(n), -1.0 * ones(n)], (1, 2*n))
    return ravel(repeat(period, N /(2*n) + 1, axis=0))[:N]
   
#
# Analysis and Synthesis Filter Banks
# ------------------------------------------------------------------------------
#

class Analyzer(object):
    """
    Analysis Filter Bank

    Compute the output of an array of causal FIR filters, critically sampled.

    Attributes
    ----------
    
    - `M`: number of subbands,

    - `N`: common filter length.

    Example
    -------

    We define an analysis filter bank with filters numbered from 0 to 3. 
    The i-th filter is a simple delay of i + 4 samples. 
    The common filter length is set to 8 (the minimal requirement).
    
        >>> Z = zeros((4, 4), dtype=float)
        >>> I = eye(4, dtype=float)
        >>> a = c_[Z, I]

        >>> analyzer = Analyzer(a)
        >>> analyzer.M, analyzer.N
        (4, 8)
        >>> analyzer([1, 2, 3, 4])
        array([ 0.,  0.,  0.,  0.])
        >>> analyzer([5, 6, 7, 8])
        array([ 4.,  3.,  2.,  1.])
        >>> analyzer([0, 0, 0, 0])
        array([ 8.,  7.,  6.,  5.])
        >>> analyzer([0, 0, 0, 0])
        array([ 0.,  0.,  0.,  0.])
    """
    def __init__(self, a, dt=1.0, gain=1.0):
        """
        Arguments:
        ----------

          - `a`: filter bank impulse responses -- a two-dimensional numpy array 
            whose row `a[i,:]` is the impulse response of the `i`-th bank 
            filter.

          - `dt`: sampling time, defaults to `1.0`,

          - `gain`: a factor applied to the output values, defaults to `1.0`.
        """
        self.M, self.N = shape(a)
        self.A = gain * a * dt
        self.buffer = zeros(self.N)

    def __call__(self, frame):
        """
        Argument
        --------

        - `frame`: a sequence of `self.M` new input value of the filter bank, 
        
        Returns
        -------

        - `subbands`: the corresponding `self.M` new output subband values.
        """
        frame = array(frame, copy=False)
        if shape(frame) != (self.M,):
            raise ValueError("shape(frame) is not ({0},)".format(self.M))
        self.buffer[self.M:] = self.buffer[:-self.M]
        self.buffer[:self.M] = frame[::-1]
        return dot(self.A, self.buffer)

class Synthesizer(object):
    """
    Synthesis Filter Bank

    Combine critically sampled subband signals with an array of causal 
    FIR filters.

    Attributes
    ----------
    
    - `M`: number of subbands,

    - `N`: common filter length.

    Example
    -------

    We define a synthesis filter bank with filters numbered from 0 to 3. 
    The i-th filter is a simple delay of 7 - i samples. This synthesis
    filter bank provides a perfect reconstruction for the analysis filter
    bank implemented in the section "example" of the `Analyzer` 
    documentation, with a combined delay of 2 frames (8 samples).


        >>> Z = zeros((4, 4), dtype=float)
        >>> J = eye(4, dtype=float)[:,::-1]
        >>> a = c_[Z, J]

        >>> synthesizer = Synthesizer(a)
        >>> synthesizer.M, synthesizer.N
        (4, 8)
        >>> synthesizer([0, 0, 0, 0])
        array([ 0.,  0.,  0.,  0.])
        >>> synthesizer([4, 3, 2, 1])
        array([ 0.,  0.,  0.,  0.])
        >>> synthesizer([8, 7, 6, 5])
        array([ 1.,  2.,  3.,  4.])
        >>> synthesizer([0, 0, 0, 0])
        array([ 5.,  6.,  7.,  8.])
        >>> synthesizer([0, 0, 0, 0])
        array([ 0.,  0.,  0.,  0.])
    """
    def __init__(self, s, dt=1.0, gain=1.0):
        """
        Arguments:
        ----------

          - `s`: filter bank impulse responses -- a two-dimensional numpy array 
            whose row `s[i,:]` is the impulse response of the `i`-th bank 
            filter.

          - `dt`: sampling time, defaults to 1.0,

          - `gain`: a factor applied to the output values, defaults to `1.0`.
        """
        self.M, self.N = shape(s)
        self.P = transpose(gain * dt * s)[::-1,:]
        self.buffer = zeros(self.N)

    def __call__(self, frame):
        """
        Argument
        --------

        - `subbands`: a sequence of `self.M` new subband values, 
        
        Returns
        -------

        - `frame`: the corresponding `self.M` new output values.

        """
        frame = array(frame, copy=False)
        if shape(frame) != (self.M,):
            raise ValueError("shape(frame) is not ({0},)".format(self.M))
        self.buffer += dot(self.P, frame)
        output = self.buffer[-self.M:][::-1].copy()
        self.buffer[self.M:] = self.buffer[:-self.M]
        self.buffer[:self.M] = zeros(self.M)
        return output

_delay = 481

def reconstruct(data, shift_delay=True):
    """
    MPEG Pseudo-Quadrature Mirror Filters
    """
    analyze = Analyzer(MPEG.A, dt=MPEG.dt)
    synthesize = Synthesizer(MPEG.S, dt=MPEG.dt, gain=MPEG.M)
    length = len(data)
    data = r_[data, zeros(512)] # take into account the delay:
    # without this extra frame, we may not have enough output values.
    frames = array(split(data, MPEG.M, zero_pad=True))

    output = []
    for frame in frames:
        output.extend(synthesize(analyze(frame)))
    output = array(output)

    if shift_delay:
        output = output[_delay:_delay+length]
    return output

def display_subbands(data):
    analyze = Analyzer(MPEG.A, dt=MPEG.dt)
    data = r_[data, zeros(512)] # take into account the delay:
    frames = array(split(data, MPEG.M, zero_pad=True))
    subband_frames = transpose([analyze(frame) for frame in frames])
    assert shape(subband_frames)[0] == 32
    for i, data in enumerate(subband_frames):
        plot(data + i*1.0, "k-")
    title("Subband Decomposition")
    ylabel("subband number")
    xlabel("subband data")
    axis("tight")

#
# Psychoacoustics Masks
# ------------------------------------------------------------------------------
#

# Optimization TODO:
#
#   - rewrite a mask_from_frame reunited with sample_mask with arguments:
#       - frame
#       - floor
#       - f_k ? or "df_k" ? or number of freq. sample ?
#       - sampler that acts on the mask(f_k) ? (replaces density and algo).
#
#   - need also to rewrite excitation pattern to avoid the nested function.
#     (or optional ? if not given return the function ?)
#
def Ik(x, window=ones, dB=True):
    """
    Compute an array of sound pressure levels / sound intensities.
    
    The `k`-th component of `Ik(x)` corresponds to the contribution of the
    frequency `fk = k * df / len(x)` where `df` is the sampling frequency. 
    The index `k` ranges from `0` to the largest value such that 
    `fk <= 0.5 * df` (the Nyquist frequency).
    
    
    Arguments
    ---------

      - `x`: a sequence of numbers, the signal to be analyzed.

      - `window`: an optional window function (no window by default). 

        The window is automatically multiplied by a gain that attempts to
        compensates the energy loss caused by the signal windowing (this
        gain depends only on the window, not on the signal values).

      - `dB`: compute the intensities as SPL in dB (defaults to `True`).


    Returns
    -------

      - `Ik`: an array of length `len(x) / 2 + 1`.


    Examples
    --------

        >>> x = ones(32, dtype=float)
        >>> P = mean(x**2)
        >>> P == sum(Ik(x, dB=False))
        True

        >>> x = ones(32, dtype=float)
        >>> Ik_ = Ik(x)
        >>> len(Ik_) == len(x) / 2 + 1
        True
        >>> k = arange(len(Ik_))
        >>> df = 1.0
        >>> all([k_ * df / len(x) <= df / 2.0 for k_ in k])
        True
        >>> (k[-1] + 1) * df / len(x) <= df / 2.0
        False
    """
    x = array(x, copy=False)
    if len(shape(x)) != 1:
        raise TypeError("the frame should be 1-dimensional.")
    n = len(x)

    # Compute a gain alpha that compensates the energy loss caused by the 
    # windowing -- a frame with constant values is used as a reference.
    alpha = 1.0 / sqrt(sum(window(n)**2) / n)
    x = alpha * window(n) * x

    xk2 = abs(fft(x)) ** 2
    Ik = 2.0 * xk2[:N/2+1] / n ** 2
    Ik[0] = 0.5 * Ik[0]
    if (n % 2 == 0):
        Ik[-1] = 0.5 * Ik[-1]
    if dB:
        Ik = 10.0 * log10(Ik) + 96.0          
    return Ik

def display_Ik(x):
    """
    Display the power spectrum of the frame `x`
    """
    fk = arange(len(x)/2 + 1) * df / len(x)
    plot(fk, Ik(x), "k-", label="rect. window")
    plot(fk, Ik(x, window=hanning), "b", label="hanning window")
    grid(True)
    xlabel("frequency $f$ [Hz]")
    ylabel("intensity [dB]")
    title("Power Spectrum")
    axis("tight")
    legend()

def sort_maskers(Ik, group=False):
    """
    Sort maskers into tonal and non-tonal masker components.

    Argument
    --------

      - `Ik`: a sequence of 257 masker levels in dB,

      - `group`: whether close tonal components should be grouped
        (defaults to `False`)

    Returns
    -------

      - `tonal`, `non_tonal`: two sequences of 257 masker levels in dB.

        The sequence `tonal` represents the tonal masker levels, 
        `non_tonal` to non-tonal masker levels.

    Example 
    -------

    Consider a `maskers` array made of a floor at 0 dB and of a localized
    masker with a level of 96 dB:

        >>> maskers = zeros(257)
        >>> maskers[128] = 96.0

        >>> tonal, non_tonal = sort_maskers(maskers)

    The localized masker and its neighbours are selected and added in `tonal`:

        >>> list(tonal[125:132])
        [-inf, -inf, 0.0, 96.0, 0.0, -inf, -inf]

    The corresponding values are removed from `non_tonal`:

        >>> list(non_tonal[125:132])
        [0.0, 0.0, -inf, -inf, -inf, 0.0, 0.0]

    Same setting but with a masker localized on 3 spectral indices with the same 
    total 96 dB level and where we group the tonal neighbours:

        >>> level = 10.0 * log10((10.0 **(96.0 / 10.0)) / 3.0)
        >>> level # doctest: +ELLIPSIS
        91.2...
        >>> maskers = zeros(257)
        >>> maskers[127:130] = level
        >>> list(maskers[125:132]) # doctest: +ELLIPSIS
        [0.0, 0.0, 91.2..., 91.2..., 91.2..., 0.0, 0.0]

        >>> tonal, non_tonal = sort_maskers(maskers, group=True)
        >>> list(tonal[125:132]) # doctest: +ELLIPSIS
        [-inf, -inf, -inf, 96.0..., -inf, -inf, -inf]
        >>> list(non_tonal[125:132])
        [0.0, 0.0, -inf, -inf, -inf, 0.0, 0.0]

    """

    Ik = array(Ik, copy=False)
    if shape(Ik) != (257,):
        raise ValueError("invalid argument shape, it should be (257,)")
    t_maskers  = - inf * ones(len(Ik))
    nt_maskers = Ik.copy()
    for k, _ in enumerate(Ik): 
        if k <= 2 or k > 250:
            continue
        elif 2 < k < 63:
            js = [-2, +2]
        elif 63 <= k < 127:
            js = [-3, -2, +2, +3]
        elif 127 <= k <= 250:
            js = [-6, -5, -4, -3, -2, +2, +3, +4, +5, +6]

        if Ik[k] >= Ik[k-1] and Ik[k] >= Ik[k+1] \
           and all([Ik[k] >= Ik[k+j] + 7.0 for j in js]):
           if group:
               t_maskers[k] = 10 * log10(10**(Ik[k-1]/10) + 10**(Ik[k]/10) + 10**(Ik[k+1]/10))
           else:   
               t_maskers[k-1:k+2] = Ik[k-1:k+2]
           nt_maskers[k-1] = nt_maskers[k] = nt_maskers[k+1] = -inf 

    assert all((t_maskers == -inf) | (nt_maskers == -inf))
    return t_maskers, nt_maskers
    

# ------------------------------------------------------------------------------
# New version, more simplifications, decomposed in stages.

def raw_maskers(frame, window=hanning):
    frame = array(frame, copy=False)
    if shape(frame) != (N,):
        error = "the frame should be 1-dim. with {0} samples."
        raise TypeError(error.format(N))

    # Compute a gain alpha that compensates the energy loss caused by the 
    # windowing -- a frame with constant values is used as a reference.
    alpha = 1.0 / sqrt(sum(window(N)**2) / N)
    x = alpha * window(N) * frame

    k = arange(N / 2 + 1)
    frame_fft_2 = abs(fft(frame)) ** 2

    P = 2.0 * frame_fft_2[:(N / 2 + 1)] / N ** 2
    P[0] = 0.5 * P[0]
    if (N % 2 == 0):
        P[-1] = 0.5 * P[-1]

    # +96 dB normalization
    P = 10.0 ** (96.0 / 10.0) * P
    
    return k, P


def classify(k, P):
    assert all(k == arange(N / 2 + 1))
    k_t, k_nt = [], []
    P_t, P_nt = [], []
    for _k in k: 
        if _k <= 2 or _k > 250:
            continue
        elif 2 < _k < 63:
            js = [-2, +2]
        elif 63 <= _k < 127:
            js = [-3, -2, +2, +3]
        elif 127 <= _k <= 250:
            js = [-6, -5, -4, -3, -2, +2, +3, +4, +5, +6]
        if (P[_k-1] <= P[_k] and P[_k+1] <= P[_k] and 
            all([P[_k] >= 5.0 * P[_k+j] for j in js])): # +7.0 dB
            k_t.append(_k)
            P_t.append(P[_k-1] + P[_k] + P[_k+1])
    for _k in k:
        if not (_k-1 in k_t or _k in k_t or _k+1 in k_t):
            k_nt.append(_k)
            P_nt.append(P[_k])
    return (array(k_t), array(P_t)), (array(k_nt), array(P_nt))

def group_by_critical_band(k, P):
    # cb_k: critical band number indexed by frequency line index k.
    f_k = arange(N / 2 + 1) * df / N
    b_k = bark(f_k)
    cb_k = array([int(b) for b in floor(b_k)])

    bands = dict([(cb, []) for cb in arange(amax(cb_k) + 1)])
    for _k, _P in zip(k, P):
        bands[cb_k[_k]].append((_k, _P))
    return bands

# rename "merge_tonals" (optimization "T" for tone)
# rk: that's not exactly what I've read about: this is not a cb by cb matter
#     but a merge if the distance between 2 k's is < 0.5 bark.
# 
# TODO: replace the entire k by (floating-point) f_k ? Would induce less
#       error in the mask computations at low-freq.
def merge_tonals(k_t, P_t):
    bands = group_by_critical_band(k_t, P_t)
    k_t_out, P_t_out = [], []
    for band, k_P_s in bands.items():
        if k_P_s:
            k_max = None
            P_max = - inf 
            for (_k, _P) in k_P_s:
               if _P > P_max:
                   k_max = _k
                   P_max = _P
            k_t_out.append(k_max)
            P_t_out.append(P_max)
    return array(k_t_out), array(P_t_out)


# (optimization "N" noise)
def merge_non_tonals(k_nt, P_nt):
    bands = group_by_critical_band(k_nt, P_nt)
    k_nt_out, P_nt_out = [], []
    for band, k_P_s in bands.items():
        if k_P_s:
            k_P_array = array(k_P_s)
            k = k_P_array[:,0]
            P = k_P_array[:,1]
            P_sum = sum(P)
            # k_mean: not sure that's the best thing to do.
            # geometric mean suggested by Rosi. I believe that an 
            # arithmetic mean in the bark scale is better yet.
            if all(P == 0.0):
                P = ones_like(P)
            k_mean = int(round(average(k, weights=P))) 
            #bark_mean = mean([bark(k[i] * df / N) for i in arange(len(k)) 
            #                  if P[i]>0])            
            #k_mean = int(round(hertz(bark_mean) * N / df))
            #print "k, k_mean:", k, k_mean
            k_nt_out.append(k_mean)
            P_nt_out.append(P_sum)
    return array(k_nt_out), array(P_nt_out)

# "T" for threshold ? "A" for absolute ?
def threshold(k, P):
    f_k = arange(N / 2 + 1) * df / N
    ATH_k = 10 ** (ATH(f_k) / 10.0)
    k_out, P_out = [], []
    for (_k, _P) in zip(k, P):
        if _P > ATH_k[_k]:
            k_out.append(_k)
            P_out.append(_P)
    return array(k_out), array(P_out)

def maskers(frame):
    k, P = raw_maskers(frame)
    (k_t, P_t), (k_nt, P_nt) = classify(k, P)
    k_t, P_t = merge_tonals(k_t, P_t)
    k_nt, P_nt = merge_non_tonals(k_nt, P_nt)
    k_t, P_t = threshold(k_t, P_t)
    k_nt, P_nt = threshold(k_nt, P_nt)
    return (k_t, P_t), (k_nt, P_nt)


#-------------------------------------------------------------------------------

def excitation_pattern(b, b_m, I, tonal):
    """
    Compute the excitation pattern of a single masker.

    The spread function and attenuation factors are from MPEG-1 Audio Model 1.

    Arguments
    --------
      - `b`: scalar or array of frequencies in barks,
      - `b_m`: masker frequency (in barks),
      - `I`: masker power (in dB),
      - `tonal`: `True` if the masker is tonal, `False` otherwise.

    Returns
    -------

      - `mask`: array of excitation values in dB.

    """
    db = b - b_m
    mask  = I \
          - (11.0 - 0.40 * I) * (-db - 1.0) * (db <= -1.0) \
          - ( 6.0 + 0.40 * I) * (-db      ) * (db <   0.0) \
          - (17.0           ) * ( db      ) * (db >=  0.0) \
          + (       0.15 * I) * ( db - 1.0) * (db >=  1.0)
    if tonal:
        mask += -1.525 - 0.275 * b - 4.5
    else:
        mask += -1.525 - 0.175 * b - 0.5
    return mask

# k is the frequency line index (257 values), i a subsampling (112 values).
k = arange(N / 2 + 1)
f_k = k * df / N
b_k = bark(f_k)

k_i = r_[0:49, 49:97:2, 97:251:4]
f_i = k_i * df / N
b_i = bark(f_i)
ATH_i = ATH(f_i)
subband_i = array([int(s) for s in floor(f_i / (0.5 * df / 32))])

def mask_from_frame(frame):
    """
    Compute the mask function for a frame.

    Arguments
    ---------

    - `frame`: sequence of 512 samples,

    Returns
    -------

    - `mask`: an array of 32 subband mask level values in dB.

    """

    # compute the mask floor (linear scale)    
    mask_i = 10.0 ** (ATH_i / 10.0)

    # add the tonals and non-tonals mask values.
    (k_t, P_t), (k_nt, P_nt) = maskers(frame)
    for masker_index in arange(len(k_t)):
        _b, _P = b_k[k_t[masker_index]], P_t[masker_index]
        mask_i += 10.0 ** (excitation_pattern(b_i, b_m=_b, I=10.0*log10(_P), tonal=True) / 10.0)
    for masker_index in arange(len(k_nt)):
        _b, _P = b_k[k_nt[masker_index]], P_nt[masker_index]
        mask_i += 10.0 ** (excitation_pattern(b_i, b_m=_b, I=10.0*log10(_P), tonal=False) / 10.0)

    # convert the resulting mask value to dB
    mask_i = 10.0 * log10(mask_i)

    # select the lowest mask value in each of the 32 subbands.
    subband_mask = [[] for _ in range(32)]
    for i, _mask_i in enumerate(mask_i):
        subband_mask[subband_i[i]].append(_mask_i)
    for i, _masks in enumerate(subband_mask):
        subband_mask[i] = amin(_masks)
    return array(subband_mask)

# TODO: control of frequency units (Hz/bark) and power unit (linear/dB)
# TODO: display individual excitation pattern fof each masker
# TODO: better display (fill-between + patch at low freq) of the ATH
def display_maskers(frame, bark=True, dB=True):
    # setup x and y-axis unit conversions
    if bark:
        convert_f = lambda f: psychoacoustics.bark(f)
    else:
        convert_f = lambda f: f

    if dB:
        convert_P = lambda P: 10.0 * log10(P)
    else:
        convert_P = lambda P: P

    # f array for high-resolution (1 Hz)
    n = 22050
    f = arange(n + 1) / float(n + 1) * 0.5 * df 
    b = psychoacoustics.bark(f)

    k, P = raw_maskers(frame)
    P = clip(P, 1e-100, 1e100) # convenience patch for plots
    plot(convert_f(k * df / N), convert_P(P), "k:")

    #f_k = arange(N / 2 + 1) * df / N
    #b_k = bark(f_k)

    def k2b(k):
        return bark(k * df / N)

    (k_t, P_t), (k_nt, P_nt) = classify(k, P)
    k_t_m, P_t_m = merge_tonals(k_t, P_t)
    k_nt_m, P_nt_m = merge_non_tonals(k_nt, P_nt)
    k_t_m_t, P_t_m_t = threshold(k_t_m, P_t_m)
    k_nt_m_t, P_nt_m_t = threshold(k_nt_m, P_nt_m)

#    fill_between(k2b(k), -100.0*ones_like(P), clip(10.0*log10(P), -1000, 1000), color="k", alpha=0.3, label="raw")
#    #plot(k2b(k_t), 10.0*log10(P_t), "r+", label="raw tonals")  
#    #plot(k2b(k_nt), 10.0*log10(P_nt), "k+", label="raw non-tonals")  
#     plot(k2b(k_t_m), 10.0*log10(P_t_m), "m+", label="merged tonals")  
#    #plot(k2b(k_nt_m), 10.0*log10(P_nt_m), "b+", label="merged non-tonals")

#    # TODO: display only these finals maskers ?
#    f_k = arange(N / 2 + 1) * df / N
#    plot(k2b(k), ATH(f_k), "g:", label="ATH") 
    plot(convert_f(k_t*df/N), convert_P(P_t), "k+")  
    plot(convert_f(k_nt*df/N), convert_P(P_nt), "k+")     
    plot(convert_f(k_t_m_t*df/N), convert_P(P_t_m_t), "mo", alpha=0.5, mew=0.0, label="tonals")  
    plot(convert_f(k_nt_m_t*df/N), convert_P(P_nt_m_t), "bo", alpha=0.5, mew=0.0, label="non-tonals") 

    P_tot = 0.0
    for _k, _P in zip(k_nt_m_t, P_nt_m_t):
        _b = psychoacoustics.bark(_k * df / N)
        ep = excitation_pattern(b, b_m=_b, I=10.0*log10(_P), tonal=False)
        P_tot += 10.0 ** (ep / 10.0)
        if not bark:
            ep = 10.0 ** (ep / 10.0)
        fill_between(convert_f(f), convert_P(1e-10*ones_like(f)), ep, color="b", alpha=0.2)

    for _k, _P in zip(k_t_m_t, P_t_m_t):
        _b = psychoacoustics.bark(_k * df / N)
        ep = excitation_pattern(b, b_m=_b, I=10.0*log10(_P), tonal=True)
        P_tot += 10.0 ** (ep / 10.0)
        if not bark:
            ep = 10.0 ** (ep / 10.0)
        fill_between(convert_f(f), convert_P(1e-10*ones_like(f)), ep, color="m", alpha=0.2)

    if bark:
        P_tot = 10 * log10(P_tot)
    plot(convert_f(f), P_tot, "k-")


# -------------------------------------------------------
    # compute the mask floor (linear scale)    
    mask_i = 10.0 ** (ATH_i / 10.0)

    # add the tonals and non-tonals mask values.
    (k_t, P_t), (k_nt, P_nt) = maskers(frame)
    for masker_index in arange(len(k_t)):
        _b, _P = b_k[k_t[masker_index]], P_t[masker_index]
        mask_i += 10.0 ** (excitation_pattern(b_i, b_m=_b, I=10.0*log10(_P), tonal=True) / 10.0)
    for masker_index in arange(len(k_nt)):
        _b, _P = b_k[k_nt[masker_index]], P_nt[masker_index]
        mask_i += 10.0 ** (excitation_pattern(b_i, b_m=_b, I=10.0*log10(_P), tonal=False) / 10.0)

    # convert the resulting mask value to dB
    if dB:
        mask_i = 10.0 * log10(mask_i)

    plot(convert_f(f_i), mask_i, "k|", ms=100.0)

# --------------------------------------------------------

    m = mask_from_frame(frame)
    b_subbands = psychoacoustics.bark((arange(32) + 0.5) * (0.5 * df / 32))
    #plot(b_subbands, m, "ro", label="subband mask")
    b_boundaries = ravel(split(psychoacoustics.bark(arange(33) * (0.5 * df / 32)), 2, overlap=1))
    values = ravel([[_m, _m] for _m in m])
    plot(b_boundaries, values, "r")
    #fill_between(b_boundaries, -100*ones_like(values), values, color="r", alpha=0.3)

    if bark:
        x_min, x_max = 0, psychoacoustics.bark(0.5 * df)
        xticks(arange(25 + 1))
    else:
        x_min, x_max = 0.0, 22050.0
        xticks(arange(0.0, 22050.0, 500.0))
    if dB:
        y_min, y_max = -10.0, 100.0
    else:
        y_min, y_max = 0.1, 10.0**10

    axis([x_min, x_max, y_min, y_max])


    grid(True)
    #legend(loc=3)

#
# Subband Data Vector (Scale Factors) Quantizers
# ------------------------------------------------------------------------------
#
# **Reference scale factors:**
# "Introduction to Digital Audio Coding and Standards", 
# by Marina Bosi and Richard E. Goldberg, p. 299.  
#

_scale_factors = logspace(1, -20, 64, base=2.0)[::-1] 

_bit_pool = 112 # corresponds roughly to 192 kb/s PER CHANNEL (aka twice the
                # classic high quality setting of MP3).

def allocate_bits(frames, mask, bit_pool=_bit_pool):
    """
    Arguments
    ---------

      - `frames:` an array of shape 12 x 32. The array `frames[:,i]` shall
        contain 12 consecutive samples generated for the subband `i` by 
        the MPEG analysis filter bank.
      
      - `mask:` a sequence of 32 mask intensity level in dB, one for each
        subband.
    
    Returns
    -------

      - `bits`: the number of bits allocated in each subband.    
    
    """
#    frames = array(frames)
#    assert shape(frames) == (12, 32)
#    mask = array(mask)
#    assert shape(mask) == (32,)
 
    sf_quantizer = ScaleFactor(_scale_factors)
    sf_subband = zeros(32)
    for subband, frame in enumerate(transpose(frames)):
        sf_index = sf_quantizer.index(frame)
        sf_subband[subband] = _scale_factors[sf_index]

    bits = zeros(32, dtype=uint8)
    # rk: bits[subband] should be limited to 15 bits (not implemented)
    delta = 2.0 * sf_subband / 2 ** bits
    noise_level = 96.0 + 10 * log10((delta ** 2) / 12.0)
    noise_to_mask = noise_level - mask
    delta_dB = 10.0 * log10(2.0)
    while bit_pool != 0:
        subband = argmax(noise_to_mask)
        noise_to_mask[subband] -= delta_dB
        bit_pool = bit_pool - 1
  
    #assert sum(bits) == bit_pool # check that all bits have been allocated
    return bits

# TODO: transfer to quantizers ? Too specific for that ? Transfer a part of it ?
class SubbandQuantizer(Quantizer):
    def __init__(self, mask=None, bit_pool=_bit_pool):
        self.mask = mask
        self.bit_pool = bit_pool
        self._bits = []
    def encode(self, frames):
        frames = array(frames)
        assert shape(frames) == (12, 32)
        bits = allocate_bits(frames, self.mask, bit_pool=self.bit_pool)
        self._bits.append(bits)
        quantizers = []
        for bit in bits:
            # 0-bit quantizer and 1-bit *midtread* quantizer are alike: 
            # they have a single admissible value: 0.0.
            if bit == 1:
                bit = 0
            N = max(1, 2**bit - 1)
            uniform_quantizer = Uniform(N=N)
            quantizers.append(ScaleFactor(_scale_factors, uniform_quantizer))
        output = []
        for subband, frame in enumerate(transpose(frames)):
            index, codes = quantizers[subband].encode(frame)
            output.append([bits[subband], index, codes])
        return output
    def decode(self, data):
        frames = []
        for subband in range(32):
            bit, index, codes = data[subband]
            if bit == 1:
                bit = 0
            N = max(1, 2**bit - 1)
            uniform_quantizer = Uniform(N=N)
            quantizer = ScaleFactor(_scale_factors, uniform_quantizer)
            frames.append(quantizer.decode((index, codes)))
        return array(transpose(frames))
    def mean_bit_alloc(self):
        return round_(mean(self._bits, axis=0), decimals=0).astype(int32)

#
# Aware Compression
# ------------------------------------------------------------------------------
#

def demo(data=None, bit_pool=_bit_pool, play=False, display=False):
    if data is None:
        data = square(1760.0, 512*100)

    assert len(data) >= 1024    
    
    # Compute the single mask used for every bit allocation.
    reference_frame = data[:512]
    length = len(data)
    mask = mask_from_frame(reference_frame)

    if display:
        figure()
        display_mask(reference_frame)
        figure()
        display_subbands(data)

    # Apply the analysis filter bank.
    analyze = Analyzer(MPEG.A, dt=MPEG.dt)
    data = r_[data, zeros(512)] # take into account the delay:
    # without this extra frame, we may not have enough output values.
    frames = array(split(data, MPEG.M, zero_pad=True))
    subband_frames = array([analyze(frame) for frame in frames])

    # Make sure we have an entire numbers of 12-sample frames.
    remainder = shape(subband_frames)[0] % 12
    if remainder:
        subband_frames = r_[subband_frames, zeros((12-remainder, 32))]

    # Quantize the data in each subband.
    quant_subband_frames = []
    subband_quantizer = SubbandQuantizer(mask, bit_pool=bit_pool)
    for i in range(shape(subband_frames)[0] / 12):
        subband_frame = subband_frames[i*12:(i+1)*12]
        quant_subband_frames.append(subband_quantizer(subband_frame))

    mean_bits = subband_quantizer.mean_bit_alloc()
    if display:
        figure()
        bar(arange(32.0)-0.4, mean_bits)
        xlabel("subband number")
        ylabel("number of bits (mean)")
        title("Bit Allocation Profile")
        grid(True)
        axis([-1, 32, 0, max(mean_bits) + 1])

    # Reconstruct the approximation of the original audio data 
    synthesize = Synthesizer(MPEG.S, dt=MPEG.dt, gain=MPEG.M)
    output = []
    for frame_12 in quant_subband_frames:
        for frame in frame_12:
            output.extend(synthesize(frame))

    # Synchronize input and output data.
    output = output[_delay:length+_delay]
    
    if display:
        figure()
        plot(arange(512, 1024), data[512:1024], "k-o", ms=3.0, label="original data")
        plot(arange(512, 1024), output[512:1024], "r-o", ms=3.0, alpha=0.7, label="compressed data")
        xlabel("sample number")
        ylabel("sample value")
        grid(True)
        axis("tight")
        legend()
        title("Waveforms before/after compression (sample)")
        
    if play:
        sh.rm("-rf", "tmp"); sh.mkdir("tmp")
        wave.write(data, "tmp/sound.wav")
        wave.write(output, "tmp/sound-aware.wav")
        print "playing the original sound ..."
        sys.stdout.flush()
        time.sleep(1.0)
        sh.play("tmp/sound.wav")
        print "playing the encoded sound ..."
        sys.stdout.flush()
        time.sleep(1.0)
        sh.play("tmp/sound-aware.wav")

    return output


def demo2(data=None, report=False):
    if data is None:
        data = square(1760.0, 44100)
    t = arange(len(data)) * dt
    length = len(t)
    assert length >= 1024

    extra = {} # export mask data and bit allocation profiles.

    # Make sure that to "push" all the relevant values from the anlysis and
    # synthesis registers by feeding extra zeros at the end of the signal.
    # As the total delay is 481 (-31 + 256 + 256), a frame of 512 is fine.
    data = r_[data, zeros(512)]
    
    # Compute the masks to use for bit allocation.
    # --------------------------------------------------------------------------
    # The masks are based on 512-sample FFT while the bit allocation applies
    # on 12 x 32 = 384 samples so there is an overlap between the data used
    # by the FFT of 128 samples, 64 before and 64 after the 'real' data.
    # This is used in conjunction with Hanning window, so this actually make
    # sense.
    # 
    # The delay computation to get the spectral analysis and the frame 
    # compression right has to be done carefully:
    # 
    #   - the "polyphase implementation hack" that avoids to compute an initial
    #     almost empty frame. It corresponds to an *advance* of M - 1 = 31 samples.
    #
    #   - the way the analysis filter is implemented (512-sample impulse with a
    #     a leading 0 added for parity) induces an extra 256 delay
    #
    # This sums up to 287 delay. To perform the spectral analysis, we could
    # create a buffer with 287 + 64 = 351 zeros in front of the real data,
    # then perform the first mask computation at the buffer start and shift
    # by 384 until the end of the signal is obtained.
    #
    
    # TODO: interleave the mask computation / bit allocation ? for frame-by
    #       frame computation ?
    mask_frames = split(r_[zeros(351), data], 512, zero_pad=True, overlap=128) 
    masks = [mask_from_frame(frame) for frame in mask_frames]
    extra["mask"] = masks

    # Apply the analysis filter bank.
    analyze = Analyzer(MPEG.A, dt=MPEG.dt)
    frames = array(split(data, MPEG.M, zero_pad=True))
    subband_frames = array([analyze(frame) for frame in frames])

    # Make sure we have an entire numbers of 12-sample frames.
    remainder = shape(subband_frames)[0] % 12
    if remainder:
        subband_frames = r_[subband_frames, zeros((12-remainder, 32))]

    # Quantize the data in each subband.
    quant_subband_frames = []
    mean_bits = []
    for i in range(shape(subband_frames)[0] / 12):
        subband_quantizer = SubbandQuantizer(masks[i])
        subband_frame = subband_frames[i*12:(i+1)*12]
        quant_subband_frames.append(subband_quantizer(subband_frame))
        mean_bits.append(subband_quantizer.mean_bit_alloc())
    extra["bits"] = mean_bits

    # Reconstruct the approximation of the original audio data 
    synthesize = Synthesizer(MPEG.S, dt=MPEG.dt, gain=MPEG.M)
    output = []
    for frame_12 in quant_subband_frames:
        for frame in frame_12:
            output.extend(synthesize(frame))

    # Synchronize input and output data.
    data = data[:length]
    output = output[_delay:length+_delay]

    if report:
        return output, extra
    else:
        return output

# 
# Unit Test Runner
# ------------------------------------------------------------------------------
#

def test():
    """
    Run the doctests of this module.
    """
    doctest.testmod()

#
# Command-Line Interface
# ------------------------------------------------------------------------------
#

if __name__ == "__main__":
    # TODO: get the extra analysis data and save it in another file ? Y.
    # AH FUCK, what to do with the dual channels ? Whatever, the support
    # for stereo should be migrated to demo2 first (BTW, rename that crap,
    # and get rid of all the display and play code in there). We export
    # the analysis data instead.
    filename = sys.argv[1]
    output_file = filename.split(".")[0] + "-aware.wav"
    data = wave.read(filename)
    # TODO: support stereo directly in demo2
    output = zeros_like(data)
    for i, channel in enumerate(data):
        output[i,:] = demo2(channel)
    wave.write(output, output_file)

