#!/usr/bin/env python
# coding: utf-8
"""
Aware -- Perceptual Audio Coder
"""

# Python 2.7 Standard Library
from __future__ import division
import doctest
import pickle
import sys
import time

# Third-Party Libraries
import sh
import numpy as np
import pylab as pl
from pylab import *; seterr(all="ignore")

# Digital Audio Coding
from filters import MPEG, Analyzer, Synthesizer
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
# Constants
# ------------------------------------------------------------------------------
#

# sampling frequency / time
df = 44100.0
dt = 1.0 / df

# fft window size
N_FFT = 512

# filter length (FIR)
N = MPEG.N

# number of subbands
M = MPEG.M

# frame size for the subband quantizer
L = 12 * M

# number of bits available for every sequence of M subband samples
BIT_POOL = 112
assert 2 <= BIT_POOL <= M * 16

# scale factor used by the subband quantizer 
SCALE_FACTORS = logspace(1, -20, 64, base=2.0)[::-1] 

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
    return ravel(repeat(period, N //(2*n) + 1, axis=0))[:N]
   
#
# Helpers
# ------------------------------------------------------------------------------
#

pass

#
# Analysis and Synthesis Filter Banks
# ------------------------------------------------------------------------------
#

def display_subbands(data):
    analyze = Analyzer(MPEG.A, dt=MPEG.dt)
    # Add zeros at the head to implement strictly the polyphase filter
    # and add zeros at the tail to account for the filter-induced delay.
    data = r_[np.zeros(M-1), data, np.zeros(N)]
    frames = np.array(split(data, MPEG.M, zero_pad=True))
    subband_frames = transpose([analyze(frame) for frame in frames])
    assert shape(subband_frames)[0] == M
    for i, data in enumerate(subband_frames):
        pl.plot(data + i*1.0, "k-")
    pl.title("Subband Decomposition")
    pl.ylabel("subband number")
    pl.xlabel("subband data")
    pl.axis("tight")


#
# Perceptual Model
# ------------------------------------------------------------------------------
#

def raw_maskers(frame, window=hanning):
    frame = array(frame, copy=False)
    if shape(frame) != (N,):
        error = "the frame should be 1-dim. with {0} samples."
        raise TypeError(error.format(N))

    # Compute a gain alpha that compensates the energy loss caused by the 
    # windowing -- a frame with constant values is used as a reference.
    alpha = 1.0 / sqrt(sum(window(N)**2) / N)
    x = alpha * window(N) * frame

    k = arange(N // 2 + 1)
    frame_fft_2 = abs(fft(frame)) ** 2

    P = 2.0 * frame_fft_2[:(N // 2 + 1)] / N ** 2
    P[0] = 0.5 * P[0]
    if (N % 2 == 0):
        P[-1] = 0.5 * P[-1]

    # +96 dB normalization
    P = 10.0 ** (96.0 / 10.0) * P
    
    return k, P

class Classifier(object):
    "Tone/Noise Classifier"
    def __init__(self):
        small  = np.array([-2, +2])
        medium = np.array([-3, -2, +2, +3]) 
        large  = np.array([-6, -5, -4, -3, -2, +2, +3, +4, +5, +6])
        self.neighbourhood = 256 * [None]
        for _k in range(2, 63):
            self.neighbourhood[_k] = small
        for _k in range(63, 127):
            self.neighbourhood[_k] = medium
        for _k in range(127, 251):
            self.neighbourhood[_k] = large        
    def __call__(self, k, P):
        assert all(k == np.arange(0, N // 2 + 1))
        k_t = []
        P_t = []
        for _k in arange(3, 251):
            if (P[_k-1] <= P[_k] and P[_k+1] <= P[_k]): # local maximum
                js = self.neighbourhood[_k]
                if all(P[_k] >= 5.0 * P[_k+js]): # +7.0 dB
                    k_t.append(_k)
                    P_t.append(P[_k-1] + P[_k] + P[_k+1])
                    P[_k-1] = P[_k] = P[_k+1] = 0.0
        return (array(k_t), array(P_t)), (k, P)        

classify = Classifier()

def group_by_critical_band(k, P):
    # cb_k: critical band number indexed by frequency line index k.
    f_k = arange(N // 2 + 1) * df / N
    b_k = bark(f_k)
    cb_k = array([int(b) for b in floor(b_k)])
    bands = [[] for _ in arange(amax(cb_k) + 1)]
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
    for band, k_P_s in enumerate(bands):
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
    k_nt_out = zeros(len(bands), dtype=uint8)
    P_nt_out = zeros(len(bands))
    for band, k_P_s in enumerate(bands):
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
            k_nt_out[band] = k_mean
            P_nt_out[band] = P_sum
    return k_nt_out, P_nt_out

# "T" for threshold ? "A" for absolute ?
def threshold(k, P):
    f_k = arange(N // 2 + 1) * df / N
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

    db_1 = np.minimum(db + 1.0, 0.0)
    db_2 = np.minimum(db      , 0.0)
    db_3 = np.maximum(db      , 0.0)
    db_4 = np.maximum(db - 1.0, 0.0)    

    mask  = I \
          + (11.0 - 0.40 * I) * db_1 \
          + ( 6.0 + 0.40 * I) * db_2 \
          - (17.0           ) * db_3 \
          + (       0.15 * I) * db_4

#    mask  = I \
#          - (11.0 - 0.40 * I) * (-db - 1.0) * (db <= -1.0) \
#          - ( 6.0 + 0.40 * I) * (-db      ) * (db <   0.0) \
#          - (17.0           ) * ( db      ) * (db >=  0.0) \
#          + (       0.15 * I) * ( db - 1.0) * (db >=  1.0)
    if tonal:
        mask += -1.525 - 0.275 * b - 4.5
    else:
        mask += -1.525 - 0.175 * b - 0.5
    return mask

# TODO: Avoid globals here, create a closure or a class, stop polluting
#       the global namespace.
# k is the frequency line index (257 values), i a subsampling (112 values).
k = arange(N // 2 + 1)
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

    #f_k = arange(N // 2 + 1) * df / N
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
#    f_k = arange(N // 2 + 1) * df / N
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

#@profile
def allocate_bits(frames, mask, bit_pool=None):
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
    assert shape(frames) == (12, 32)
    assert shape(mask) == (32,)
 
    bit_pool = bit_pool or BIT_POOL
    assert 2 <= bit_pool <= M * 16

    sf_quantizer = ScaleFactor(SCALE_FACTORS)
    sf_subband = zeros(32)
    for subband, frame in enumerate(transpose(frames)):
        sf_index = sf_quantizer.index(frame)
        sf_subband[subband] = SCALE_FACTORS[sf_index]

    bits = zeros(32, dtype=uint8)
    delta = 2.0 * sf_subband
    noise_level = 96.0 + 10 * log10((delta ** 2) / 12.0)
    noise_to_mask = noise_level - mask
    delta_dB = 10.0 * log10(2.0)
    while bit_pool >= 2:
        subband = np.argmax(noise_to_mask)
        # avoid subbands with a single bit allocated.
        num_bits = 1 + (noise_to_mask[subband] == 0)
        bits[subband] += num_bits                   
        if bits[subband] < 16:
            noise_to_mask[subband] -= num_bits * delta_dB
        else: # maximal number of bits reached for this subband
            noise_to_mask[subband] = - np.inf
        bit_pool = bit_pool - num_bits
    if bit_pool == 0: 
        penalty = np.inf * (bits == 0)
        subband = np.argmax(noise_to_mask - penalty)
        if 0 < bits[subband] < 16: # call me paranoid.
            bits[subband] += 1

    return bits


class SubbandQuantizer(Quantizer):
    def __init__(self, mask, bit_pool=None):
        self.mask = mask
        self.bit_pool = bit_pool or BIT_POOL
        self.bits = []

    def encode(self, frames):
        frames = np.array(frames)
        assert np.shape(frames) == (12, 32)
        bits = allocate_bits(frames, self.mask, bit_pool=self.bit_pool)
        self.bits.append(bits)
        quantizers = []
        for i, bit in enumerate(bits):
            N = 2**bit - 1
            quantizer = ScaleFactor(SCALE_FACTORS, Uniform(-1.0, 1.0, N))
            quantizers.append(quantizer)
        output = []
        for subband, frame in enumerate(transpose(frames)):
            index, codes = quantizers[subband].encode(frame)
            output.append([bits[subband], index, codes])
        return output

    def decode(self, data):
        frames = []
        for subband in range(32):
            bit, index, codes = data[subband]
            N = 2**bit - 1
            uniform_quantizer = Uniform(-1.0, 1.0, N)
            quantizer = ScaleFactor(SCALE_FACTORS, uniform_quantizer)
            frames.append(quantizer.decode((index, codes)))
        return array(transpose(frames))

    #def mean_bit_alloc(self):
    #    return np.round(mean(self._bits, axis=0), decimals=0).astype(int32)

#
# Aware Compression
# ------------------------------------------------------------------------------
#

# ... DEPRECATED ...............................................................
#def __demo(data=None, bit_pool=None, play=False, display=False):
#    if data is None:
#        data = square(1760.0, 512*100)

#    assert len(data) >= 1024    
#    
#    bit_pool = bit_pool or BIT_POOL

#    # Compute the single mask used for every bit allocation.
#    reference_frame = data[:512]
#    length = len(data)
#    mask = mask_from_frame(reference_frame)

#    if display:
#        figure()
#        display_mask(reference_frame)
#        figure()
#        display_subbands(data)

#    # Apply the analysis filter bank.
#    analyze = Analyzer(MPEG.A, dt=MPEG.dt)
#    data = r_[data, zeros(512)] # take into account the delay:
#    # without this extra frame, we may not have enough output values.
#    frames = array(split(data, MPEG.M, zero_pad=True))
#    subband_frames = array([analyze(frame) for frame in frames])

#    # Make sure we have an entire numbers of 12-sample frames.
#    remainder = shape(subband_frames)[0] % 12
#    if remainder:
#        subband_frames = r_[subband_frames, zeros((12-remainder, 32))]

#    # Quantize the data in each subband.
#    quant_subband_frames = []
#    subband_quantizer = SubbandQuantizer(mask, bit_pool=bit_pool)
#    for i in range(shape(subband_frames)[0] / 12):
#        subband_frame = subband_frames[i*12:(i+1)*12]
#        quant_subband_frames.append(subband_quantizer(subband_frame))

#    mean_bits = subband_quantizer.mean_bit_alloc()
#    if display:
#        figure()
#        bar(arange(32.0)-0.4, mean_bits)
#        xlabel("subband number")
#        ylabel("number of bits (mean)")
#        title("Bit Allocation Profile")
#        grid(True)
#        axis([-1, 32, 0, max(mean_bits) + 1])

#    # Reconstruct the approximation of the original audio data 
#    synthesize = Synthesizer(MPEG.S, dt=MPEG.dt, gain=MPEG.M)
#    output = []
#    for frame_12 in quant_subband_frames:
#        for frame in frame_12:
#            output.extend(synthesize(frame))

#    # Synchronize input and output data.
#    output = output[_delay:length+_delay]
#    
#    if display:
#        figure()
#        plot(arange(512, 1024), data[512:1024], "k-o", ms=3.0, label="original data")
#        plot(arange(512, 1024), output[512:1024], "r-o", ms=3.0, alpha=0.7, label="compressed data")
#        xlabel("sample number")
#        ylabel("sample value")
#        grid(True)
#        axis("tight")
#        legend()
#        title("Waveforms before/after compression (sample)")
#        
#    if play:
#        sh.rm("-rf", "tmp"); sh.mkdir("tmp")
#        wave.write(data, "tmp/sound.wav")
#        wave.write(output, "tmp/sound-aware.wav")
#        print "playing the original sound ..."
#        sys.stdout.flush()
#        time.sleep(1.0)
#        sh.play("tmp/sound.wav")
#        print "playing the encoded sound ..."
#        sys.stdout.flush()
#        time.sleep(1.0)
#        sh.play("tmp/sound-aware.wav")

#    return output
# ..............................................................................


# TODO: instead of the log approach, develop a "locals capture" aka snapshot ?
#       such that ALL relevant variables are captured but there is no change
#       is the pattern that is returned ? Capture only the variables that 
#       are listed as keys in snapshot ?
def demo(data, snapshot=None):
    data = np.array(data)

    # Synchronisation
    # --------------------------------------------------------------------------
    #
    # The implementation should be careful to synchronize the input signal,
    # the mask computations, the subband quantization and the output signal.
    # Here is how we proceed:
    # 
    #  1. First off, we add M - 1 zero samples at the head of the data,
    #     to compensate for the advance induced by our implementation of
    #     the polyphase analysis filter.
    #
    #  2. Given that compensation the subband data is delayed with respect to 
    #     the input signal by N // 2 samples as a consequence of the causal 
    #     implementation of the analysis filters. We process this early data
    #     in the subbands anyway for a correct synthesis filter warm-up.
    #
    #  3. We need to delay the first mask computation by N // 2 samples.
    #     We do that by adding N // 2 zeros at the start of the data used
    #     by the mask computations. To take into account the overlap of 
    #     N_FFT - L samples between successive analysis windows, we add
    #     (N_FFT - L) // 2 extra zero samples.
    #  
    #  4. The total delay induced by the analysis and synthesis being N samples,
    #     we add an extra frame at the end of the signal to be able to produce
    #     the last values. We may have to add a little more samples at the end,
    #     just to make sure than the subband data may be grouped in an entire
    #     number of frames of length L.
    #
    #  5. During the decompression, all we have to do is to drop the first
    #     frame of N samples and to used the data length of the binary format 
    #     to stop at the right point.

    # TODO: add some sync. info, in order to be able, given a sample number
    #       of the *original* signal, to get the appropriate mask, bit alloc.
    #       pattern, etc. Use that information for example to display the
    #       chunk of L samples that are handled together ?

    # Compute the masks
    overlap = N_FFT - L
    head = np.zeros(N // 2 + overlap // 2)
    tail = np.zeros(N + L + overlap // 2) # covers the worst-case
    data_mask_sync = np.r_[head, data, tail]
    mask_frames = split(data_mask_sync, N_FFT, zero_pad=True, overlap=(N - L)) 
    masks = [mask_from_frame(frame) for frame in mask_frames]

    # Apply the analysis filter bank.
    head = np.zeros(M-1)
    tail = np.zeros(N)
    data_filter_sync = np.r_[head, data, tail]
    # enforce a data length that is a multiple of L
    extra_tail = np.zeros((L - len(data_filter_sync) % L) % L)
    data_filter_sync = np.r_[data_filter_sync, extra_tail]
    analyze = Analyzer(MPEG.A, dt=MPEG.dt)
    frames = np.array(split(data_filter_sync, MPEG.M, zero_pad=True))
    subband_frames = array([analyze(frame) for frame in frames])

    # Quantize the subband data
    quant_subband_frames = []
    bits = []
    assert shape(subband_frames)[0] % (L // M) == 0
    num_blocks = shape(subband_frames)[0] // (L // M)
    for i in range(num_blocks):
        subband_quantizer = SubbandQuantizer(masks[i])
        subband_frame = subband_frames[i*(L // M) : (i+1)*(L // M)]
        quant_subband_frames.append(subband_quantizer(subband_frame))
        bits.extend(subband_quantizer.bits)
    bits = np.array(bits)

    # Reconstruct the (approximation of) the original audio data 
    synthesize = Synthesizer(MPEG.S, dt=MPEG.dt, gain=MPEG.M)
    output = []
    for frames in quant_subband_frames:
        for frame in frames:
            output.extend(synthesize(frame))
    output = output[N:N+len(data)]

    if snapshot is not None:
        snapshot.update(locals())

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

# TODO: drop the first frames in the quantizer to deliver some data in
#       phase with the input ? That may prove to be difficult, otherwise
#       store some offset info in the binary file for the offset deletion
#       at the reconstruction. Yes, we need the same trick for the end
#       anyway.


if __name__ == "__main__":
    demo()

if __name__ == "__main____":
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

