# coding: utf-8
# cython: profile = True
# cython: boundscheck = False
# cython: wraparound = False

# Digital Audio Coding
import aware

# Numpy
import numpy as np
cimport numpy as np

# Cython
from cpython cimport bool
from libc.math cimport log10


# Issues: can't put docstring into overriden method (from aware) after the def ...
# So far, either I have to duplicate docstrings manually, or we are losing them.

cpdef excitation_pattern(np.ndarray[np.float64_t, ndim=1] b, double b_m, double I, bool tonal):
    cdef np.ndarray[np.float64_t, ndim=1] mask
    cdef double db
    cdef float db_1, db_2, db_3, db_4
    cdef int i
    cdef int n = b.shape[0]

    mask = np.zeros(n)
    for i in range(n):
        db = b[i] - b_m
        db_1 = min(db + 1.0, 0.0)
        db_2 = min(db      , 0.0)
        db_3 = max(db + 1.0, 0.0)
        db_4 = max(db - 1.0, 0.0)
        mask[i]  = I                 \
          + (11.0 - 0.40 * I) * db_1 \
          + ( 6.0 + 0.40 * I) * db_2 \
          - (17.0           ) * db_3 \
          + (       0.15 * I) * db_4

    if tonal:
        for i in range(n):
            mask[i] += -1.525 - 0.275 * b[i] - 4.5
    else:
        for i in range(n):
            mask[i] += -1.525 - 0.175 * b[i] - 0.5
    return mask

cdef unsigned char argmax(np.ndarray[np.float64_t, ndim=1] array):
     cdef unsigned char j
     cdef double current
     cdef unsigned char n = array.shape[0]

     cdef double value = - np.inf
     cdef unsigned char i = 0
     for j in range(n):
         current = array[j]
         if current > value:
             value = current
             i = j
     return i
     
cdef double frame_scale_factor(
  np.ndarray[np.float64_t, ndim=1] scale_factors, 
  np.ndarray[np.float64_t, ndim=1] frame):

    cdef unsigned char i, n
    cdef double max_, current, sf

    n = frame.shape[0]
    max_ = 0.0
    for i in range(n):
        current = abs(frame[i])
        if current > max_:
            max_ = current
    n = scale_factors.shape[0]
    for i in range(n):
        sf = scale_factors[i]
        if sf >= max_:
            break
    return sf

cpdef np.ndarray[np.uint8_t, ndim=1] allocate_bits(
  np.ndarray[np.float64_t, ndim=2] frames, 
  np.ndarray[np.float64_t, ndim=1] mask, 
  unsigned int bit_pool=aware.BIT_POOL):
    cdef unsigned char M = aware.M
    cdef unsigned char num_bits
    cdef unsigned char subband
    cdef np.ndarray[np.uint8_t, ndim=1] bits
    cdef np.ndarray[np.float64_t, ndim=1] frame
    cdef np.ndarray[np.float64_t, ndim=1] sf_subband
    cdef np.ndarray[np.float64_t, ndim=1] noise_to_mask
    cdef double delta
    cdef double noise_level
    cdef double delta_dB
    cdef double inf = np.inf
    cdef unsigned char sf_index

    cdef np.ndarray[np.float64_t, ndim=1] scale_factors = aware.SCALE_FACTORS

    sf_subband = np.zeros(M)
    for subband in range(frames.shape[1]):
        frame = frames[:, subband]
        sf_subband[subband] = frame_scale_factor(scale_factors, frame)

    noise_to_mask = np.zeros(M)
    for i in range(M):
        delta = 2.0 * sf_subband[i]
        noise_level = 96.0 + 10.0 * log10((delta ** 2) / 12.0)
        noise_to_mask[i] = noise_level - mask[i]
    delta_dB = 10.0 * log10(2.0)

    bits = np.zeros(M, dtype=np.uint8)
    while bit_pool >= 2:
        subband = argmax(noise_to_mask)
        # avoid subbands with a single bit allocated.
        num_bits = 1 + (bits[subband] == 0)
        bits[subband] += num_bits                   
        if bits[subband] < 16:
            noise_to_mask[subband] -= num_bits * delta_dB
        else: # maximal number of bits reached for this subband
            noise_to_mask[subband] = - inf
        bit_pool = bit_pool - num_bits
    if bit_pool != 0: 
        for i in range(M):
            if bits[i] == 0:
                noise_to_mask[i] = - inf
        subband = argmax(noise_to_mask)
        if 0 < bits[subband] < 16: # call me paranoid.
            bits[subband] += 1

    return bits


cpdef classify(np.ndarray[np.int_t, ndim=1] k, np.ndarray[np.float64_t, ndim=1] P):
    cdef unsigned char j, _k
    cdef unsigned char * small  = [-2, 2]
    cdef unsigned char * medium = [-3, -2, +2, +3]
    cdef unsigned char * large  = [-6, -5, -4, -3, -2, +2, +3, +4, +5, +6]
    cdef unsigned char length
    cdef unsigned char * nh

    k_t = []
    P_t = []
    nh = small
    length = 2
    for _k in range(3, 251):
        if _k == 63:
            nh = medium
            length = 4
        elif _k == 127:
            nh = large
            length = 10

        if (P[_k-1] <= P[_k] and P[_k+1] <= P[_k]): # local maximum
            for j in range(length):
                if P[_k] < 5.0 * P[_k + nh[j]]: # +7.0 dB
                    break
            else:
                k_t.append(_k)
                P_t.append(P[_k-1] + P[_k] + P[_k+1])
                P[_k-1] = P[_k] = P[_k+1] = 0.0
    # N.B.: there is no cleanup of 0 power maskers in non-tonals.
    return (np.array(k_t), np.array(P_t)), (k, P)     


