


def demo2():
    """
    So far, we're doing 10-15 x slower than real-time.
    Essentially, we are spending 90% of this time in the Mask abstraction from
    psychoacoustics.

    OK, after vectorization of excitation_pattern stuff and merge of sampling
    of the subbands, we are down to 9.39 s for a 1.29 s signal. That's 7.3 
    slower than real-time (for mono data). The time is essentially spent in:
    excitation_pattern (35%), bark (29% !) and mask_from_frame (18%).

    There is probably not much we can do about bark, except calling it less
    often by caching the results. For a 1.16 s signal, there are 66000 calls
    to bark when ... 2 should be enough !

    OK, down to 6.5 / 1.29 with a precomputation of the barks (5x slower than
    real-time). Now excitation_pattern is taking 50% of the time (self) and
    mask_from_frame 25%.

    The bottleneck now is excitation_pattern, called too many times. At this
    stage, I feel that there are two options:

      - vectorize excitation_patterns to avoid the loop in mask_from_frame:
        that way we may reduce the number of calls by 257, which may be
        enough.

      - go Cython in excitation_pattern to reduce the cost of a function call
        and do the same in the loop of mask_from_frame and maybe in the demo2
        calling loop. That's *probably* more efficient ...

    >>> from aware import *
    >>> demo2()
    """
