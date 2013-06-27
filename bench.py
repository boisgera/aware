


def demo2():
    """
    So far, we're doing 10-15 x slower than real-time.
    Essentially, we are spending 90% of this time in the Mask abstraction from
    psychoacoustics.

    OK, after vectorization of excitation_pattern stuff and merge of sampling
    of the subbands, we are down to 9.39 s for a 1.29 s signal. That's 7.3 
    slower than real-time (for mono data). The time is essentially spent in:
    excitation_pattern (35%), bark (29% !) and mask_from_frame (18%).

    >>> from aware import *
    >>> demo2()
    """
