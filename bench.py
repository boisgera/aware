


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

    Cython without any modification to the source produces only a minor 
    improvement if any -- can hardly be measured. cdef'ing excitation_pattern
    did not produce anything measurable either. So far my attempts at getting
    more juice from Cython have been totally unsuccesful ...

    I should probably try to "whiten" completely excitation_pattern to begin
    with ... Yes, that produces some OUMPHA, with time down from 6.5 to
    3.55. And now, 50% of the time is spent in `mask_from_frame`. And the
    quantizer module is now taking 30% of the time ...

    I try to de-vectorize excitation_pattern to be able to unroll the loops
    in mask_from_framee, so far I am losing performance (around 8.5 sec now).
    
    Well, honnestly I don't get it now. I have inlined everything in a single
    function and that's worse than the numpy/array version by a factor of 2 ...
    While almost all the code is white ... OK, I am kinda stuck now, mask_from_frame
    is essentually "white", without Python overhead and I am worse than I was
    in the first place ... Issue about cache locality, that kind of things ?

    Commenting stuff shows that the double k/i loop is taking all the time ...
    while the i loops only take no measurable time ... OK, so half of the
    time there (in the k loop) is taken by the log to lin computation, the
    rest by the sum of product used to compute the contribution in the db
    scale.

    I can achieve SOME reduction by a reduction of the density, but this
    method has some limits: with a density of 8 instead of 16, I am down
    to 3.5 (instead of 5.8).


    >>> from aware import *
    >>> demo2()
    """
