
# TODO: use direcly compress/decompress to avoid the dependency in sound files.

def compress():
    """
    Now with the introduction of the binary stream, we are 3 times slower than real-time for
    compression (15 sec for a 5 sec-long signal). It's not that bad given that we're working
    on a stereo signal here. OK, the writing part is taking something like 1/3 of the time for 
    the compression, this is good enough, probably not worth optimizing over.

    No, that's even better, i was duplicating the quantization code, now we're doing 2x slower
    than real-time for CD data (or ~ 1x for single channel data).

    >>> from aware import main
    >>> main(["sample.wav"])
    """

def decompress():
    """
    The decompression is (slightly) faster than real-time: 4.24 sec for a 5 sec sample.

    >>> from aware import main
    >>> main(["sample.awr"])
    """
