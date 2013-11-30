
def make_lens(L, frets=20):
    a = 2 ** (1./12)
    xs = [0] * (frets + 1)
    for i in xrange(1, frets+1):
        xs[i] = (a - 1) / a * L + xs[i-1] / a
    return xs

def plot_fretboard(L, frets=20):
    xs = plot_fretboard(L, frets)
    # TODO

def test_make_lens():
    import numpy as np
    xs = make_lens(1, 20)
    assert np.allclose(xs[12], .5)

if __name__ == '__main__':
    test_make_lens()
