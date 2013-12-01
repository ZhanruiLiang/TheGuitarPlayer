from pyglet.gl import *
import utils

class FretPos:
    def __init__(self, stringIdx, fretIdx):
        self.stringIdx = stringIdx
        self.fretIdx = fretIdx

    def __repr__(self):
        return 'FretPos({}, {})'.format(self.stringIdx, self.fretIdx)

class FretBoard:
    PRESS_POS_RATE = 0.4
    def __init__(self, top, length, bottom, nFrets):
        self.top = top
        self.length = length
        self.bottom = bottom
        self.bars = make_bars(length, nFrets) # would have len = nFrets + 1

    def pos_fret_to_plane(self, fretPos):
        y1, y2 = self.get_string_ys(fretPos.stringIdx)
        k = self.PRESS_POS_RATE
        x = self.bars[fretPos.fretIdx - 1] * k + self.bars[fretPos.fretIdx] * (1 - k)
        y = y1 + (y2 - y1) / self.length * x
        return (x, y)

    def get_string_ys(self, stringIdx):
        y1 =  (7 / 12. - stringIdx / 6.) * self.top
        y2 =  (7 / 12. - stringIdx / 6.) * self.bottom
        return y1, y2

class FretBoardSprite(FretBoard):
    def __init__(self, localMat, top, length, bottom, nFrets):
        super().__init__(top, length, bottom, nFrets)
        assert localMat.shape == (4, 4)
        self.localMat = localMat
        self.marks = []

    def draw(self):
        glColor3f(0., 0., 0.)
        height = 0.002
        with utils.glPreserveMatrix():
            glMultMatrixf(utils.npmat_to_glmat(self.localMat))
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            with utils.glPrimitive(GL_POLYGON):
                glVertex3f(0, -self.top / 2, 0)
                glVertex3f(0, self.top / 2, 0)
                glVertex3f(self.length, self.bottom / 2, 0)
                glVertex3f(self.length, -self.bottom / 2, 0)
            with utils.glPrimitive(GL_LINES):
                y1 = self.top / 2
                y2 = self.bottom / 2
                l = self.length
                # draw bars
                for x in self.bars[1:]:
                    h = y1 + (y2 - y1) / l * x
                    glVertex3f(x, -h, 0)
                    glVertex3f(x, h, 0)
                # draw strings
                for i in range(1, 7):
                    y1, y2 = self.get_string_ys(i)
                    glVertex3f(0, y1, height)
                    glVertex3f(l, y2, height)
                # draw marks
                d = self.top / 12
                glColor3f(1., .2, .2)
                for fp in self.marks:
                    x, y = self.pos_fret_to_plane(fp)
                    glVertex3f(x - d, y + d, 0)
                    glVertex3f(x + d, y - d, 0)
                    glVertex3f(x + d, y + d, 0)
                    glVertex3f(x - d, y - d, 0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def set_marks(self, marks:"[FretPos]"):
        self.marks = marks

def make_bars(L, frets=20):
    a = 2 ** (1./12)
    xs = [0] * (frets + 1)
    for i in range(1, frets+1):
        xs[i] = (a - 1) / a * L + xs[i-1] / a
    return xs

def plot_fretboard(L, frets=20):
    xs = plot_fretboard(L, frets)
    # TODO

def test_make_bars():
    import numpy as np
    xs = make_bars(1, 20)
    assert np.allclose(xs[12], .5)

if __name__ == '__main__':
    test_make_bars()
