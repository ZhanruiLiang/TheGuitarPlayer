import contextlib
from pyglet.gl import *

@contextlib.contextmanager
def glPrimitive(mode):
    glBegin(mode)
    yield
    glEnd()

@contextlib.contextmanager
def glPreserveMatrix():
    glPushMatrix()
    yield
    glPopMatrix()
