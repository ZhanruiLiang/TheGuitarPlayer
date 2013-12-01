import contextlib
from pyglet.gl import *
import numpy as np
import ctypes

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

M16 = ctypes.c_float * 16

def npmat_to_glmat(mat):
    return M16(*np.array(mat.transpose()).reshape(16))

def extract_pos(mat):
    return mat[0:4, 3].reshape(4)

def extract_pos3(mat):
    return (mat[0:3, 3] / mat[3, 3]).reshape(3)

