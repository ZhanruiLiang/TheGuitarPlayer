#! /usr/bin/env python3
import numpy as np
from shapes import WiredCylinder, AxisSystem
import utils
from pyglet.gl import *
import ctypes

# all angles are in radius

def mat_rotate_z(mat, angle):
    mat1 = mat.copy()
    c = np.cos(angle)
    s = np.sin(angle)
    ix = mat[0:3, 0]
    iy = mat[0:3, 1]
    mat1[0:3, 0] = ix * c - iy * s
    mat1[0:3, 1] = ix * s + iy * c
    return mat1

M16 = ctypes.c_float * 16

def npmat_to_glmat(mat):
    return M16(*np.array(mat.transpose()).reshape(16))

def extract_pos(mat):
    return mat[0:4, 3]

def extract_pos3(mat):
    return mat[0:3, 3] / mat[3, 3]

class Link:
    def __init__(self, parent, child):
        child.parent = parent
        self.parent = parent
        self.child = child

class Joint:
    names = {}

    @staticmethod
    def make_mat(pos3, angle):
        mat = np.matrix(np.eye(4))
        mat[0, 3] = pos3[0]
        mat[1, 3] = pos3[1]
        mat[2, 3] = pos3[2]
        return mat_rotate_z(mat, angle)

    @staticmethod
    def get(name):
        return Joint.names[name]

    def __init__(self, name, localMat, subJoints):
        assert name not in Joint.names
        Joint.names[name] = self

        self._localMatBase = localMat
        self._angle = 0
        self.angle = 0
        self.name = name
        self.parent = None
        self.links = [Link(self, joint) for joint in subJoints]

    @property
    def globalMat(self):
        if self.parent is None:
            return self.localMat
        return self.parent.globalMat().dot(self.localMat)

    @property
    def localMat(self):
        return self._localMat

    @property
    def globalPos(self):
        return extract_pos(self.globalMat)

    @property
    def localPos3(self):
        return extract_pos3(self.localMat)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = angle
        self._localMat = mat_rotate_z(self._localMatBase, angle)

cylinder = WiredCylinder((.2, .8, .2, 1.), .5, 2)
axisSystem = AxisSystem((0., 0., 0., 1.), 2.)

class JointSprite:
    def __init__(self, joint):
        super().__init__()
        self._shapes = [cylinder, axisSystem]
        self.joint = joint
        self.links = [Link(self, JointSprite(x.child)) for x in joint.links]

    def draw(self):
        with utils.glPreserveMatrix():
            mat = self.joint.localMat
            glMultMatrixf(npmat_to_glmat(mat))
            for shape in self._shapes:
                shape.draw()
            for link in self.links:
                with utils.glPrimitive(GL_LINES):
                    glColor3f(0., 0., 0.)
                    glVertex3f(0., 0., 0.)
                    glVertex3f(*link.child.joint.localPos3)
                link.child.draw()

def example_1():
    import testbase
    L1, L2 = 5, 4
    root = Joint('j1', Joint.make_mat((0, 0, 0), 0), [
        Joint('j2', Joint.make_mat((L1, 0, 0), 0), [
            Joint('j3', Joint.make_mat((L2, 0, 0), 0), []),
        ]),
    ])
    rootSp = JointSprite(root)
    eyePos = (4, 4, 4)
    centerPos = (0, 0, 0)
    upPos = (0, 1, 0)
    scaleRate = .4
    @testbase.set_draw_func
    def on_draw():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(*(eyePos + centerPos + upPos))
        glScalef(scaleRate, scaleRate, scaleRate)
        rootSp.draw()

    angle = 0
    @testbase.set_update_func
    def update(dt):
        nonlocal angle
        angle += dt * np.pi / 6
        Joint.get('j1').angle = angle
        Joint.get('j2').angle = angle
        print(angle)

    testbase.start()

example_1()
