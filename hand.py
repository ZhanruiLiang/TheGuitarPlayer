#! /usr/bin/env python3
import numpy as np
from shapes import WiredCylinder, AxisSystem
import utils
from pyglet.gl import *
from display import Display, SubWindow
import ctypes
import testbase
import solver

# All angles are in radius

def mat_rotate_z(mat, angle):
    mat1 = mat.copy()
    ix = mat[0:3, 0].copy()
    iy = mat[0:3, 1].copy()
    c = np.cos(angle)
    s = np.sin(angle)
    mat1[0:3, 0] = ix * c + iy * s
    mat1[0:3, 1] = ix * (-s) + iy * c
    # mat1[0:2, 0:2] = np.matrix([[c, -s], [s, c]]) * mat1[0:2, 0:2]
    return mat1

M16 = ctypes.c_float * 16

def npmat_to_glmat(mat):
    return M16(*np.array(mat.transpose()).reshape(16))

def extract_pos(mat):
    return np.array(mat[0:4, 3]).reshape(4)

def extract_pos3(mat):
    return np.array(mat[0:3, 3] / mat[3, 3]).reshape(3)

class Link:
    def __init__(self, parent, child):
        child.parent = parent
        self.parent = parent
        self.child = child

class Joint:
    names = {}

    @staticmethod
    def clear():
        Joint.names = {}

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

        self.name = name
        self.links = [Link(self, joint) for joint in subJoints]
        self._localMatBase = localMat
        self._globalMat = None
        self._angle = 0
        self.angle = 0
        self.parent = None

    def update(self):
        if self._globalMat is None: 
            return
        self._globalMat = None
        for link in self.links:
            link.child.update()

    @property
    def globalMat(self):
        if self._globalMat is None:
            self._globalMat = self.parent.globalMat.dot(self.localMat) \
                    if self.parent else self.localMat
        return self._globalMat

    @property
    def norm3(self):
        mat = self.globalMat
        return np.array(mat[0:3, 2]).reshape(3)

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
    def globalPos3(self):
        return extract_pos3(self.globalMat)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = angle
        self._localMat = mat_rotate_z(self._localMatBase, angle)
        self.update()

class ShapeSprite:
    def __init__(self, shape):
        self.shape = shape

    def draw(self):
        self.shape.draw()

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

def setup_windows():
    scale1 = .1
    testbase.display.subWindows = [
            SubWindow('From Z', 
                pos=(0, 0), size=(.5, .5),
                eye=(0, 0, 4), center=(0, 0, 0), up=(0, 1, 0),
                scale=scale1,
            ),
            SubWindow('From Y', 
                pos=(0, .5), size=(.5, .5),
                eye=(0, 4, 0), center=(0, 0, 0), up=(0, 0, 1),
                scale=scale1,
            ),
            SubWindow('Perspective', 
                pos=(.5, 0), size=(.5, 1),
                eye=(5, 5, 5), center=(0, 0, 0), up=(0, 1, 0),
                scale=.2,
                ortho=False,
            ),
    ]

Pi = np.pi

def example_1():
    L1, L2, L3, L4 = 5, 5, 5, 5
    root = Joint('j0', Joint.make_mat((0, 0, 0), np.pi/4), [
        Joint('j1', Joint.make_mat((L1, 0, 0), -np.pi/4), [
            Joint('j2', Joint.make_mat((L2, 0, 0), -np.pi/4), [
                Joint('j3', Joint.make_mat((L3, 0, 0), 0), [
                    # Joint('j4', Joint.make_mat((L4, 0, 0), np.pi/4), [
                    # ]),
                ]),
            ]),
        ]),
    ])
    dest = [
            ('j3', (6.5, 0, 0)),
        ]
    return root, dest

def example_2():
    L1, L2, L3 = 5, 4, 3
    d = 2
    J = lambda name, pos, angle, subs: Joint(name, Joint.make_mat(pos, angle), subs)
    root = J('j0', (0, 0, 0), Pi/4, [
        J('j1', (L1, 0, d), -Pi/4, [
            J('j2', (L2, 0, 0), -Pi/4, [
                J('j3', (L2, 0, 0), -Pi/4, [
                ]),
            ]),
        ]),
        J('j4', (L1, 0, -d), -Pi/4, [
            J('j5', (L2, 0, 0), -Pi/4, [
                J('j6', (L2, 0, 0), -Pi/4, [
                ]),
            ]),
        ])
    ])
    dest = [
            ('j3', (9, 0, d)),
            ('j6', (7, 0, -d)),
        ]
    return root, dest

class JointDOF2(Joint):
    def __init__(self, name, localMat, subJoints):
        matY = np.matrix([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
            ])
        matYinv = np.linalg.inv(matY)
        super().__init__(name+'y', np.dot(localMat, matY), [
            Joint(name+'z', matYinv, subJoints)])

def hand_example():
    def make_finger(nameSuffix, mat, L2, L3, L4):
        i = str(nameSuffix)
        return JointDOF2('MCP'+i, mat, [
            Joint('PIP'+i, Joint.make_mat((L2, 0, 0), -Pi/6), [
                Joint('DIP'+i, Joint.make_mat((L3, 0, 0), -Pi/6), [
                    Joint('EP'+i, Joint.make_mat((L3, 0, 0), -Pi/6), [])
                ])
            ])
        ])
    L1, L2, L3, L4 = 10., 4., 3., 2.5
    scales = [0, .8, 1., .9, .7]
    HB = 8 # width
    poss = [(), 
            (L1 * scales[1], 0, HB/2),
            (L1 * scales[2], 0, HB/6),
            (L1 * scales[3], 0, -HB/6),
            (L1 * scales[4], 0, -HB/2),
            ]
    hand = JointDOF2('W', Joint.make_mat((0, 0, 0), Pi/6), [
            make_finger(i, Joint.make_mat(poss[i], 0.),
                L2 * scales[i], L3 * scales[i], L4 * scales[i])
            for i in range(1, 5)
        ])
    dest = [
        ('EP1', (14, 0, HB/3)),
        ('EP2', (13, 0, HB/6)),
        ('EP3', (16, 0, -HB/6)),
        ('EP4', (14, 0, -HB/3)),
    ]
    return hand, dest

def solve(solver, root, dest):
    rootSp = JointSprite(root)
    testbase.display.add(rootSp)
    solver.set_joints(root)
    solver.set_target_pos(dest)
    solver.start_solve()
    plotted = False
    def update(dt):
        nonlocal plotted
        # Joint.get('MCP1y').angle += dt
        if not solver.is_ended():
            solver.step()
        elif not plotted:
            solver.plot()
            plotted = True
            print('iterCount:', solver.iterCount)
            # exit(0)
    pyglet.clock.schedule_interval(update, 1./60)

if __name__ == '__main__':
    cylinder = WiredCylinder((.2, .2, .2, 1.), .2, 1)
    axisSystem = AxisSystem((0., 0., 0., 1.), .2)

    setup_windows()
    testbase.display.add(ShapeSprite(AxisSystem((0, 0, 0, 1), 3)))
    sol = solver.SimpleSolver()
    # sol = solver.DLSSolver()
    # sol = solver.JacobianTransposeSolver()
    # data = example_2()
    data = hand_example()
    solve(sol, *data)
    testbase.start()
