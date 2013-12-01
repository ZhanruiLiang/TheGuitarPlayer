#! /usr/bin/env python3
import numpy as np
from shapes import WiredCylinder, AxisSystem
import utils
from pyglet.gl import *
from display import Display, SubWindow
import testbase
import fretboard

# All angles are in radius

x0, y0, z0 = .06, 0., -.06

def mat_rotate_z(mat, angle):
    mat1 = mat.copy()
    ix = mat[0:3, 0].copy()
    iy = mat[0:3, 1].copy()
    c = np.cos(angle)
    s = np.sin(angle)
    mat1[0:3, 0] = ix * c + iy * s
    mat1[0:3, 1] = ix * (-s) + iy * c
    return mat1

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
        mat = np.eye(4)
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
        return utils.extract_pos(self.globalMat)

    @property
    def localPos3(self):
        return utils.extract_pos3(self.localMat)

    @property
    def globalPos3(self):
        return utils.extract_pos3(self.globalMat)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = angle
        self._localMat = mat_rotate_z(self._localMatBase, angle)
        self.update()

class JointDOF2(Joint):
    def __init__(self, name, localMat, subJoints):
        matY = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
            ])
        matYinv = np.linalg.inv(matY)
        super().__init__(name+'y', np.dot(localMat, matY), [
            Joint(name+'z', matYinv, subJoints)])

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
            glMultMatrixf(utils.npmat_to_glmat(mat))
            for shape in self._shapes:
                shape.draw()
            for link in self.links:
                with utils.glPrimitive(GL_LINES):
                    glColor3f(0., 0., 0.)
                    glVertex3f(0., 0., 0.)
                    glVertex3f(*link.child.joint.localPos3)
                link.child.draw()

def setup_windows():
    scale1 = 10
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
                scale=3 * scale1,
                ortho=False,
            ),
    ]

norm = lambda x:np.sqrt(x.dot(x))

class SimpleSolver:
    """
    positions: vector of size m
    targetPoss: vector of size m
    angles: vector of size n
    """
    def __init__(self):
        self.joints = []
        self.targets = []
        self.cTargets = []
        self.rTargets = []

    def set_joints(self, rootJoint):
        def search_all(joint):
            if joint.links:
                self.joints.append(joint)
            for link in joint.links:
                search_all(link.child)
        search_all(rootJoint)
        self.n = len(self.joints)

    def set_target_pos(self, jointPosPairs):
        """
        @param jointPosPairs: [(str, Pos3D)], a list of (jointName, effectorPos) 
            pairs.
        """
        self.m = m = 3 * len(jointPosPairs)
        self.targetPoss = T = np.zeros(m, dtype=np.double)
        for i, (jname, pos) in enumerate(jointPosPairs):
            self.targets.append(Joint.get(jname))
            T[3*i:3*i+3] = pos

    def set_constraint_angles(self, jointAngleWeights):
        pass

    def set_reference_angles(self, jointAngleWeights):
        pass

    @staticmethod
    def is_parent(j1, j2):
        while j2:
            if j2 is j1:
                return True
            j2 = j2.parent
        return False

    def make_jacobian(self):
        """
        make a m x n Jacobian matrix, where m is the number of effectors
        and n is the number of angles.

        J(i, j) = D(position(i), angle(j)) = norm(j) x (p(i) - p(j))
        """
        J = np.array(np.zeros((self.m, self.n), dtype=np.double))
        js = self.joints
        P = self.positions
        for i in range(0, self.m, 3):
            for j in range(self.n):
                if self.is_parent(js[j], self.targets[i//3]):
                    J[i:i+3, j] = np.cross(js[j].norm3, P[i:i+3] - js[j].globalPos3)
                else:
                    J[i:i+3, j] = 0
        return J

    def start_solve(self):
        self.angles = np.array([x.angle for x in self.joints], dtype=np.double)
        self._ended = not self.targets
        self.iterCount = 0
        self.ds = []

    def is_ended(self):
        return self._ended

    def plot(self):
        import pylab
        pylab.plot(self.ds)
        pylab.show()

    def make_delta_angles(self):
        J = self.make_jacobian()
        # solve : J dA = dP = T - S for dA
        e = self.clamp_err(self.targetPoss - self.positions)
        deltaAngles = np.linalg.lstsq(J, e)[0]
        # print('--------i:', self.iterCount)
        # print('globalPos:', np.array([x.globalPos3 for x in self.joints]))
        # print('norms:', np.array([x.norm3 for x in self.joints]))
        # print('e:', e)
        # print('J:')
        # print(J)
        # print('dA:', deltaAngles)
        # print('close?:', np.allclose(np.dot(J, deltaAngles), e))
        return self.clamp_step(deltaAngles)

    d1 = .005 # stop thresold
    d2 = .05 # step length
    d3 = .02 # error clamping value

    def clamp_err(self, e):
        d3 = self.d3
        for i in range(0, self.m, 3):
            w = e[i:i+3]
            wL = norm(w)
            if wL > d3:
                e[i:i+3] = d3 / wL * w
        return e

    def clamp_step(self, dA):
        d = norm(dA)
        if d > self.d2:
            return self.d2 / norm(dA) * dA
        else:
            return dA

    def step(self):
        # In the following comment:
        #   A stands for "angle"
        #   P stands for "position"
        #   T stands for "target position vector"
        #   S stands for "current position vector"
        js = self.joints
        self.positions = np.hstack([x.globalPos3
            for x in self.targets])
        d = norm(self.targetPoss - self.positions)
        print('d:', d)
        self.ds.append(d)
        if d <= self.d1: 
            self._ended = True
            return
        self.angles += self.make_delta_angles()
        for joint, angle in zip(js, self.angles):
            joint.angle = angle
        self.iterCount += 1

class DLSSolver(SimpleSolver):
    # d2 = .05
    k = .2
    def make_delta_angles(self):
        J = self.make_jacobian()
        k2 = self.k**2
        e = self.clamp_err(self.targetPoss - self.positions)
        Jt = J.transpose()
        A = np.dot(J, Jt) + k2 * np.eye(self.m)
        f = np.linalg.solve(A, e)
        assert np.allclose(np.dot(A, f), e)
        return self.clamp_step(np.dot(Jt, f))

class JacobianTransposeSolver(SimpleSolver):
    def make_delta_angles(self):
        J = self.make_jacobian()
        e = self.clamp_err(self.targetPoss - self.positions)
        Jt = J.transpose()
        J1 = np.dot(np.dot(J, Jt), e)
        a = np.dot(e, J1) / np.dot(J1, J1)
        return self.clamp_step(a * np.dot(Jt, e))

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
    # All lengths are in meters
    L1, L2, L3, L4 = .10, .04, .03, .025
    scales = [0, .8, 1., .9, .7]
    HB = 0.08 # width
    poss = [(), 
            (L1 * scales[1], 0, HB/2),
            (L1 * scales[2], 0, HB/6),
            (L1 * scales[3], 0, -HB/6),
            (L1 * scales[4], 0, -HB/2),
            ]
    hand = JointDOF2('W', np.array([
        # wrist matrix
            [0, 0, -1, .04 - x0],
            [0, 1, 0, 0 - y0],
            [1, 0, 0, -.12 - z0],
            [0, 0, 0, 1],
        ]), [
            make_finger(i, Joint.make_mat(poss[i], 0.),
                L2 * scales[i], L3 * scales[i], L4 * scales[i])
            for i in range(1, 5)
        ])
    def get_pos(stringIdx, fretIdx):
        fp = fretboard.FretPos(stringIdx, fretIdx)
        x, y = fretSp.pos_fret_to_plane(fp)
        return np.dot(fretSp.localMat, [x, y, 0, 1])[:3]
    X1 = [
        ('EP1', (6, 1)),
        ('EP2', (2, 2)),
        ('EP3', (3, 3)),
        ('EP4', (1, 3)),
    ]
    Am = [
        ('EP1', (2, 1)),
        ('EP2', (4, 2)),
        ('EP3', (3, 2)),
    ]
    arrangment = Am
    fretSp.set_marks([fretboard.FretPos(i, j) for (_, (i, j)) in arrangment])
    dest = [(f, get_pos(i, j)) for (f, (i, j)) in arrangment]
    return hand, dest

def solve(solver, root, dest):
    rootSp = JointSprite(root)
    testbase.display.add(rootSp)
    solver.set_joints(root)
    solver.set_target_pos(dest)
    solver.start_solve()
    plotted = False
    # plotted = True
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
    cylinder = WiredCylinder((.2, .2, .2, 1.), .002, 0.01)
    axisSystem = AxisSystem((0., 0., 0., 1.), .01)

    setup_windows()
    testbase.display.add(ShapeSprite(AxisSystem((0, 0, 0, 1), .1)))
    fretSp = fretboard.FretBoardSprite(np.array([
        [1, 0, 0, 0 - x0],
        [0, 0, 1, 0 - y0],
        [0, -1, 0, 0 - z0],
        [0, 0, 0, 1],
        ]), top=.06, length=.70, bottom=.08, nFrets=19)
    testbase.display.add(fretSp)
    sol = SimpleSolver()
    # sol = DLSSolver()
    # sol = JacobianTransposeSolver()
    # data = example_2()
    data = hand_example()
    solve(sol, *data)
    testbase.start()
