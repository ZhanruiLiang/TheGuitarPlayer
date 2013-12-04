#! /usr/bin/env python3
import numpy as np
from shapes import WiredCylinder, AxisSystem, Cylinder
import utils
from pyglet.gl import *
from display import Display, SubWindow
import testbase
import fretboard
import OpenGL.GL as G

# All angles are in radius

x0, y0, z0 = .06, 0., -.06

dot = np.dot

norm = lambda x:np.sqrt(x.dot(x))

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

    def __repr__(self):
        return 'Link({}, {})'.format(self.parent, self.child)

class Joint:
    names = {}

    def __repr__(self):
        return 'Joint(name={})'.format(self.name)

    def is_parent_of(self, j2):
        while j2:
            if j2 is self:
                return True
            j2 = j2.parent
        return False

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
        super().__init__(name+'y', dot(localMat, matY), [
            Joint(name+'z', matYinv, subJoints)])

class JointDOF3(Joint):
    def __init__(self, name, localMat, subJoints):
        matX = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            ])
        matXinv = np.linalg.inv(matX)
        super().__init__(name+'x', dot(localMat, matX), [
            JointDOF2(name, matXinv, subJoints)])

class ShapeSprite:
    def __init__(self, shape):
        self.shape = shape

    def draw(self):
        self.shape.draw()

class JointSprite:
    LINK_COLOR = (.9, .9, .1, 1.)
    LINK_RADIUS = .004

    def __init__(self, joint):
        super().__init__()
        self._shapes = [cylinder, axisSystem]
        self.joint = joint
        self.links = [Link(self, JointSprite(x.child)) for x in joint.links]

    def __repr__(self):
        return repr(self.joint)

    def draw_link(self, link):
        pos = link.child.joint.localPos3
        r = norm(pos)
        if r < 1e-8:
            return
        n = np.cross((0, 0, 1), pos / r)
        a = np.arccos(np.dot((0, 0, 1), pos / r))
        with utils.glPreserveMatrix():
            if norm(n) > 1e-8:
                glRotated(a * 180 / np.pi, *n/norm(n))
            quad = gluNewQuadric()
            G.glMaterialfv(GL_FRONT, GL_DIFFUSE, self.LINK_COLOR)
            gluCylinder(quad, self.LINK_RADIUS, self.LINK_RADIUS, r, 16, 1)
            gluDeleteQuadric(quad)
        # glDisable(GL_LIGHTING)
        # with utils.glPrimitive(GL_LINES):
        #     glColor3i(0, 0, 0)
        #     glVertex3f(0, 0, 0)
        #     glVertex3d(*pos)
        # glEnable(GL_LIGHTING)

    def draw(self):
        with utils.glPreserveMatrix():
            mat = self.joint.localMat
            glMultMatrixf(utils.npmat_to_glmat(mat))
            for shape in self._shapes:
                shape.draw()
            # draw links
            for link in self.links:
                self.draw_link(link)
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
            SubWindow('From X', 
                pos=(.5, 0), size=(.5, .5),
                eye=(4, 0, 0), center=(0, 0, 0), up=(0, 1, 0),
                scale=scale1,
            ),
            SubWindow('Perspective', 
                pos=(.5, .5), size=(.5, .5),
                eye=(5, 10, 5), center=(0, 0, 0), up=(0, 1, 0),
                scale=5 * scale1,
                ortho=False,
            ),
    ]


class SimpleSolver:
    """
    positions: vector of size m
    targetPoss: vector of size m
    angles: vector of size n
    """
    def __init__(self):
        self.joints = []
        self.targets = []

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
                if js[j].is_parent_of(self.targets[i//3]):
                    J[i:i+3, j] = np.cross(js[j].norm3, P[i:i+3] - js[j].globalPos3)
                else:
                    J[i:i+3, j] = 0
        return J

    def start_solve(self):
        self.angles = np.array([x.angle for x in self.joints], dtype=np.double)
        self.iterCount = 0
        self.ds = []
        if self.targets:
            self.d = self.d1 + 1
        else:
            self.d = 0

    def is_ended(self):
        return self.d < self.d1

    def plot(self):
        import pylab
        pylab.plot(self.ds)
        pylab.show()

    def make_err(self):
        return self.clamp_err(self.targetPoss - self.positions)

    def make_delta_angles(self):
        J = self.make_jacobian()
        # solve : J dA = dP = T - S for dA
        e = self.make_err()
        deltaAngles = np.linalg.lstsq(J, e)[0]
        # print('--------i:', self.iterCount)
        # print('globalPos:', np.array([x.globalPos3 for x in self.joints]))
        # print('norms:', np.array([x.norm3 for x in self.joints]))
        # print('e:', e)
        # print('J:')
        # print(J)
        # print('dA:', deltaAngles)
        # print('close?:', np.allclose(dot(J, deltaAngles), e))
        return self.clamp_step(deltaAngles)

    d1 = .005 # stop thresold
    d2 = .05 # step length
    d3 = .01 # error clamping value, 0.01 should be good

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
        self.angles += self.make_delta_angles()
        for joint, angle in zip(js, self.angles):
            joint.angle = angle
        self.iterCount += 1
        self.d = d = norm(self.targetPoss - self.positions)
        print('d:', d)
        self.ds.append(d)

class DLSSolver(SimpleSolver):
    # d2 = .05
    k = .2
    def make_delta_angles(self):
        J = self.make_jacobian()
        k2 = self.k**2
        e = self.make_err()
        Jt = J.transpose()
        A = dot(J, Jt) + k2 * np.eye(self.m)
        f = np.linalg.solve(A, e)
        assert np.allclose(dot(A, f), e)
        return self.clamp_step(dot(Jt, f))

class JacobianTransposeSolver(SimpleSolver):
    def make_delta_angles(self):
        J = self.make_jacobian()
        e = self.make_err()
        Jt = J.transpose()
        J1 = dot(dot(J, Jt), e)
        a = dot(e, J1) / dot(J1, J1)
        return self.clamp_step(a * dot(Jt, e))

class Constraint:
    K = 1.
    # If the return value is None, this constraint will be ignore is 
    # this iteration.
    def get_err(self):
        raise NotImplemented

    def get_deriv(self, joint):
        raise NotImplemented

class RefAngleConstraint(Constraint):
    size = 1
    K = 0.1
    def __init__(self, joint, angle, weight):
        self.joint = joint
        self.angle = angle
        self.weight = weight

    def __repr__(self):
        return 'RefAngleConstraint(jname={}, a={}, w={})'.format(
                self.joint.name, self.angle, self.weight)

    def get_err(self):
        e = self.weight * (self.angle - self.joint.angle)
        if e > self.K:
            e = np.sign(e) * self.K
        return e

    def get_deriv(self, joint):
        " @return: D(self.joint.angle, joint.angle) "
        if joint is self.joint:
            return self.weight
        else:
            return 0.

class RefPosConstraint(Constraint):
    size = 3
    def __init__(self, joint, pos, weight):
        self.joint = joint
        self.pos = pos
        self.weight = weight

    def __repr__(self):
        return 'RefPosConstraint(jname={}, p={}, w={})'.format(
                self.joint.name, self.pos, self.weight)

    def get_err(self):
        return self.K * self.weight * (self.pos - self.joint.globalPos3)

    def get_deriv(self, joint):
        " @return: D(self.joint.angle, joint.angle) "
        if joint.is_parent_of(self.joint):
            return self.weight * np.cross(joint.norm3, self.joint.globalPos3 - joint.globalPos3)
        else:
            return 0.

class RangeConstraint(Constraint):
    size = 1
    def __init__(self, joint, minAngle, maxAngle, weight):
        self.joint = joint
        self.minAngle = minAngle
        self.maxAngle = maxAngle
        self.weight = weight

    def __repr__(self):
        return 'RangeConstraint(jname={}, r={}, w={})'.format(
                self.joint.name, (self.minAngle, self.maxAngle), self.weight)

    def get_err(self):
        a = self.joint.angle
        m1, m2 = self.minAngle, self.maxAngle
        if a < m1:
            return self.K * self.weight * (m1 - a)
        elif a > m2:
            return self.K * self.weight * (m2 - a)
        else:
            return None

    def get_deriv(self, joint):
        " @return: D(self.joint.angle, joint.angle) "
        if joint is self.joint:
            return self.weight
        else:
            return 0.

class ConstraintedSolver(SimpleSolver):
    def __init__(self):
        super().__init__()
        self.constraints = []

    def set_constaints(self, constraints):
        self.constraints.extend(constraints)

    def start_solve(self):
        super().start_solve()
        self.dL = self.d1 + 1

    def make_jacobian_low(self, constraints):
        # use PLow to make JLow
        m = sum(c.size for c in constraints)
        n = len(self.joints)
        JL = np.zeros((m, n), dtype=np.double)
        i = 0
        for c in constraints:
            for j, joint in enumerate(self.joints):
                JL[i:i+c.size, j] = c.get_deriv(joint)
            i += c.size
        return JL

    def make_err_low(self):
        eL = []
        effectiveConstraints = []
        for c in self.constraints:
            e1 = c.get_err()
            if e1 is None:
                continue
            effectiveConstraints.append(c)
            eL.append(e1)
        eL = np.hstack(eL).astype(np.double)
        return eL, effectiveConstraints

    dL1 = 0.03
    def is_ended(self):
        return self.d < self.d1 and self.dL < self.dL1

    def make_delta_angles(self):
        e = self.make_err()
        J = self.make_jacobian()
        Jp = np.linalg.pinv(J)
        a = dot(Jp, e)
        k = 10
        # a = np.linalg.lstsq(J, e)[0]
        if self.constraints:
            T = np.eye(Jp.shape[0]) - dot(Jp, J)
            eL, cs = self.make_err_low()
            self.dL = norm(eL)
            # print('eL:', eL)
            print('dL:', self.dL)
            print('d:', norm(e))
            # print('cs:', cs)
            if cs:
                JL = self.make_jacobian_low(cs)
                deL = eL - dot(JL, a)
                # method 1
                S = dot(JL, T)
                W = dot(S, S.T) + k * np.eye(S.shape[0])
                y = dot(S.T, np.linalg.solve(W, deL))
                # method 2
                # u = np.linalg.lstsq(dot(JL, T), deL)
                # print('lstsq:', u)
                # y = u[0]
                # print('T:', T)
                # print('y:', y)
                # print('Ty:', dot(T, y))
                # print('norm:', norm(dot(J, dot(T, y))))
                a += dot(T, y)
        else:
            self.dL = 0
        return a

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


def add_sprite(sprite):
    testbase.display.add(sprite)

def hand_example(solver):
    fretSp = fretboard.FretBoardSprite(np.array([
        [1, 0, 0, 0 - x0],
        [0, 0, 1, 0 - y0],
        [0, -1, 0, 0 - z0],
        [0, 0, 0, 1],
        ]), top=.08, length=.90, bottom=.08, nFrets=19)
    add_sprite(fretSp)

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
    Larm = .28
    scales = [0, .8, 1., .9, .7]
    HB = 0.08 # width
    poss = [(), 
            (L1 * scales[1], 0, HB/2),
            (L1 * scales[2], 0, HB/6),
            (L1 * scales[3], 0, -HB/6),
            (L1 * scales[4], 0, -HB/2),
            ]
    hand = JointDOF3('W', np.array([
        # wrist matrix
            # [0, 0, -1, .04 - x0],
            # [0, 1, 0, 0 - y0],
            # [1, 0, 0, -.12 - z0],
            # [0, 0, 0, 1],
            [1, 0, 0, Larm],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]), [
            make_finger(i, Joint.make_mat(poss[i], 0.),
                L2 * scales[i], L3 * scales[i], L4 * scales[i])
            for i in range(1, 5)
        ])

    arm = JointDOF3('ARM', np.array([
            [0, 0, -1, .04 - x0],
            [1, 0, 0, .03 - y0 - Larm],
            [0, -1, 0, -z0],
            [0, 0, 0, 1],
        ]), [hand])

    root = arm
        
    add_sprite(JointSprite(root))

    def get_pos(stringIdx, fretIdx, z=0.):
        fp = fretboard.FretPos(stringIdx, fretIdx)
        x, y = fretSp.pos_fret_to_plane(fp)
        return np.dot(fretSp.localMat, [x, y, z, 1])[:3]
    X1 = [ ('EP1', (6, 1)), ('EP2', (2, 2)), ('EP3', (3, 3)), ('EP4', (1, 3)), ]
    Am = [ ('EP1', (2, 1)), ('EP2', (4, 2)), ('EP3', (3, 2)), ]
    F = [ ('EP1', (6, 1)), ('EP2', (3, 2)), ('EP3', (5, 3)), ('EP4', (4, 3)), ]

    C = [ ('EP1', (2, 1)), ('EP2', (4, 2)), ('EP3', (5, 3)), ] 
    bar = 0
    C7 = [ ('EP1', (2, bar+1)), ('EP2', (4, bar+2)), ('EP3', (5, bar+3)), ('EP4', (1, bar+3)), ]
    G = [ ('EP3', (6, 3)), ('EP2', (5, 2)), ('EP4', (1, 3)), ]
    Em = [ ('EP2', (5, 2)), ('EP3', (4, 2)), ]
    Dm = [ ('EP1', (1, 1)), ('EP2', (3, 2)), ('EP4', (2, 3)), ('EP3', (4, 3)), ]
    GB = [ ('EP1', (6, 3)), ('EP4', (1, 7)), ]
    B7 = [ ('EP1', (4, 1)), ('EP2', (5, 2)), ('EP3', (3, 2)), ('EP4', (1, 2)), ]
    A = [('EP1', (4, 2)),('EP2', (3, 2)),('EP3', (2, 2))]

    # Choose an arrangement
    arrangement = A
    fretSp.set_marks([fretboard.FretPos(i, j) for (_, (i, j)) in arrangement])

    # Prepare the solver
    solver.set_joints(root)
    solver.set_target_pos([(f, get_pos(i, j)) for (f, (i, j)) in arrangement])
    for (f, (i, j)) in arrangement:
        joint = Joint.get(f)
        s = scales[['EP0', 'EP1', 'EP2', 'EP3', 'EP4'].index(f)]
        pos = get_pos(i, j, s * L4)
        solver.set_constaints([
            RefPosConstraint(joint.parent, pos, .8)
            ])

    for i in range(1, 5):
        # MCP, PIP, DIP, EP
        i = str(i)
        solver.set_constaints([
            RangeConstraint(Joint.get('MCP'+i+'z'), -Pi/2, 0, 1.),
            RangeConstraint(Joint.get('MCP'+i+'y'), -Pi/9, Pi/9, 1.),
            RangeConstraint(Joint.get('PIP'+i), -Pi/2, 0, 1.),
            RangeConstraint(Joint.get('DIP'+i), -Pi/2, 0, 1.),
        ])
    # define constraints
    solver.set_constaints([
        # RefAngleConstraint(Joint.get('MCP1z'), -Pi/6, 1),
        # RefAngleConstraint(Joint.get('MCP2z'), -Pi/6, 1),
        # RefAngleConstraint(Joint.get('MCP3z'), -Pi/6, 1),
        # RefAngleConstraint(Joint.get('MCP4z'), -Pi/6, 1),
        RefAngleConstraint(Joint.get('MCP1y'), -Pi/8, 1),
        RefAngleConstraint(Joint.get('MCP2y'), 0, 1),
        RefAngleConstraint(Joint.get('MCP3y'), 0, 1),
        RefAngleConstraint(Joint.get('MCP4y'), Pi/8, 1),
    ])

def solve(solver):
    solver.start_solve()
    plotted = False
    # plotted = True
    def update(dt):
        nonlocal plotted
        if not solver.is_ended():
            solver.step()
        elif not plotted:
            solver.plot()
            plotted = True
            print('iterCount:', solver.iterCount)
            # exit(0)
    pyglet.clock.schedule_interval(update, 1./60)
    testbase.start()

if __name__ == '__main__':
    cylinder = Cylinder((.2, .8, .2, 1.), .002, 0.01)
    axisSystem = AxisSystem((0., 0., 0., 1.), .01)

    setup_windows()
    testbase.display.add(ShapeSprite(AxisSystem((0, 0, 0, 1), .1)))
    # sol = SimpleSolver()
    # sol = DLSSolver()
    # sol = JacobianTransposeSolver()
    sol = ConstraintedSolver()
    # data = example_2()
    hand_example(sol)
    solve(sol)
