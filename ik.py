from joint import Joint
import numpy as np
import utils

norm = utils.norm
dot = utils.dot

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
