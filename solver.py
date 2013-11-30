import numpy as np

norm = lambda x:np.sqrt(x.dot(x))

class SimpleSolver:
    """
    positions: vector of size m
    targetPoss: vector of size m
    angles: vector of size n
    """
    def __init__(self):
        pass

    def set_joints(self, rootJoint):
        js = self.joints = []
        def search_all(joint):
            self.joints.append(joint)
            for link in joint.links:
                search_all(link.child)
        search_all(rootJoint)
        self.nameToId = {}
        for i, joint in enumerate(self.joints):
            self.nameToId[joint.name] = i
        self.n = len(self.joints)

    def set_target_pos(self, jointPosPairs):
        """
        @param jointPosPairs: [(str, Pos3D)], a list of (jointName, effectorPos) 
            pairs.
        """
        self.targetIDs = []
        self.m = m = 3 * len(jointPosPairs)
        self.targetPoss = T = np.zeros(m, dtype=np.double)
        for i, (jname, pos) in enumerate(jointPosPairs):
            self.targetIDs.append(self.get_id(jname))
            T[3*i:3*i+3] = pos

    def get_id(self, jname):
        return self.nameToId[jname]

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
                if self.is_parent(js[j], js[self.targetIDs[i//3]]):
                    J[i:i+3, j] = np.cross(js[j].norm3, P[i:i+3] - js[j].globalPos3)
                else:
                    J[i:i+3, j] = 0
        return J

    def start_solve(self):
        self.angles = np.array([x.angle for x in self.joints], dtype=np.double)
        self._ended = not self.targetIDs
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

    d1 = .05 # stop thresold
    d2 = .5 # step length
    d3 = 2 # error clamping value

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
        self.positions = np.hstack([js[x].globalPos3
            for x in self.targetIDs])
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
    k = 2
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
