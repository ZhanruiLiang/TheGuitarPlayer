import utils
import numpy as np

dot = utils.dot

norm = utils.norm

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

    def __init__(self, name, localMat, range, subJoints):
        assert name not in Joint.names
        Joint.names[name] = self

        self.name = name
        self.links = [Link(self, joint) for joint in subJoints]
        self._localMatBase = localMat
        self._globalMat = None
        self._angle = 0
        self.angle = 0
        self.range = range
        self.parent = None

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

    def traverse(self):
        for link in self.links:
            joint = link.child
            yield from joint.traverse()

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
    def __init__(self, name, localMat, rangeY, rangeZ, subJoints):
        matY = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
            ])
        matYinv = np.linalg.inv(matY)
        super().__init__(name+'y', dot(localMat, matY), rangeY, [
            Joint(name+'z', matYinv, rangeZ, subJoints)])

class JointDOF3(Joint):
    def __init__(self, name, localMat, rangeX, rangeY, rangeZ, subJoints):
        assert isinstance(localMat, np.ndarray)
        matX = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            ])
        matXinv = np.linalg.inv(matX)
        super().__init__(name+'x', dot(localMat, matX), rangeX, [
            JointDOF2(name, matXinv, rangeY, rangeZ, subJoints)])

