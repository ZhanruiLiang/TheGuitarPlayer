import joint as _joint
import shapes
import model
import utils
from pyglet.gl import *
import OpenGL.GL as G
import numpy as np

class Sprite:
    def draw(self):
        pass

class ShapeSprite:
    def __init__(self, shape):
        self.shape = shape

    def draw(self):
        self.shape.draw()


class JointSprite:
    cylinder = shapes.Cylinder((.2, .8, .2, 1.), .002, 0.01)
    LINK_COLOR = (.9, .9, .1, 1.)
    LINK_RADIUS = .004

    def __init__(self, joint):
        super().__init__()
        self._shapes = [JointSprite.cylinder]
        self.joint = joint
        self.links = [_joint.Link(self, JointSprite(x.child)) for x in joint.links]

    def __repr__(self):
        return repr(self.joint)

    def draw_link(self, link):
        pos = link.child.joint.localPos3
        r = utils.norm(pos)
        if r < 1e-8:
            return
        n = np.cross((0, 0, 1), pos / r)
        a = np.arccos(np.dot((0, 0, 1), pos / r))
        with utils.glPreserveMatrix():
            if utils.norm(n) > 1e-8:
                glRotated(a * 180 / np.pi, *n/utils.norm(n))
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

axisSp = ShapeSprite(model.axisSystem)
