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

    def __init__(self, root):
        super().__init__()
        self.root = root

    def __repr__(self):
        return repr(self.joint)

    def draw_link(self, link):
        pos = link.child.localPos3
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

    def draw_joint(self, joint):
        with utils.glPreserveMatrix():
            mat = joint.localMat
            glMultMatrixf(utils.npmat_to_glmat(mat))
            self.cylinder.draw()
            # draw links
            for link in joint.links:
                self.draw_link(link)
                self.draw_joint(link.child)

    def draw(self):
        self.draw_joint(self.root)

axisSp = ShapeSprite(model.axisSystem)
