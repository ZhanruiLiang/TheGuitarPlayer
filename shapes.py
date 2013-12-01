from pyglet.gl import *
# from OpenGL.GLUT import *
import numpy as np
import utils

class Shape:
    def __init__(self, color):
        self.color = color
        self._displayList = None

    @property
    def displayList(self):
        if self._displayList is None:
            self._displayList = glGenLists(1)
            glNewList(self._displayList, GL_COMPILE)

            glColor4f(*self.color)
            self.compile()

            glEndList()
        return self._displayList

    def compile(self):
        " Slot for overriding. "
        pass

    def draw(self):
        glCallList(self.displayList)

class Cube(Shape):
    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius

    def compile(self):
        glutWireCube(self.radius)

class Square(Shape):
    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius

    def compile(self):
        a = self.radius / 2.
        with utils.glPrimitive(GL_QUADS):
            # glVertex3f(a, a, 0)
            # glVertex3f(-a, a, 0)
            # glVertex3f(-a, -a, 0)
            # glVertex3f(a, -a, 0)

            glVertex3f(a, a, 0)
            glVertex3f(a, -a, 0)
            glVertex3f(-a, -a, 0)
            glVertex3f(-a, a, 0)

class WiredCylinder(Shape):
    def __init__(self, color, radius, height, slices=16):
        super().__init__(color)
        self.radius = radius
        self.height = height
        self.slices = slices

    def compile(self):
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        A = np.linspace(0, np.pi * 2, self.slices, endpoint=True)
        X = np.cos(A) * self.radius
        Y = np.sin(A) * self.radius
        # Draw top face
        with utils.glPrimitive(GL_TRIANGLE_FAN):
            h = self.height / 2.
            glVertex3f(0, 0, h)
            for x, y in zip(X, Y):
                glVertex3f(x, y, h)
        # Draw bottom face
        with utils.glPrimitive(GL_TRIANGLE_FAN):
            h = -self.height / 2.
            glVertex3f(0, 0, h)
            for x, y in zip(*map(reversed, (X, Y))):
                glVertex3f(x, y, h)
        # Draw side faces
        with utils.glPrimitive(GL_QUAD_STRIP):
            h = self.height / 2.
            for x, y in zip(X, Y):
                glVertex3f(x, y, -h)
                glVertex3f(x, y, h)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

class AxisSystem(Shape):
    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius

    def compile(self):
        r = self.radius
        d = 0.05 * r
        h = 0.4 * d
        b = .9

        colors = [(b, .2, .2), (.2, b, .2), (.2, .2, b)]
        vertices = [(r, 0., 0.), (r - d, h, -h), (r - d, -h, h),
                (r, 0., 0.), (r - d, h, h), (r - d, -h, -h)]
        replaces = [(0, 1, 2), (2, 0, 1), (1, 2, 0)]
        with utils.glPrimitive(GL_LINES):
            for color, replace in zip(colors, replaces):
                glColor3f(*color)
                glVertex3f(0., 0., 0.)
                glVertex3f(*[(r, 0., 0.)[replace[i]] for i in range(3)])
        with utils.glPrimitive(GL_TRIANGLES):
            for color, replace in zip(colors, replaces):
                glColor3f(*color)
                for v in vertices:
                    glVertex3f(*[v[replace[i]] for i in range(3)])
