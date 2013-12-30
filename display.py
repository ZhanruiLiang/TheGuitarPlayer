import sys
import pyglet
from pyglet.gl import *
import OpenGL.GL as G
import config
import utils

class SubWindow:
    def __init__(self, title, pos, size, eye, center, up, scale=1., ortho=True):
        self.title = title
        self.pos = pos
        self.size = size
        self.eye = eye
        self.center = center
        self.up = up
        self.scale = scale
        self.ortho = ortho

        self.realPos = None
        self.realSize = None

        # self.labelTitle = pyglet.text.Label(self.title, 
        #         font_name='Times New Roman',
        #         x=1, y=1, font_size=20, width=30, height=10)

    def set_camera(self):
        gluLookAt(*(self.eye + self.center + self.up))
        glScalef(*(self.scale,)*3)

    def adjust_view(self):
        x, y = self.realPos
        w, h = self.realSize
        glViewport(x, y, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        far = 1000
        if self.ortho:
            glOrtho(-w/h, w/h, -1, 1, .1, far)
        else:
            # glFrustum(-w/h, w/h, -1, 1, .1, far)
            gluPerspective(60, w/h, .1, far)
        glMatrixMode(GL_MODELVIEW)

    def draw(self):
        x, y = self.realPos
        w, h = self.realSize

class Display:
    _instance = None

    @staticmethod
    def get_instance():
        if Display._instance is None:
            Display._instance = Display()
        return Display._instance

    def __init__(self):
        self.sprites = []
        self.window = None
        self.size = (1, 1)
        self.subWindows = []

    def set_light(self):
        R = 1
        # G.glLightf(GL_LIGHT0, GL_POSITION, R, 0, 0)
        # G.glLightf(GL_LIGHT1, GL_POSITION, 0, R, 0)
        # G.glLightf(GL_LIGHT2, GL_POSITION, 0, 0, R)

    def draw(self):
        # glViewport(0, 0, self.size[0], self.size[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        width, height = self.size
        far = 1000
        for win in self.subWindows:
            win.adjust_view()
            glLoadIdentity()
            with utils.glPreserveMatrix():
                win.set_camera()
                self.set_light()
                # draw sprites for this window
                for sp in self.sprites:
                    sp.draw()
        self.draw_borders()

    def draw_borders(self):
        glViewport(0, 0, *self.size)
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.size[0], 0, self.size[1])
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        d = 1
        for win in self.subWindows:
            x, y = win.realPos
            w, h = win.realSize
            glColor3f(0., 0., 0.)
            glRectd(x + 1, y + 1, x + w - 1, y + h - 1)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def setup(self):
        self.window = pyglet.window.Window(resizable=True)
        # glutInit(*sys.argv)
        glClearColor(*config.BACKGROUND_COLOR)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        # glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)
        # glBlendFunc(GL_ONE, GL_ONE)

        # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_LIGHT2)

        G.glLightfv(GL_LIGHT0, GL_DIFFUSE, (.8, .8, .8, 1.))

        @self.window.event
        def on_resize(width, height):
            self.size = (width, height)
            for win in self.subWindows:
                self._adjust_subwindow(win)
            return pyglet.event.EVENT_HANDLED

        @self.window.event
        def on_draw():
            self.draw()

    def _adjust_subwindow(self, win):
        width, height = self.size
        win.realPos = int(win.pos[0] * width), int(win.pos[1] * height)
        win.realSize = int(win.size[0] * width), int(win.size[1] * height)

    def add(self, sprite):
        self.sprites.append(sprite)

