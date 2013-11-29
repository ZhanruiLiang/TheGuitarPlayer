from pyglet.gl import *
from OpenGL.GLUT import *
import sys
import pyglet
import shapes

window = pyglet.window.Window(resizable=True)

def init():
    glClearColor(1, 1, 1, 1)
    glColor3f(0., 0., 0.)
    glEnable(GL_DEPTH_TEST)
    # glEnable(GL_BLEND)
    # glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)
    # glBlendFunc(GL_ONE, GL_ONE)

    # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glutInit(*sys.argv)

@window.event
def on_resize(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, width / float(height), .1, 1000)
    glMatrixMode(GL_MODELVIEW)
    return pyglet.event.EVENT_HANDLED

def set_update_func(on_update):
    pyglet.clock.schedule_interval(on_update, 1. / 30)
    return on_update

def set_draw_func(on_draw):
    window.event(on_draw)
    return on_draw

def start():
    init()
    pyglet.app.run()
