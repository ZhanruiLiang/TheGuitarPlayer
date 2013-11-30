from pyglet.gl import *
from OpenGL.GLUT import *
import sys
import pyglet
import config
import shapes

from display import Display

display = Display.get_instance()
def start():
    display.setup()
    pyglet.app.run()
