from pyglet.gl import *
from OpenGL.GLUT import *
import numpy as np
import sys
import pyglet
import config
import shapes
import sprites
import ik
import model

from display import Display, SubWindow
from fretboard import fretSp, FretPos
from joint import Joint
from ik import RefAngleConstraint, RefPosConstraint, RangeConstraint, ConstraintedSolver

display = Display.get_instance()

def setup_windows():
    scale1 = 10
    display.subWindows = [
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

def add_sprites():
    display.add(fretSp)
    display.add(sprites.JointSprite(model.root))

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
    pyglet.app.run()

def setup_solver(solver):
    Pi = np.pi
    def get_pos(stringIdx, fretIdx, z=0.):
        fp = FretPos(stringIdx, fretIdx)
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
    fretSp.set_marks([FretPos(i, j) for (_, (i, j)) in arrangement])

    # Prepare the solver
    solver.set_joints(model.root)
    solver.set_target_pos([(f, get_pos(i, j)) for (f, (i, j)) in arrangement])
    for (f, (i, j)) in arrangement:
        joint = Joint.get(f)
        s = model.scales[['EP0', 'EP1', 'EP2', 'EP3', 'EP4'].index(f)]
        pos = get_pos(i, j, s * model.L4)
        solver.set_constaints([
            RefPosConstraint(joint.parent, pos, .8)
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
    return solver

def start():
    display.setup()
    setup_windows()
    add_sprites()
    solver = setup_solver(ik.ConstraintedSolver())
    solve(solver)

if __name__ == '__main__':
    start()
