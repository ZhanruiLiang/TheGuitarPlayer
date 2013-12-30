import fretboard
import numpy as np
import config
from joint import Joint, JointDOF2, JointDOF3
import shapes

Pi = np.pi

scales = [0, .8, 1., .9, .7]
L1, L2, L3, L4 = .10, .04, .03, .025
Larm = .28
HB = 0.08 # width

def make_hand():
    def make_finger(nameSuffix, mat, L2, L3, L4):
        i = str(nameSuffix)
        return JointDOF2('MCP'+i, mat, (-Pi/9, Pi/9), (-Pi/2, 0), [
            Joint('PIP'+i, Joint.make_mat((L2, 0, 0), -Pi/6), (-Pi/2, 0), [
                Joint('DIP'+i, Joint.make_mat((L3, 0, 0), -Pi/6), (-Pi/2, 0), [
                    Joint('EP'+i, Joint.make_mat((L3, 0, 0), -Pi/6), None, [])
                ])
            ])
        ])
    # All lengths are in meters
    poss = [(), 
            (L1 * scales[1], 0, HB/2),
            (L1 * scales[2], 0, HB/6),
            (L1 * scales[3], 0, -HB/6),
            (L1 * scales[4], 0, -HB/2),
            ]
    hand = JointDOF3('W', np.array([
        # wrist matrix
            [1, 0, 0, Larm],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]), 
            None, None, None, [
            make_finger(i, Joint.make_mat(poss[i], 0.),
                L2 * scales[i], L3 * scales[i], L4 * scales[i])
            for i in range(1, 5)
        ])

    arm = JointDOF3('ARM', np.array([
            [0, 0, -1, .04 - config.x0],
            [1, 0, 0, .03 - config.y0 - Larm],
            [0, -1, 0, -config.z0],
            [0, 0, 0, 1],
        ]), None, None, None, [hand])

    root = arm
    return root

root = make_hand()

axisSystem = shapes.AxisSystem((0., 0., 0., 1.), .01)
