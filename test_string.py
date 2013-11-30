import numpy as np
import pylab
import fretboard

L = 70.
d0 = 0.1
d1 = 0.4
s = 0.5
X = np.array(fretboard.make_lens(L, 20))
r = 0.3
X = (1 - r) * X[:-1] + r * X[1:]
D = d0 + (d1 - d0) / L * X
Y = np.sqrt(X**2 + D**2) + np.sqrt((L - X)**2 + D**2) - L
pylab.plot(X, Y, '.-')
pylab.show()
