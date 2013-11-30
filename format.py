import fractions
import math

class TimeDelta:
    def __init__(self, dt):
        self.dt = fractions.Fraction(dt)

    def div2(self):
        return self.div(2)

    def div(self, x):
        return TimeDelta(self.dt / x)

    def dotted(self):
        return TimeDelta(self.dt * 3 / 2)

    def __repr__(self):
        return 'TimeDelta(%r)' % self.dt

Temperament = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FreqA4 = 440
DP = 2 ** (1./12)

def comb_name(t, y):
    if len(t) == 1:
        return t + str(y)
    else:
        return t[0] + str(y) + t[1]

def split_name(name):
    if len(name) == 2:
        x, y = Temperament.index(name[0]), int(name[1])
    else:
        x, y = Temperament.index(name[0] + name[2]), int(name[1])
    return x, y

class Pitch:
    """
    A4 = 440Hz
    """
    def __init__(self, freq):
        self.freq = freq

    @staticmethod
    def from_name(name):
        freq = Pitch.name_to_freq(name)
        return Pitch(freq)

    def get_name_simple(self):
        x, y = split_name(Pitch.freq_to_name(self.freq))
        return Temperament[x]

    def get_name(self):
        return Pitch.freq_to_name(self.freq)

    def sharpen(self):
        return Pitch(self.freq * DP)

    def flatten(self):
        return Pitch(self.freq / DP)

    @staticmethod
    def name_to_freq(name):
        return FreqA4 * (DP ** (Pitch.name_to_pos(name) - PosA4))

    @staticmethod
    def freq_to_name(freq):
        pos = int(.5 + PosA4 + 12 * math.log(freq / FreqA4) / math.log(2))
        x, y = pos % 12, pos / 12
        return comb_name(Temperament[x], y)

    @staticmethod
    def name_to_pos(name):
        x, y = split_name(name)
        return x + y * 12

    @staticmethod
    def pos_to_name(pos):
        x, y = pos % 12, pos / 12
        return comb_name(Temperament(x), y)

PosA4 = Pitch.name_to_pos('A4')

class Note:
    """
    pitch
    startTime
    duration
    """
    def __init__(self, pitch, startTime, duration):
        self.pitch = pitch
        self.startTime = startTime
        self.duration = duration

class Section:
    def __init__(self, notes):
        self.notes = list(notes)

    def new_notes(self, notes):
        pass

class Score:
    def __init__(self):
        self.sections = []

    def change_staff(raises):
        pass

    def new_section(self):
        pass

def gen_tikz(sections):
    head = r"""\documentclass{article}
\usepackage[left=2cm, right=1cm]{geometry}
\usepackage{tikz}
\usetikzlibrary{backgrounds}

\begin{document}
\begin{figure} \begin{center}
\begin{tikzpicture}[scale=5, show background rectangle]
    """
    foot = r"""
\end{tikzpicture}
\end{center} \end{figure}
\end{document}
    """
    print head
    vis = set()
    xMax = len(sections) + .02
    print r"""
    \draw[->] (0, 0) -- ({xMax}, 0);
    \foreach \x in {{ 0, 0.25, ..., {xMax} }}
        \draw (\x, 0) -- (\x, -0.01) node[below]  {{\tiny \x}};
    \foreach \x in {{ 0.125, 0.25, ..., {xMax} }}
        \draw[dotted, color=gray] (\x, 0) -- (\x, 2);
    \draw[->] (0, 0) -- (0, 2);
    """.format(xMax = xMax)

    for i, section in enumerate(sections):
        for note in section.notes:
            x1 = note.startTime + i
            y1 = (math.log(note.pitch.freq) - math.log(Pitch.name_to_freq('C3')))
            x2 = x1 + note.duration.dt - 0.01
            y2 = y1 + 0.05
            print r'    \draw[fill=gray] ({x1}, {y1}) rectangle ({x2}, {y2});'.format(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            name = note.pitch.get_name().replace('#', '\\#')
            if name not in vis:
                print r'    \draw[dotted] ({x1}, {y1}) -- (0, {y1}) node[left] {{\tiny {name}({freq:.2f}Hz) }};'.format(
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), 
                        name=name, freq=note.pitch.freq)
                vis.add(name)
            else:
                print r'    \draw[dotted] ({x1}, {y1}) -- (0, {y1});'.format(
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), 
                        name=name, freq=note.pitch.freq)
        print r'    \draw[dotted, thick, color=gray] ({x}, 0) -- ({x}, 2);'.format(x=i + 1)


    print foot

def sample1():
    # raised = []
    flattened = []

    raised = ['C', 'F']
    def convert(p):
        if p.get_name_simple() in raised:
            return p.sharpen()
        elif p.get_name_simple() in flattened:
            return p.flatten()
        else:
            return p
    def P(name, startTime, duration):
        return Note(convert(Pitch.from_name(name)), startTime, duration)

    F = fractions.Fraction
    D = TimeDelta

    sections = [
        Section([
            P('F5', F(0), D(1./4).dotted()),
            P('D4', F(1./8), D(1./8)),
            P('A4', F(1./4), D(1./4)),
            P('F5', F(3./8), D(1./8)),

            P('E5', F(4./8), D(1./4).dotted()),
            P('A3', F(5./8), D(1./8)),
            P('G4', F(6./8), D(1./4)),
            P('E5', F(7./8), D(1./8)),
        ]),
        # Section([
        #     P('C4', F(0), D(1./4)),
        #     P('D4', F(1./4), D(1./4)),
        #     P('E4', F(2./4), D(1./4)),
        #     P('F4', F(3./4), D(1./4)),
        # ]),
        # Section([
        #     P('G4', F(0./4), D(1./4)),
        #     P('A4', F(1./4), D(1./4)),
        #     P('B4', F(2./4), D(1./4)),
        #     P('C5', F(3./4), D(1./4)),
        # ])
    ]
    gen_tikz(sections)

sample1()


