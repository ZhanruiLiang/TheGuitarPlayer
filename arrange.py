import sys
from fractions import Fraction
from itertools import combinations
from collections import namedtuple
import pysheetmusic
from raygllib.utils import timeit
from raygllib import ui
import pyximport
pyximport.install()
from _arrange import CostCalculator
from _arrange import FRET_CHANGE_PENALTY, MISS_PENALTY

NoteEvent = namedtuple('NoteEvent', 'start end pitch effect note')

N_STRINGS = 6
N_FINGERS = 4
# FINGER_DISTANCES = [.06, .04, .05]
FINGER_DISTANCES = [.06, .03, .04]
TOTAL_FINGER_DISTANCE = sum(FINGER_DISTANCES)


class NoteEffect:
    REGULAR = 'regular'
    SLIDE = 'slide'
    HIT = 'hit'

def pitch_to_level(pitch):
    return 'C D EF G A B'.index(pitch.step) + int(pitch.alter) + pitch.octave * 12

def pitch_to_level2(step, octave):
    return 'C D EF G A B'.index(step) + octave * 12


class PlayState:
    __slots__ = ['bar', 'frets', 'strings', 'rings']

    def __init__(self):
        self.bar = False
        self.frets = [0] * 4
        self.strings = [-1] * 4
        self.rings = [False] * N_STRINGS

    def froze(self):
        self.frets = tuple(self.frets)
        self.strings = tuple(self.strings)
        self.rings = tuple(self.rings)

    def match(self, fretboard, frame):
        matched = [None] * N_FINGERS
        pitchToEvent = {e.pitch: e for e in frame}
        for i in range(N_FINGERS):
            if self.bar and i == 0:
                continue
            if self.strings[i] != -1:
                pitch = fretboard.basePitches[self.strings[i]] + self.frets[i]
                matched[i] = ((self.frets[i], self.strings[i]), pitchToEvent.pop(pitch, None))
        if self.bar:
            matched[0] = []
            f = self.frets[0]
            for r in range(self.strings[0] + 1):
                try:
                    matched[0].append(((f, r), pitchToEvent.pop(fretboard.basePitches[r] + f)))
                except KeyError:
                    pass
        empty = []
        for r in range(N_STRINGS):
            if self.rings[r]:
                pitch = fretboard.basePitches[r]
                try:
                    empty.append((r, pitchToEvent.pop(pitch)))
                except KeyError:
                    pass
        missed = len(pitchToEvent)
        return matched, empty, missed

    def dump(self):
        try:
            s = '|---' * max(f for f, r in zip(self.frets, self.strings) if r >= 0)
        except ValueError:
            s = '|---'
        s = '-' + s
        t = [list(s) for i in range(N_STRINGS)]
        for i in range(N_FINGERS):
            r = self.strings[i] 
            if r >= 0:
                f = self.frets[i]
                t[r][4 * f - 1] = str(i + 1)
        if self.bar:
            i = 0
            f = self.frets[i]
            for r in range(self.strings[i]):
                t[r][4 * f - 1] = str(i + 1)
        for r in range(N_STRINGS):
            if self.rings[r]:
                t[r][0] = 'o'
        return str((self.frets, self.strings, self.rings)) + '\n' +\
            '\n'.join(map(''.join, t))

    def __hash__(self):
        return hash((self.bar, self.frets, self.strings, self.rings))


def get_note_events(sheet):
    noteEvents = []
    for timeStart, timeEnd, note in sheet.iter_note_sequence():
        if note.duration > 0:
            noteEvents.append(NoteEvent(
                timeStart, timeEnd, pitch_to_level(note.pitch), NoteEffect.REGULAR, note))
    noteEvents.sort(key=lambda x: x[:2])
    return noteEvents


def get_time_points(noteEvents):
    timePoints = []
    for event in noteEvents:
        if not timePoints or timePoints[-1] != event.start:
            timePoints.append(event.start)
    return timePoints


def get_frames(timePoints, noteEvents):
    """
    frames = {
        time0: [events at time0],
        time1: [events at time1],
        ...
    }
    """
    frames = {}
    activeEvents = []
    currentEventIdx = 0
    for time in timePoints:
        activeEvents = [event for event in activeEvents if event.end > time]
        while currentEventIdx < len(noteEvents) \
                and noteEvents[currentEventIdx].start == time:
            event = noteEvents[currentEventIdx]
            activeEvents.append(event)
            currentEventIdx += 1
        frame = {}
        for event in activeEvents:
            if event.pitch not in frame:
                frame[event.pitch] = event
            else:
                if event.end > frame[event.pitch].end:
                    frame[event.pitch] = event
        frames[time] = list(frame.values())

    return frames 

class StateCalculator:
    rates = []

    def __init__(self, fretboard):
        self.fretboard = fretboard
        self._cache = {}
        self._callCount = 0
        self._cacheHit = 0

    def report_stats(self):
        print('Total calls: {}'.format(self._callCount))
        print('Cache hit rate: {:.2f}'.format(self._cacheHit / self._callCount))

    # @profile
    def get_matched_states(self, frame):
        self._callCount += 1
        cacheKey = frozenset(e.pitch for e in frame)
        if cacheKey in self._cache:
            self._cacheHit += 1
            return self._cache[cacheKey].copy()

        collected = []
        iterCount = 0

        get_fret_distance = self.fretboard.get_fret_distance
        get_positions = self.fretboard.get_positions

        # @profile
        def collect(eventIdx):
            nonlocal iterCount
            while eventIdx < len(frame) and preHandled[eventIdx]:
                eventIdx += 1
            if eventIdx == len(frame):
                items = [
                    (f, -r) for i, (f, r) in enumerate(positions)
                    if f > 0 and not preHandled[i]
                ]
                items.sort()
                for fingers in combinations(range(1, N_FINGERS), len(items)):
                    state = PlayState()
                    lastFinger = 0
                    state.frets[0] = indexFret
                    if indexString is not None:
                        state.strings[0] = indexString
                    for finger, (f, r) in zip(fingers, items):
                        state.frets[finger] = f
                        state.strings[finger] = -r
                        if lastFinger is not None:
                            if state.frets[finger] == state.frets[lastFinger]\
                                    and state.strings[finger] > state.strings[lastFinger]:
                                break
                            maxDistance = sum(
                                FINGER_DISTANCES[i] for i in range(lastFinger, finger))
                            distance = get_fret_distance(f, state.frets[lastFinger])
                            if distance > maxDistance:
                                break
                        lastFinger = finger
                    else:
                        state.rings[:] = stringUsed
                        state.bar = bar
                        state.froze()
                        collected.append(state)
                    iterCount += 1
                return
            event = frame[eventIdx]
            for (f, r) in get_positions(event.pitch):
                if not stringUsed[r] and (f == 0 or f >= indexFret + bar and 
                        get_fret_distance(f, indexFret) <= TOTAL_FINGER_DISTANCE):
                    # Assign this position to current event
                    stringUsed[r] = True
                    positions[eventIdx] = (f, r)
                    collect(eventIdx + 1)
                    stringUsed[r] = False

        dropCount = 0
        frame0 = frame
        basePitches = self.fretboard.basePitches
        while not collected and dropCount < len(frame0):
            for frame in combinations(frame, len(frame0) - dropCount):
                stringUsed = [False] * N_STRINGS
                preHandled = [False] * len(frame)
                positions = [None] * len(frame)
                bar = False
                indexFret = 0
                indexString = None
                for indexFret in range(1, self.fretboard.maxFret):
                    bar = False
                    # Use index finger, choose a event for it.
                    for i, event in enumerate(frame):
                        preHandled[i] = True
                        for r in range(N_STRINGS):
                            if basePitches[r] == event.pitch - indexFret:
                                stringUsed[r] = True
                                positions[i] = (indexFret, r)
                                indexString = r
                                collect(0)
                                stringUsed[r] = False
                        preHandled[i] = False
                    # Do not use index finger
                    indexString = None
                    collect(0)
                    # Bar
                    if len(frame) > 2:
                        bar = True
                        for r in range(1, N_STRINGS):
                            # Bar strings (0, 1, ..., r) using index finger
                            # Select all barred pitches
                            indexString = r
                            barred = {basePitches[r1] + indexFret: r1 for r1 in range(r + 1)}
                            for i, event in enumerate(frame):
                                if event.pitch in barred:
                                    preHandled[i] = True
                                    positions[i] = (indexFret, barred[event.pitch])
                            stringUsed[r] = True
                            collect(0)
                            preHandled = [False] * len(frame)
                        stringUsed = [False] * N_STRINGS
            dropCount += 1
        dropCount -= 1
        if dropCount == 0:
            self._cache[cacheKey] = collected.copy()
        if dropCount:
            print('drop', dropCount)

        # if iterCount > 0:
        #     self.rates.append(len(collected) / iterCount)
        return collected


class FingeringArranger:
    def __init__(self, sheet):
        self.noteEvents = noteEvents = get_note_events(sheet)
        self.timePoints = timePoints = get_time_points(noteEvents)
        self.frames = get_frames(timePoints, noteEvents)
        self._make_fretboard()
        self.stateCalc = StateCalculator(self.fretboard)
        self._arrange()
        # self._dump_frames(timePoints, self.frames)
        # self.stateCalc.report_stats()

    # @profile
    # @timeit
    def _arrange(self):
        fb = self.fretboard
        timePoints = self.timePoints
        nTimePoints = len(timePoints)
        frames = self.frames
        # costs[i, j] = min cost of state statess[i][j] at time[i]
        costss = []
        choicess = []
        # statess[i] = states of frames[timePoints[i]]
        statess = [
            self.stateCalc.get_matched_states(frames[timePoints[i]])
            for i in range(nTimePoints)
        ]
        costCalc = CostCalculator(fb)
        i = 0
        costss.append([0] * len(statess[0]))
        for j, state in enumerate(statess[0]):
            matched, empty, missed = state.match(fb, frames[timePoints[0]])
            costss[0][j] += missed * MISS_PENALTY
        choicess.append([None] * len(statess[0]))
        costCalc.add_frame(frames[timePoints[0]], statess[0], timePoints[0])

        for i in range(1, nTimePoints):
            print(i, nTimePoints)
            states1 = statess[i - 1]
            states2 = statess[i]
            costs = [None] * len(states2)
            costss.append(costs)
            choices = [None] * len(states2)
            choicess.append(choices)
            t1, t2 = timePoints[i - 1], timePoints[i]
            dt = float(t2 - t1)
            costCalc.add_frame(frames[t2], states2, t2)
            # print('  ', len(states2) * len(states1))
            tot = len(states2) * len(states1)
            cnt = 0
            j1Order = list(range(len(states1)))

            for j2, state2 in enumerate(states2):
                minCost = 1e20
                choice = None
                fret0 = state2.frets[0]
                j1Order.sort(key=lambda j: abs(states1[j].frets[0] - fret0))
                for j1 in j1Order:
                    state1 = states1[j1]
                    lastCost = costss[i - 1][j1]
                    # if costss[i - 1][j1] >= minCost:
                    #     break
                    tmp = (state1.frets[0] - state2.frets[0])
                    if lastCost + tmp * tmp * FRET_CHANGE_PENALTY / dt >= minCost:
                        break
                    cnt += 1
                    cost = lastCost + costCalc.get_cost(j1, j2, minCost)
                    if minCost > cost:
                        minCost = cost
                        choice = j1
                assert choice is not None
                costs[j2] = minCost
                choices[j2] = choice
            print(cnt / tot, cnt, tot)

        j = min(range(len(statess[-1])), key=lambda j:costss[-1][j])
        i = nTimePoints - 1
        print('minCost', costss[-1][j])
        while i >= 0:
            state = statess[i][j]
            t = timePoints[i]
            matched, empty, missed = state.match(fb, frames[t])
            # print('=======', i, t)
            # print('matched', matched)
            # print('empty', empty)
            # print('missed', missed)
            # print(state.dump())
            if missed:
                print('missed', missed, 'at', t, i)
            for finger in range(N_FINGERS):
                if not matched[finger]:
                    continue
                if finger == 0 and state.bar:
                    events = matched[finger]
                else:
                    events = [matched[finger]]
                for (f, r), event in events:
                    if event.start == t:
                        fingering = event.note.fingering
                        fingering.fret = f
                        fingering.string = r + 1
                        fingering.finger = finger + 1
                        # print(fingering.string, fingering.fret, event)
            for r, event in empty:
                if event.start == t:
                    fingering = event.note.fingering
                    fingering.fret = 0
                    fingering.string = r + 1
                    # print(fingering.string, fingering.fret, event.note)
            j = choicess[i][j]
            i -= 1

    def _make_fretboard(self):
        fretboard = Fretboard()
        for event in self.noteEvents:
            if fretboard.minPitch > event.pitch:
                fretboard = Fretboard(tuning='drop-d')
                break
        self.fretboard = fretboard

    @timeit
    def _dump_frames(self, timePoints, frames):
        # import matplotlib.pyplot as plt
        stateLens = []
        for time in timePoints[:]:
            frame = frames[time]
            states = self.stateCalc.get_matched_states(frame)
            stateLens.append(len(states))
            if len(states) >= 100:
                print('>>>>>>>>>>>>>>>>>>>')
                print(time, float(time), frame[0].note.measure)
                print('len(frame)', len(frame), 'len(states)', len(states))
                for event in frame:
                    pitch = event.pitch
                    step = (
                        'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
                    )[pitch % 12]
                    octave = pitch // 12
                    print('   ', step, octave, event)
                for state in states:
                    print('========================')
                    print(state.dump())

        # plt.hist(stateLens)
        # plt.show()
        print('max len states', max(stateLens))


def dump_frame(frame):
    print(frame[0].note.measure)
    for event in frame:
        pitch = event.pitch
        step = (
            'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
        )[pitch % 12]
        octave = pitch // 12
        print('   ', step, octave, event)

class Fretboard:
    TUNING = {
        'standard': [('E', 4), ('B', 3), ('G', 3), ('D', 3), ('A', 2), ('E', 2)],
        'drop-d': [('E', 4), ('B', 3), ('G', 3), ('D', 3), ('A', 2), ('D', 2)],
    }
    # Fretboard dimensions.
    TOP = .08
    BOTTOM = .10
    STRING_LENGTH = .90

    def __init__(self, maxFret=18, tuning='standard'):
        self.basePitches = [pitch_to_level2(*args) for args in self.TUNING[tuning]]
        self.minPitch = minPitch = self.basePitches[-1]
        self.maxPitch = maxPitch = self.basePitches[0] + maxFret
        self.maxFret = maxFret
        positions = self._positions = [[] for _ in range(maxPitch - minPitch + 1)]
        nStrings = len(self.basePitches)
        for r in range(nStrings):
            for f in range(maxFret + 1):
                pitch = self.basePitches[r] + f
                positions[pitch - minPitch].append((f, r))
        for ps in positions:
            ps.sort()
        self._make_bars()
        # print(self.basePitches)
        # print(self._fretPos, self.get_fret_distance(3, 7))

    def _make_bars(self):
        a = 2 ** (1. / 12)
        xs = [0] * (self.maxFret + 1)
        for i in range(1, len(xs)):
            xs[i] = ((a - 1) * self.STRING_LENGTH + xs[i - 1]) / a
        self._barPos = xs
        b = .8
        self._fretPos = [0] + \
            [xs[i] * (1 - b) + xs[i + 1] * b for i in range(len(xs) - 1)]

    def get_fret_distance(self, f1, f2):
        return abs(self._fretPos[f1] - self._fretPos[f2])

    def get_positions(self, pitch, minFret=0):
        " return: A list of (fret, string) tuples. "
        assert pitch >= self.minPitch
        ps = self._positions[pitch - self.minPitch]
        for i in range(len(ps)):
            if ps[i][0] >= minFret:
                return ps[i:]
        return []


class Window(ui.Window):
    def __init__(self):
        self.viewer = pysheetmusic.viewer.SheetViewer()
        super().__init__(resizable=True)
        self.root.children.append(self.viewer)
        K = ui.key
        self.player = player = pysheetmusic.player.Player()
        self.viewer.set_player(player)

        self.add_shortcut(K.chain(K.Q), sys.exit)
        self.add_shortcut(K.chain(K.SHIFT, K.P), lambda: player.pause() or True)
        self.add_shortcut(K.chain(K.P), lambda: player.play() or True)

    def on_close(self):
        self.player.stop()
        super().on_close()

    def set_sheet(self, sheet):
        layout = pysheetmusic.layout.LinearTabLayout(sheet)
        layout.layout()
        self.viewer.set_sheet_layout(layout)
        self.player.set_sheet(sheet)
        self.player.play()

    def update(self, dt):
        super().update(dt)
        self.viewer.update(dt)


SHEETS = [
    # 'Debug.mxl',
    'Allegro_by_Bernardo_Palma_V.mxl',
    'Chord_test.mxl',
    'Untitled_in_D_Major.mxl',
    'Divertimento_No._1.mxl',
    'Giuliani_-_Op.50_No.1.mxl',
    'Chrono_Cross_-_Quitting_the_Body.mxl',
    'Unter_dem_Lindenbaum.mxl',
    'Lagrima.mxl',
    'Guitar_Solo_No._116_in_A_Major.mxl',
    'Almain.mxl',
    'Somewhere_In_My_Memory.mxl',
    'Tango_Guitar_Solo_2.mxl',
    'Air.mxl',
    'Allegretto_in_C_Major_for_Guitar_by_Carcassi_-_arr._by_Gerry_Busch.mxl',
    'Fernando_Sor_Op.32_Mazurka.mxl',
    'Fernando_Sor_Op.32_Galop.mxl',
    'Guitar_Solo_No._117_in_E_Minor.mxl',
    'Chrono_Cross_-_Frozen_Flame.mxl',
    'Fernando_Sor_Op.32_Andante_Pastorale.mxl',
    'Fernando_Sor_Op.32_Andantino.mxl',
    'Guitar_Solo_No._118_-_Barcarolle_in_A_Minor.mxl',
    'Guitar_Solo_No._119_in_G_Major.mxl',
    'Guitar_Solo_No._15_in_E_Major.mxl',
    'Maria_Luisa_Mazurka_guitar_solo_the_original_composition.mxl',
    'Minuet_in_G_minor.mxl',
    'Pavane_No._6_for_Guitar_Luis_Milan.mxl',
    'People_Imprisoned_by_Destiny.mxl',

    'Jeux_interdits.mxl',
    'K27_Domenico_Scarlatti.mxl',
    'Auld_Lang_Syne_guitar.mxl',
    'We_wish_you_a_Merry_Christmas.mxl',
    'Lute_Suite_No._1_in_E_Major_BWV_1006a_J.S._Bach.mxl',
]

if __name__ == '__main__':
    import crash_on_ipy
    parser = pysheetmusic.parse.MusicXMLParser()
    BASE = '/home/ray/python/pysheetmusic/tests/sheets/'
    for name in SHEETS[:]:
        StateCalculator.rates.clear()
        sheet = parser.parse(BASE + name)
        pysheetmusic.tab.attach_tab(sheet)
        pysheetmusic.tab.attach_fingerings(sheet)
        arranger = FingeringArranger(sheet)

        window = Window()
        window.set_sheet(sheet)
        window.start()

        # import matplotlib.pyplot as plt
        # plt.hist(StateCalculator.rates, bins=20); plt.show()
