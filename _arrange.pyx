cdef:
    DEF N_STRINGS = 6
    DEF N_FINGERS = 4
    DEF MAX_STATES = 500

    # Parameters about movement distance.
    float b1 = .04  # Change fret.
    float b2 = .02  # Change string.
    float b3 = 2.   # Change index fret.
    # Parameters about finger change cost.
    float a1 = 0.0  # free
    float a2 = 0.5  # release
    float a3 = 0.2  # keep
    float a4 = 2.  # press
    float a5 = 3.  # jump

    float MIN_RELEASE_TIME = 0.05
    float MIN_JUMP_TIME = 0.1
    float BAR_PENALTY = .2 # FIXME
    float HIGH_FRET_PENALTY = 1.
    DEF HIGH_FRET = 9

FRET_CHANGE_PENALTY = b3
MISS_PENALTY = 1000.

cdef float sqr(float x):
    return x * x

ctypedef struct State:
    int frets[N_FINGERS]
    int strings[N_FINGERS]
    int rings[N_STRINGS]
    bint bar
    # Variables calculated by `match`.
    int missed
    int indexMatched[N_STRINGS + 1]
    int matched[N_FINGERS]  # Matched indices of the frame

cdef match_frame(State *s, fretboard, frame):
    cdef:
        int i, f, r, nIndexMatched, pitch
        dict pitchToEvent

    for i in range(N_FINGERS):
        s.matched[i] = -1
    pitchToEvent = {e.pitch: i for i, e in enumerate(frame)}
    for i in range(N_FINGERS):
        if s.bar and i == 0:
            continue
        if is_pressed(s, i):
            pitch = fretboard.basePitches[s.strings[i]] + s.frets[i]
            s.matched[i] = pitchToEvent.pop(pitch, -1)
    if s.bar:
        nIndexMatched = 0
        f = s.frets[0]
        for r in range(s.strings[0] + 1):
            pitch = fretboard.basePitches[r] + r
            if pitch in pitchToEvent:
                s.indexMatched[nIndexMatched] = pitchToEvent.pop(pitch)
                nIndexMatched += 1
        s.indexMatched[nIndexMatched] = -1
    for i in range(N_STRINGS):
        if s.rings[i]:
            pitchToEvent.pop(fretboard.basePitches[i], None)
    s.missed = len(pitchToEvent)

cdef float get_end_time(State *s, list frame, int finger):
    if not s.bar or finger > 0:
        assert s.matched[finger] >= 0
        return float(frame[s.matched[finger]].end)
    cdef float end = -1
    i = 0
    while i < N_STRINGS and s.indexMatched[i] != -1:
        end = max(end, float(frame[s.indexMatched[i]].end))
        i += 1
    return end

cdef is_pressed(State *s, int finger):
    return s.strings[finger] != -1

cdef class CostCalculator:
    cdef:
        State states[2][MAX_STATES]
        int nStates[2]
        object fretboard, times, frames

    def __init__(self, fretboard):
        self.fretboard = fretboard
        self.nStates[0] = 0
        self.nStates[1] = 0
        self.times = [0, 0]
        self.frames = [None, None]

    def add_frame(self, list frame, list states, time):
        cdef:
            int i, j, n
            State *s
        self.nStates[0] = self.nStates[1]
        for i in range(self.nStates[1]):
            self.states[0][i] = self.states[1][i]
        self.times[0] = self.times[1]
        self.times[1] = time
        self.frames[0] = self.frames[1]
        self.frames[1] = frame
        n = self.nStates[1] = len(states)
        for i in range(n):
            state = states[i]
            s = &self.states[1][i]
            for j in range(N_FINGERS):
                s.frets[j] = state.frets[j]
                s.strings[j] = state.strings[j]
            for j in range(N_STRINGS):
                s.rings[j] = state.rings[j]
            s.bar = state.bar
            match_frame(s, self.fretboard, frame)

    def get_cost(self, int stateIdx1, int stateIdx2, float minCost):
        cdef:
            float dt, totalCost, end1, keepTime, releaseTime, jumpTime
            int i
            State *s1, *s2

        assert stateIdx1 < self.nStates[0]
        assert stateIdx2 < self.nStates[1]
        s1 = &self.states[0][stateIdx1]
        s2 = &self.states[1][stateIdx2]
        t1 = self.times[0]
        t2 = self.times[1]
        dt = float(t2 - t1)
        totalCost = 0
        frame1 = self.frames[0]
        frame2 = self.frames[1]

        totalCost += MISS_PENALTY * s2.missed
        # Index fret change cost
        totalCost += b3 * sqr(s1.frets[0] - s2.frets[0]) / dt
        if s2.bar:
            totalCost += BAR_PENALTY * s2.strings[0]
        for i in range(N_FINGERS):
            if totalCost >= minCost:
                return totalCost
            if is_pressed(s1, i) and is_pressed(s2, i):
                if s1.strings[i] == s2.strings[i] and s1.frets[i] == s2.frets[i]:
                    # Keep
                    totalCost += dt * a3
                else:
                    # Jump
                    totalCost += (
                        b1 * sqr((s1.frets[i] - s1.frets[0]) - (s2.frets[i] - s2.frets[0]))
                        + b2 * sqr(s1.strings[i] - s2.strings[i])) / dt
                    end1 = get_end_time(s1, frame1, i)
                    keepTime = (end1 - t1)
                    jumpTime = max(MIN_JUMP_TIME, float(t2 - end1))
                    totalCost += keepTime * a3 + a5 / jumpTime
            elif not is_pressed(s1, i) and not is_pressed(s2, i):
                # Free
                totalCost += a1
            elif is_pressed(s1, i) and not is_pressed(s2, i):
                # Release
                end1 = get_end_time(s1, frame1, i)
                keepTime = (end1 - t1)
                releaseTime = max(MIN_RELEASE_TIME, float(t2 - end1))
                totalCost += keepTime * a3 + a2 / releaseTime
            else:
                # Press
                totalCost += a4 / dt

            if is_pressed(s2, i) and s2.frets[i] > HIGH_FRET:
                totalCost += (s2.frets[i] - HIGH_FRET) * HIGH_FRET_PENALTY

        # TODO: bar and unbar
        return totalCost
