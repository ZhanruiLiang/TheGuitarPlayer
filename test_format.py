import format
from format import Pitch

names = [format.comb_name(x, y) for y in range(9) for x in format.Temperament]
freqs = [Pitch.name_to_freq(name) for name in names]
# print '\n'.join(map(str, zip(names, freqs)))
names1 = [Pitch.freq_to_name(freq) for freq in freqs]
assert names == names1
