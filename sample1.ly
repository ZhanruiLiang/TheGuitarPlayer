\version "2.16.2"

% title: 练习曲第6号 索尔 P97

song = {
  \clef treble
  \key d \major
  \time 4/4
  \tempo "Allegro"

  \partial 8 a'8 |
  <<
    \new Voice \relative d'' {
      \voiceOne
       fis4. fis8 e4. e8 | % sec 1
       d4. fis8 a,4. a8 |
       b4. b8 e4. d8|
       d4. cis8  e4 r8 a,8 |
       fis'4. fis8 e4. e8 | % sec 5
       d4. d8 dis4. dis8 |
       e4. b8 cis4. e8 |
       <fis, d'>4 r4 r4 r4 |

    } 
    \new Voice \relative d' {
      \voiceTwo
      r8 d8 a'4 r8 a,8 g'4 | % sec 1
      r8 d8 fis4 r8 d8 fis4 |
      r8 g,8 g'4 r8 gis,8 e'4 |
      r8 a,8 e'4 r8 a,8 e'4 |
      r8 d8 a'4 r8 a,8 g'4 | % sec 5
      r8 b,8 fis'4 r8 a,8 fis'4 |
      r8 g,8 <e' g>4 r8 g,8 <e' g>4 |
      r8 d8 a'8 fis8  d8 r8 r4 |
    }
  >>

}

\score {
  \new Staff \with {midiInstrument = #"acoustic guitar (nylon)" }
    \song
  \midi {}
  \layout {}
}
