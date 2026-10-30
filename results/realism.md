# Realism check (48 drives, 3.01 h of driving)

12 of 12 measured events happen in the simulator at a rate inside the range our footage gives (6 of 12 with the event generator off). Per hour of driving:

| Event | Real roads | Range | Simulator | Traffic alone | Match |
|---|---|---|---|---|---|
| Two-wheeler squeezing past | 126 | 69 to 211 | 76 | 23 | yes |
| Overtaken on the right | 72 | 31 to 142 | 128 | 116 | yes |
| Overtaken on the left | 45 | 15 to 105 | 32 | 34 | yes |
| Pedestrian crossing in front | 36 | 10 to 92 | 17 | 6 | yes |
| Vehicle coming the wrong way | 27 | 6 to 79 | 14 | 0 | yes |
| Cut-in from the left | 18 | 2 to 65 | 7 | 10 | yes |
| Cut-in from the right | 9 | 0 to 50 | 8 | 9 | yes |
| Car ahead braking hard | 18 | 2 to 65 | 30 | 46 | yes |
| Slow vehicle ahead | 18 | 2 to 65 | 5 | 0 | yes |
| Vehicle pulling out | 18 | 2 to 65 | 14 | 0 | yes |
| Pedestrian walking in the road | 18 | 2 to 65 | 7 | 0 | yes |
| Red light jumped | 0 | 0 to 33 | 0 | 0 | yes |

Assumed rates (no measurement yet): cow standing in the road 5, dog running across 4, u-turn across the road 5

| Speeds | Median km/h | Mean km/h | Stopped |
|---|---|---|---|
| Our dashcam car, play area | 20.3 | 23.2 | 14% |
| Our dashcam car, Delhi NCR | 20.3 | 24.1 | 16% |
| Cars going our way, simulator | 18.4 | 19.3 | 34% |
| Our car in the simulator (IDM planner) | 12.9 | 15.7 | 30% |
