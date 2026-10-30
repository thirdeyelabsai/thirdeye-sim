# Planner benchmark (100 seeded routes of 800 m per cell)

| Planner / traffic | nuPlan-style CLS | CARLA-style DS | routes finished | collisions per km | red lights per km | hard brakes per km (3.0 / 4.4 m/s²) | mean speed |
|---|---|---|---|---|---|---|---|
| idm / generic | 0.73 ± 0.06 | 83.2 ± 6.3 | 78 of 100 | 0.18 | 0.00 | 3.7 / 2.0 | 19 km/h |
| idm / gurgaon | 0.62 ± 0.07 | 72.8 ± 7.3 | 64 of 100 | 0.39 | 0.00 | 4.7 / 3.2 | 17 km/h |
| lane_keeper / generic | 0.13 ± 0.06 | 29.8 ± 5.9 | 14 of 100 | 2.31 | 0.16 | 0.0 / 0.0 | 27 km/h |
| lane_keeper / gurgaon | 0.05 ± 0.04 | 23.1 ± 4.6 | 7 of 100 | 2.97 | 0.16 | 0.0 / 0.0 | 27 km/h |

± is the half-width of a 95% confidence interval over the routes. How episodes ended:

- idm / generic: 78 arrive dest, 14 crash, 8 max step
- idm / gurgaon: 64 arrive dest, 27 crash, 9 max step
- lane_keeper / generic: 14 arrive dest, 84 crash, 2 out of road
- lane_keeper / gurgaon: 6 arrive dest, 93 crash, 1 out of road

Scores are our implementations of the nuPlan closed-loop score and the CARLA Leaderboard 2.1 driving score (thirdeye_sim/evaluate.py says where they differ); the reference progress is the IDM planner's.
