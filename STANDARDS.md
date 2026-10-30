# Standards the simulator follows

Our rule for the open-source release (29 Oct 2026): follow the standards the AV open-source community
already uses, strictly. Anything we add beyond them must be genuinely usable, never a gimmick. The launch shows what is possible and stays honest about limits. This file is
the release gate: nothing ships until its row says done and the check in section 5 passes.

Versions were checked on 29 Oct 2026. Sources are listed at the end.

## 1. What we follow

| Area | Standard | Our choice | Now |
|---|---|---|---|
| Env API | Gymnasium 1.4 | `reset(*, seed, options) -> (obs, info)`, `step -> (obs, reward, terminated, truncated, info)`, `render_mode` fixed at init | Done |
| Seeding | Gymnasium | `super().reset(seed=seed)`, then every episode draws its world from `self.np_random` | Fixed 29 Oct (seed was ignored); identical across processes and hash seeds |
| Checker | Gymnasium `check_env` | Runs in CI on every change | Passes with no warnings for the dict, flat, discrete and privileged settings; CI to do |
| Env ID | Gymnasium registry | `ThirdEye/Gurgaon-v0`, bump `-vN` whenever dynamics, reward or event rates change. No `max_episode_steps`: the env truncates at `max_seconds` itself, as highway-env does with its `duration`, so a `max_seconds` given at `make()` is never cut short | Done 29 Oct |
| Observation | highway-env Kinematics | The N nearest road users within 100 m, features `presence, x, y, vx, vy, cos_h, sin_h` in the ego frame, normalised by highway-env's default ranges, plus `class` as a documented extra feature; ego and route state in their own keys (docs/environment.md) | Done 29 Oct |
| Action | highway-env ContinuousAction and DiscreteMetaAction | `[acceleration, steering]` in [-1, 1]: positive acceleration up to 2.5 m/s² (engine), negative down to 8 m/s² (brakes), steering up to 35°. The discrete set is the standard five (LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER) | Done 29 Oct |
| Info keys | MetaDrive | `crash`, `crash_vehicle`, `crash_human`, `crash_object`, `out_of_road`, `arrive_dest`, `max_step`, `cost`, plus `crash_animal`, `off_route`, `red_light`, `wrong_way`, `over_limit`, `route_completion`, `events`, `rewards` | Done 29 Oct |
| Reward | MetaDrive terms, highway-env `info["rewards"]` | Progress, speed and success terms; crash and off-road penalties; each term reported in `info["rewards"]`, weights in the docs | Done 29 Oct |
| Batch use | Gymnasium vector envs | Headless mode, `gymnasium.make_vec` (sync and async), and a published steps-per-second figure: on an Apple M5, 174 to 271 env steps/s for one env and 556 to 677 for 8 async envs (random actions, 0.1 s per step), about 210 MB per env | Done 29 Oct |
| Multi-agent | PettingZoo Parallel API | Later, optional | Later |
| Planner score | nuPlan closed-loop score (reactive agents), CARLA Leaderboard 2.1 driving score | Both, implemented faithfully and named as "nuPlan-style" and "CARLA LB 2.1-style"; collisions classified by nuPlan's types, only at-fault ones counted; raw infraction counts beside every score; an IDM planner baseline; 100 seeds with 95% CIs (thirdeye_sim/evaluate.py, results/benchmark.md) | Done 29 Oct |
| Traffic realism | Waymo Open Sim Agents Challenge 2025 feature groups | Our own implementation (Waymo's metric code carries a non-commercial licence): speed, acceleration, angular speed, distance to nearest object, collision, TTC, distance to road edge, off-road, red-light violations, compared with held-out tracked footage | Partly, 30 Oct: python -m thirdeye_sim.realism checks event rates (counted the way the clips were labelled) and speeds against our footage. The WOSAC features need metric tracks, which single-camera footage does not give: not claimed |
| Surrogate safety | FHWA SSAM, nuPlan | TTC (report at 1.5 s and 0.95 s), PET (5 s), DRAC (3.35 m/s²), hard braking reported at 3.0 and 4.4 m/s² because no standard threshold exists | To do |
| Event names | ISO 34504 tag vocabulary | Each event type tagged with road user type, longitudinal action, lateral action and role relative to the ego. We say "uses ISO 34504 tags", never "compliant" | To do |
| Event rates | De facto (Waymo safety reports) | Per hour and per 1,000 km, exact Poisson 95% CIs, the exposure behind each number, the labelling method | Per hour done (section 4); per km needs GPS distance |
| Scripted vs emergent | (credibility) | Every event record carries `source: scripted` or `traffic` | Done |
| World frame | ENU (REP 103) | x east, y north in metres: UTM 43N less SUMO's netOffset; heading in radians anticlockwise from east | Done; conversions to SUMO and CARLA to document |
| Ego frame | ISO 8855 | x forward, y left, z up | Done |
| Play area | (our choice) | The map is cut at the box we downloaded from OpenStreetMap (gurgaon/clip_osm.py): roads end at a wall at its edge instead of running on as fragments without the streets around them | Done 29 Oct |
| Grade separation | OpenStreetMap `highway=motorway` (grade separated by definition) | OSM tags almost no bridges in Gurgaon (none on NH-48). A road that crosses a motorway with no junction passes under or over it: no shared stretch, and neither road's traffic stops for the other's. The 3D view still draws both at street level there (a known limit) | Done 29 Oct |
| Routes | (our choice, a feasibility check) | A route never runs onto a lane within 4 m of the wall round the map, nor round a junction corner tighter than a car can turn (4 m; netconvert draws small residential corners at about 2.6 m, and our car turns no tighter than 3.7 m) | Done 29 Oct |
| Lane counts | OpenStreetMap `lanes` tags | Where OSM gives a lane count it is used; secondary and tertiary roads without one (most of them in Gurgaon) are ASSUMED to have two lanes each way (gurgaon/osmGurgaon.typ.xml, netconvert's default is one); residential streets keep one | Done 29 Oct |
| Map | ASAM OpenDRIVE, SUMO | Ship the SUMO network and an OpenDRIVE export (netconvert writes 1.4) with the UTM 43N geoReference; claim only what we test: esmini opens it; CARLA only after a test | Done 30 Oct: exports/gurgaon.xodr (tools/export_opendrive.sh); esmini 3.9.0 loads it and drives a car 200 m along Sunset Boulevard, every sampled point on our own road surface. CARLA not tested |
| Scenarios | ASAM OpenSCENARIO XML 1.3, ScenarioNet | One OpenSCENARIO 1.3 file per event type that plays in esmini (it reads 1.0 to 1.3); ScenarioNet export of real-drive replays and sim rollouts | To do |
| Traffic model | Treiber and Kesting | IDM (v0, T, s0, a, b, δ) and MOBIL (p, Δa_th, b_safe, Δa_bias) parameters published under those symbols in SI units | To do |
| Licences | OSI, ODbL | Code Apache-2.0. Road network ODbL with "© OpenStreetMap contributors" (README, game splash and credits). Event rates and labels CC BY 4.0, like our Hugging Face frames. No non-commercial dependencies (no Waymax, Waymo data or Waymo metric code) | Done 30 Oct: LICENSE, NOTICE (which files are ODbL and CC BY), README, footer of every web page |
| Release | Common practice | CITATION.cff, semantic versions, Zenodo DOI per release, pinned versions | CITATION.cff and version 0.1.0 done; the wheel builds and installs clean (tools/build_package.py); Zenodo DOI at release |

## 2. What we add, and the test each must pass

Each of these goes beyond the standards. Each ships only when its gate holds.

| Addition | Why it is useful | Gate |
|---|---|---|
| Measured Indian event rates, with a director that adds only what traffic misses | No open simulator has Indian exposure rates | The rates are plain data with CIs and method; `events=False` gives the pure traffic model; every record says scripted or emergent; a "generic" preset to compare against |
| Indian lateral behaviour (two-wheelers filtering, riding lane lines, squeezing past) | IDM and MOBIL cannot produce it, and it is most of what makes Indian traffic different | Documented as its own model beside IDM and MOBIL, citing SUMO's sublane model as the nearest standard; passing distances and lateral gaps checked against our tracked footage |
| Horn and dipper, with road users reacting | Indian drivers use them to communicate | Off by default, so the standard action space is unchanged; a documented wrapper turns them on; the reaction chances are marked "assumed" until we measure them |
| Real-drive replay | The standard log-replay mode (nuPlan non-reactive, Waymax log playback) on Indian roads | Both non-reactive and reactive modes; exported as ScenarioNet so MetaDrive users can load it |
| Browser version | Anyone can try it in seconds | Runs the same Python as the package; a test shows the same seed gives the same drive in the browser and in Python; the score it shows is the published metric, not a separate game score |
| Chaos setting (`World(chaos=...)`, the game's , and . keys, `?chaos=` in the browser) | Shows how a planner copes with busier traffic than measured | 1.0 reproduces the measured behaviour exactly (same code path, same drive); other values are a stress setting, never presented as measured, and the screen says so whenever it is not 1.0 (what it scales: the note at the top of actors.py) |
| Indian scenes: speed breakers, potholes, bus stops, autos stopping for passengers, cows, dogs, U-turns | What makes Indian roads hard to drive, beyond the measured events | Every one says where its numbers come from: speed breakers and bus stops from OpenStreetMap where it maps them (`source: "osm"`), else `"assumed"` (roadmap.py); potholes, the pickup rate, the cow, dog and U-turn event rates and how long a cow stands (20 to 60 s) are assumed (`events.ASSUMED`), to be replaced by labelled footage; the methods page lists them; the generic style leaves out the behaviours but keeps the road surface |
| Probability map | Shows what may happen around the car next (reframed 29 Oct, limits stated) | Ships only as the simulator's own event hazard (what the generator will do), as an optional privileged observation labelled as such. Never presented as a prediction of real-world risk: we have no measured scene effects yet |

Banners, rings and tags are game interface only. They are not in the env's default observation.

## 3. Never

1. A custom headline score without raw counts, baselines and CIs.
2. "ISO 26262, SOTIF or ISO 34504 compliant". These standards define vocabulary and processes; they do not certify simulators.
3. Event rates without the exposure, CI and labelling method beside them.
4. Scripted events presented as emergent behaviour.
5. A feature in the browser that the headless package lacks.
6. An assumed number presented as measured.
7. A dependency with a non-commercial licence in the core.
8. A map or screenshot without OpenStreetMap attribution.

## 4. Event rates as they stand

Method: 100 random 4-second clips from our dashcam drives (400 s, 6.67 minutes, day and night), and a person names the
main event in each clip (taxonomy v0.4, `event-rates/labels/truth_v04.json`). A rate is the share of clips whose
main event is X, per hour. A clip with two events counts only its main one, so these are lower bounds.

| Event | Clips | Per hour | 95% CI per hour |
|---|---|---|---|
| Two-wheeler passing close | 14 | 126 | 69 to 211 |
| Overtaken on the right | 8 | 72 | 31 to 142 |
| Overtaken on the left | 5 | 45 | 15 to 105 |
| Pedestrian crossing | 4 | 36 | 10 to 92 |
| Wrong-way road user in our lane | 3 | 27 | 6 to 79 |
| Lead vehicle braking | 2 | 18 | 2 to 65 |
| Slow vehicle ahead | 2 | 18 | 2 to 65 |
| Cut-in from the left | 2 | 18 | 2 to 65 |
| Pulling out | 2 | 18 | 2 to 65 |
| Pedestrian walking in the road | 2 | 18 | 2 to 65 |
| Cut-in from the right | 1 | 9 | 0 to 50 |
| Red-light violation | 0 | 0 | 0 to 33 (the sim uses 6 per hour as a placeholder) |

These intervals are wide. The launch says so plainly and invites people to help us label more.

## 5. Release checks (automated, in CI)

1. `gymnasium.utils.env_checker.check_env` passes.
2. The same seed gives the same drive; different seeds give different drives; unseeded resets differ.
3. The same seed gives the same drive in the browser (Pyodide) and in Python. (Passed by hand on 29 Oct: a 60 s drive hashes identically; to automate.)
4. Steps per second, headless, measured and published.
5. The planner score and the realism report are regenerated from code in the repo.
6. The rate table in section 4 is regenerated from the labels file by a script in the repo.
7. The OpenDRIVE and OpenSCENARIO exports open and play in esmini.
8. The licence files and OSM attribution are present.

## Sources

- Gymnasium env API and registry: gymnasium.farama.org/api/env, gymnasium.farama.org/api/registry
- highway-env observations and actions: highway-env.farama.org
- MetaDrive info and reward: metadrive-simulator.readthedocs.io
- CARLA Leaderboard 2.1 scoring: leaderboard.carla.org/evaluation_v2_1
- nuPlan closed-loop score: github.com/motional/nuplan-devkit (metric aggregator config)
- NAVSIM PDMS and EPDMS: github.com/autonomousvision/navsim (docs/metrics.md)
- Waymo Sim Agents 2025: waymo.com/open/challenges/2025/sim-agents
- OpenSCENARIO XML and OpenDRIVE: asam.net/standards
- esmini: github.com/esmini/esmini
- ScenarioNet format: metadrive-simulator.readthedocs.io/en/latest/scenario_description.html
- SUMO OpenDRIVE export: sumo.dlr.de/docs/Networks/Export.html
- ISO 34504 tags: iso.org (preview on standards.iteh.ai)
- SSAM: fhwa.dot.gov/publications/research/safety/08050
- IDM and MOBIL: traffic-flow-dynamics.org, mtreiber.de/publications/MOBIL_TRB.pdf
- OSM attribution: osmfoundation.org/wiki/Licence/Attribution_Guidelines
