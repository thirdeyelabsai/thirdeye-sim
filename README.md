# Gurgaon driving simulator

Drive Gurgaon's real roads in traffic that behaves like India's. Two-wheelers squeeze past, autos cut in,
people cross anywhere and cows stand in the lane. How often each of these happens comes from dashcam footage of
Delhi NCR that the Third Eye Labs team labelled by hand.

- **Play it in your browser:** https://sim.thirdeyelabs.ai (nothing to install)
- **Use it in Python:** a [Gymnasium](https://gymnasium.farama.org) environment for training and testing driving
  software, with planner scores in the style of nuPlan and CARLA.

## Install

```
pip install "thirdeyelabs[sim]"          # the environment (installs thirdeye-sim)
pip install "thirdeye-sim[render]"       # plus the 3D view and the desktop game
```

Python 3.10 or later. `thirdeyelabs` is our dataset package; its `sim` extra installs this simulator, which is
also on PyPI on its own as `thirdeye-sim`.

## Quick start

```python
import gymnasium as gym
import thirdeye_sim.env                  # registers ThirdEye/Gurgaon-v0

env = gym.make("ThirdEye/Gurgaon-v0")
obs, info = env.reset(seed=0)
while True:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
```

The observation, actions, reward and options are in [docs/environment.md](docs/environment.md).

## Score a planner

```
python -m thirdeye_sim.evaluate --episodes 20
```

Drives each planner in `thirdeye_sim/planners.py` over the same seeded routes in Gurgaon traffic and in
orderly traffic, and reports nuPlan-style and CARLA-style scores beside the raw counts. Add your own planner
to `PLANNERS` to score it.

## Check the traffic against real roads

```
python -m thirdeye_sim.realism
```

Compares how often each kind of event happens in the simulator with how often it happens in our footage, and
how fast traffic moves. The results and what they do not cover yet are at https://sim.thirdeyelabs.ai/realism.

## What is measured and what is assumed

Event rates come from labelled footage, with their uncertainty. Some things are assumed until we can measure
them, and the code marks them as assumed: lane counts on roads OpenStreetMap leaves untagged, most speed
breakers, potholes, and the rates of the cow, dog and U-turn events. [STANDARDS.md](STANDARDS.md) lists every
standard we follow and every addition we make.

## Licences

- Code: Apache-2.0 ([LICENSE](LICENSE))
- Map data: © OpenStreetMap contributors, under the ODbL ([NOTICE](NOTICE))
- Event rates and measurements: CC BY 4.0

## Citation

See [CITATION.cff](CITATION.cff). Built by [Third Eye Labs](https://thirdeyelabs.ai).
