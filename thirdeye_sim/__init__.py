"""ThirdEye's Gurgaon driving simulation: a 2D top-down world built from our road map and measured traffic.

World holds the map, our car, signals and the clock; play.py drives it from the keyboard; env.py wraps it
in the Gymnasium interface so driving software can be trained and tested in it without a window.
"""
from .roadmap import RoadMap
from .world import World
