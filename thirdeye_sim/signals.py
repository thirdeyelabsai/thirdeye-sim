"""Traffic signals: each controller cycles through its phases (the plans netconvert built for the network).

A phase state has one letter per controlled link: G or g green, y yellow, r red. Controllers start at
staggered points in their cycle, as real signals are not in step with each other, except those of one junction
(the road map's signal_group), which keep one clock.
"""
import random


class Signals:
    def __init__(self, signals, seed=0, group=None):
        rng = random.Random(seed)
        self.items = signals
        self.cycle = [sum(d for d, _ in s["phases"]) or 1 for s in signals]
        self.offset = [rng.uniform(0, c) for c in self.cycle]
        for k, g in enumerate(group or ()):
            self.offset[k] = self.offset[g]

    def state(self, k, t):
        s = self.items[k]
        u = (t + self.offset[k]) % self.cycle[k]
        for d, st in s["phases"]:
            if u < d:
                return st
            u -= d
        return s["phases"][-1][1]

    def until_change(self, k, t, links):
        """Seconds until the light shown to these links changes colour (what a countdown timer shows)."""
        now = self.light(self.state(k, t), links)
        s = self.items[k]
        u = (t + self.offset[k]) % self.cycle[k]
        waited, phases = 0.0, s["phases"]
        i = 0
        while u >= phases[i][0]:   # the phase we are in, and how far into it
            u -= phases[i][0]
            i = (i + 1) % len(phases)
        left = phases[i][0] - u
        for step in range(1, 2 * len(phases) + 1):
            j = (i + step) % len(phases)
            if self.light(phases[j][1], links) != now:
                return left
            left += phases[j][0]
        return left

    @staticmethod
    def light(state, links):
        c = [state[i] for i in links if i < len(state)]
        if any(x in "Gg" for x in c):
            return "green"
        if any(x in "yY" for x in c):
            return "yellow"
        return "red"
