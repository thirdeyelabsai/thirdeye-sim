"""Horn sounds. Our car (a Creta-sized SUV) honks with sim/sounds/horn.wav when that file is there (drop a
recording of the horn you want), otherwise with a made-up dual-tone disc horn of the kind Hyundai fits: two
buzzy tones about a major third apart (415 and 510 Hz) with the harsh ring of the vibrating disc around 2.4 kHz.
Road users honking at us get a higher, shorter beep, quieter the further away they are."""
import os

import numpy as np
import pygame

RATE = 44100
CUSTOM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sounds", "horn.wav")


def _biquad_peak(x, f0, q, gain_db):
    """A resonant peak at f0 (the disc's ring)."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / RATE
    al = np.sin(w) / (2 * q)
    b0, b1, b2 = 1 + al * a, -2 * np.cos(w), 1 - al * a
    a0, a1, a2 = 1 + al / a, -2 * np.cos(w), 1 - al / a
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, v in enumerate(x):   # (0.5 s of sound once at start-up: plain Python is fast enough)
        out = (b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
        x2, x1, y2, y1 = x1, v, y1, out
        y[i] = out
    return y


def _disc_horn(freqs, seconds, attack=0.015, release=0.06):
    t = np.arange(int(RATE * seconds)) / RATE
    wave = np.zeros_like(t)
    for f in freqs:
        phase = (f * t * (1 + 0.003 * np.sin(2 * np.pi * 5 * t))) % 1.0
        wave += np.where(phase < 0.32, 1.0, -0.45)       # a narrow pulse: the disc slapping its stop
    wave -= wave.mean()
    wave = _biquad_peak(wave, 2400.0, 3.0, 9.0)
    wave = np.tanh(1.6 * wave / (np.abs(wave).max() or 1.0))
    env = np.minimum(1.0, t / attack) * np.minimum(1.0, (seconds - t) / release)
    return (wave * env / (np.abs(wave).max() or 1.0) * 0.4 * 32767).astype(np.int16)


class Sounds:
    def __init__(self):
        self.ok = False
        try:
            pygame.mixer.init(frequency=RATE, size=-16, channels=2, buffer=512)
            ch = pygame.mixer.get_init()[2]
            def make(a):
                return pygame.sndarray.make_sound(np.ascontiguousarray(np.repeat(a[:, None], ch, axis=1)) if ch > 1 else a)
            self.horn = pygame.mixer.Sound(CUSTOM) if os.path.exists(CUSTOM) else make(_disc_horn((415.0, 510.0), 0.5))
            self.beep = make(_disc_horn((560.0, 680.0), 0.3))
            self.ok = True
        except Exception:   # no audio device: play silently
            pass

    def our_horn(self):
        if self.ok:
            self.horn.play()

    def their_horn(self, distance):
        if self.ok:
            ch = self.beep.play()
            if ch is not None:
                ch.set_volume(max(0.05, min(1.0, 12.0 / max(12.0, distance))))
