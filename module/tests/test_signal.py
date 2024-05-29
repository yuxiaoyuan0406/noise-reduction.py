import unittest
from module.signal import *

class TestSquareWave(unittest.TestCase):
    def setUp(self) -> None:
        timeout = .5
        dt = 5e-7
        t = np.array(0, timeout, int(timeout/dt), endpoint=False)
        self.signal = SquareWave(t=t, freq=62.5*1e3, amp=1, label='test square wave')
        # return super().setUp()
