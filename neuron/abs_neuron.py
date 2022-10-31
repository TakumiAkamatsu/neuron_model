import numpy as np

class AbsNeuron:
    def __init__(
        self, 
        I: float, 
        tau: float,
    ) -> None:
        self.I = I
        self.tau = tau
        self.v = []
    
    def work(self):
        raise NotImplementedError

    def runge_kutta(
        self, 
        period: float,
        step: float,
        v_init: float, 
        w_init: float,
    ) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError