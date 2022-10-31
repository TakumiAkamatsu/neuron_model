from typing import Tuple
from neuron.abs_neuron import AbsNeuron

class FitzHughNagumo(AbsNeuron):
    def __init__(
        self, 
        I: float, 
        tau: float, 
        a: float, 
        b: float,
        period: float,
        step: int,
        v_init: float,
        w_init: float,
    ) -> None:
        super().__init__(I, tau)
        self.a = a
        self.b = b
        self.period = period
        self.step = step
        self.v = [v_init]
        self.w = [w_init]
    
    def work(
        self, 
        v: float, 
        w: float,
    ) -> Tuple[float, float]:
        dv = v - v**3/3 - w + self.I
        dw = (v + self.a - self.b * w) / self.tau
        return dv, dw

    def runge_kutta(
        self, 
    ) -> None:

        dt = self.period/self.step

        for i in range(self.step-1):
            k1_v, k1_w = self.work(
                v = self.v[i], 
                w = self.w[i], 
            )
            k2_v, k2_w = self.work(
                v = self.v[i]+0.5*dt*k1_v, 
                w = self.w[i]+0.5*dt*k1_w
            )
            k3_v, k3_w = self.work(
                v = self.v[i]+0.5*dt*k2_v, 
                w = self.w[i]+0.5*dt*k2_w,
            )
            k4_v, k4_w = self.work(
                v = self.v[i]+dt*k3_v, 
                w = self.w[i]+dt*k3_w, 
            )
            v_next = self.v[i] + dt*(k1_v+2*k2_v+2*k3_v+k4_v)/6
            w_next = self.w[i] + dt*(k1_w+2*k2_w+2*k3_w+k4_w)/6
            self.v.append(v_next)
            self.w.append(w_next)

    def clear(self) -> None:
        del self.v[1:]
        del self.w[1:]