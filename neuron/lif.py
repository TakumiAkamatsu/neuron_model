from neuron.abs_neuron import AbsNeuron

class LIF(AbsNeuron):
    def __init__(
        self, 
        I: float, 
        tau: float, 
        threshold: float,
        reset: float, 
        period: float, 
        step: int, 
        v_init: float,
    ) -> None:
        super().__init__(I, tau)
        self.threshold = threshold
        self.reset = reset
        self.period = period
        self.step = step
        self.v = [v_init]

    def work(
        self, 
        v: float, 
    ) -> float:
        dv = - (v - self.I) / self.tau
        return dv

    def runge_kutta(
        self, 
    ) -> None:

        dt = self.period/self.step

        for i in range(self.step-1):
            k1 = self.work(self.v[i])
            k2 = self.work(self.v[i]+0.5*dt*k1)
            k3 = self.work(self.v[i]+0.5*dt*k2)
            k4 = self.work(self.v[i]+dt*k3)
            v_next = self.v[i] + dt*(k1+2*k2+2*k3+k4)/6
            if v_next >= self.threshold:
                v_next = self.reset
            self.v.append(v_next)

    def clear(self) -> None:
        del self.v[1:]