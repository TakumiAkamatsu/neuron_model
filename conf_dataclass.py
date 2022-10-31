from dataclasses import dataclass

@dataclass
class Neuron:
    I: float
    tau: float 
    a: float
    b: float
    threshold: float
    reset: float 
    period: float
    step: int
    v_init: float 
    w_init: float

@dataclass
class Solver:
    period: float
    step: float
    v_init: float 
    w_init: float

@dataclass
class FICurve:
    I_min: float
    I_max: float
    delta: float
    fire_threshold: float

@dataclass
class AllConfig:
    neuron: Neuron
    solver: Solver
    fi_curve: FICurve