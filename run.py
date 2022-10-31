import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
import mlflow
from conf_dataclass import AllConfig
from fi_curve import fi_curve
from neuron.abs_neuron import AbsNeuron

cs = ConfigStore.instance()
cs.store(name="all_config", node=AllConfig)

@hydra.main(config_path='conf/', config_name="config.yaml")
def main(cfg: AllConfig) -> None:
    neuron: AbsNeuron = instantiate(cfg.neuron)
    with mlflow.start_run():

        neuron.runge_kutta()
        
        fig = plt.figure()
        plt.plot(
            np.arange(len(neuron.v))*cfg.neuron.period/cfg.neuron.step, 
            neuron.v,
        )
        plt.xlabel("t")
        plt.ylabel("v")
        mlflow.log_figure(fig, "vt.png")
        fig.clear()
        neuron.clear()

        # f-I curve
        I_min = cfg.fi_curve.I_min
        I_max = cfg.fi_curve.I_max
        delta = cfg.fi_curve.delta
        fire_threshold = cfg.fi_curve.fire_threshold
        fire_list = []
        for I in np.arange(I_min, I_max, delta):
            neuron.I = I
            neuron.runge_kutta()
            n_fire = fi_curve(
                v_list=neuron.v, 
                threshold=fire_threshold,
            )
            fire_list.append(n_fire)
            neuron.clear()
        
        fig = plt.figure()
        plt.plot(
            np.arange(I_min, I_max, delta), 
            fire_list,
        )
        plt.xlabel("I")
        plt.ylabel("f")
        mlflow.log_figure(fig, "fi.png")

if __name__ == "__main__":
    main()