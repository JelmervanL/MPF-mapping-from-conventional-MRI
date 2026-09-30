"""Train a q-map model.

    python train.py                                   # full model of the paper
    python train.py experiment=no_adain               # any config in configs/experiment/
    python train.py experiment=domain_shift/mprage_only train.epochs=2 data.max_sessions.train=20
"""
import hydra

from qmap.training import train


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg):
    train(cfg)


if __name__ == '__main__':
    main()
