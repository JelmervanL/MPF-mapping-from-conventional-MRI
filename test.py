"""Run inference with a trained model on a test set and write q-maps, metrics and statistics.

    python test.py experiment=full_model test_set=umcu checkpoint=paper   # published checkpoint
    python test.py experiment=full_model test_set=mrrate                  # own run in outputs/
"""
import hydra

from qmap.testing import run_test


@hydra.main(config_path="configs", config_name="test", version_base="1.3")
def main(cfg):
    run_test(cfg)


if __name__ == '__main__':
    main()
