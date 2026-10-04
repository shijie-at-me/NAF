"""Train an upsampler on feature regression, the counterpart of evaluation.py.

    python train.py [overrides]

The task and the training loop are in ``src/training/``. Checkpoints (``model_<steps>steps.pth``, four along the
run plus the final one), the log and the TensorBoard events go to the Hydra run dir.

Examples:
    python train.py model=naf backbone.name=vit_small_patch14_reg4_dinov2 train_steps=25000
    python train.py dataset=imagenet img_size=448 sanity=true
"""

import hydra
from omegaconf import DictConfig

from src.training import train
from src.utils.run import start_run


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    with start_run(cfg, "train.log", tensorboard=True) as run:
        train(cfg, run)


if __name__ == "__main__":
    main()
