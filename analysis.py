"""Analyze upsamplers beyond their evaluation scores, the companion of evaluation.py.

    python analysis.py analysis=<analysis> [overrides]

Analyses (``config/analysis/<analysis>.yaml``; code in ``src/analysis/``):
    attention_entropy  how peaked NAF's attention is around object boundaries
    guide_ratio        output / guide size settings scored with one probe trained on native features
    failure            where the upsamplers whose seg probes are registered fail, by pixel group, vs bilinear
    key_dilution       whether keys pooled over mixed patches mislead the attention (NAF, PixelUp)

Examples:
    python analysis.py analysis=attention_entropy dataset.root=<VOC root> img_size=224
    python analysis.py analysis=failure analysis.max_images=100 analysis.out=output/failure_analysis
"""

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.analysis import ANALYSES
from src.utils.run import start_run
from src.utils.training import seed_everything


@hydra.main(config_path="config", config_name="analysis", version_base=None)
def main(cfg: DictConfig):
    name = HydraConfig.get().runtime.choices.analysis
    with start_run(cfg, f"{name}.log", title=f"Analysis ({name})") as run:
        seed_everything(cfg.get("seed", 0))
        torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
        run.console.print(f"[bold yellow]Using device: {run.device}[/bold yellow]")
        ANALYSES[name](cfg, run)
        run.console.print(f"[bold blue]Done. Outputs and log in {run.dir}[/bold blue]")


if __name__ == "__main__":
    main()
