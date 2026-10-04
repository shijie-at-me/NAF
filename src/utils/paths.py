"""Where the project keeps its files; each directory can be moved through an environment variable."""

import os

# Repository root (this file is <root>/src/utils/paths.py)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Downloaded and trained checkpoints: <root>/weights (git-ignored), or $NAF_WEIGHTS_DIR
WEIGHTS_DIR = os.environ.get("NAF_WEIGHTS_DIR", os.path.join(ROOT, "weights"))
# The dvt_ / fit3d_ fine-tuned backbones, as <tag>_<model name>.pth: <WEIGHTS_DIR>/finetuned, or $NAF_FINETUNED_CKPT_DIR
FINETUNED_CKPT_DIR = os.environ.get("NAF_FINETUNED_CKPT_DIR", os.path.join(WEIGHTS_DIR, "finetuned"))
