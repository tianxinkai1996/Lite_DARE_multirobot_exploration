"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
import torch

from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy', 'config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 不可用。")

    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.3, device=0)


    print("PyTorch GPU 30%")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()