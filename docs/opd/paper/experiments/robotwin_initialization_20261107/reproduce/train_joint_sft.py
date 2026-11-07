"""Explicit four-task cold start using the unchanged audited SFT objective."""
import hashlib,json,os
from pathlib import Path
import torch
from omegaconf import open_dict
from toolkits.robotwin_mopd import joint_sft_data
TASKS=('open_laptop','handover_mic','dump_bin_bigbin','stack_bowls_two')
joint_sft_data.TASKS=TASKS
from toolkits.robotwin_mopd import joint_sft
joint_sft.TASKS=TASKS
HERE=Path(__file__).resolve().parent
original=joint_sft.compose_config

def compose(args):
 cfg=original(args)
 with open_dict(cfg):
  cfg.actor.model.opd_task_norm_stats={t:str(HERE/'assets'/t/'norm_stats.json') for t in TASKS}
 return cfg
joint_sft.compose_config=compose
identity=joint_sft.source_identity

def provenance():
 return {**identity(),'exploration_configuration':{'tasks':TASKS,'wrapper':str(Path(__file__).resolve()),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'initialization':'same pi05 base; no expert weights merged; four-task demonstration SFT'}}
joint_sft.source_identity=provenance
if __name__ == '__main__':
 if '--inspect-data' not in __import__('sys').argv:
  torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
  torch.cuda.set_per_process_memory_fraction(0.70,torch.cuda.current_device())
 joint_sft.main()
