"""Declare the replacement four-task development protocol explicitly."""
import os,sys
from pathlib import Path
import yaml
from toolkits.robotwin_mopd import author_pool
from rlinf.envs.robotwin import multitask_env
TASKS=('open_laptop','handover_mic','dump_bin_bigbin','stack_bowls_two')
horizons=yaml.safe_load((Path(os.environ['ROBOTWIN_PATH'])/'task_config/_eval_step_limit.yml').read_text())
author_pool.TASKS=TASKS
for task in TASKS:multitask_env.TASK_HORIZONS[task]=int(horizons[task])
mode=sys.argv.pop(1)
for task in ('open_laptop','dump_bin_bigbin'):
 sys.argv.extend(['--override',f'+env.eval.task_horizons.{task}={int(horizons[task])}'])
if mode=='pool':author_pool.main()
elif mode=='eval':
 import torch
 torch.cuda.set_per_process_memory_fraction(0.28,0)
 from toolkits.robotwin_mopd.evaluate import main
 main()
else:raise ValueError(mode)
