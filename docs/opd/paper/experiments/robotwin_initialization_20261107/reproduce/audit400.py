"""Audit raw development episodes and checkpoint provenance before recommendation."""
import argparse,copy,csv,hashlib,json,math,os
from pathlib import Path
if not os.environ.get('OPD_INITIALIZATION_STUDY'):
 raise RuntimeError('Set OPD_INITIALIZATION_STUDY to the preserved study output directory')
HERE=Path(os.environ['OPD_INITIALIZATION_STUDY']).resolve()
TASKS=('open_laptop','handover_mic','dump_bin_bigbin','stack_bowls_two')
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load_result(path,task,n):
 d=json.loads(Path(path).read_text());assert d['task']==task
 assert d['completed_episodes']==d['requested_episodes']==len(d['episodes'])==n
 assert d['chunk_length']==25 and d['denoising_steps']==10 and d['model_config']['openpi']['action_horizon']==50
 assert d['evaluation_protocol']=='author_cache_replay_unseen'
 p=d['evaluation_pool'];pool=json.loads(Path(p['path']).read_text());assert digest(p['path'])==p['file_sha256']
 assert pool['pool_content_sha256']==p['pool_content_sha256']==canonical({k:v for k,v in pool.items() if k!='pool_content_sha256'})
 assert pool['protocol']['author_cohort']==0 and pool['protocol']['purpose']=='development'
 assert d['verified_runtime']==pool['protocol']['runtime_profile']
 checks=d['fixed_scene_checks'];assert len(checks)==n
 seeds=[]
 for i,(row,check,expected) in enumerate(zip(d['episodes'],checks,pool['tasks'][task],strict=True)):
  assert type(row['success_once']) is bool
  assert row['actual_reset_seed']==row['reset_seed']==expected['scene_seed']==check['scene_seed']
  assert row['instruction']==expected['instruction'] and row['actual_scene_info']==expected['episode_info']
  assert check['matched'] is True and check['expected_episode_info']==check['actual_scene_info']==expected['episode_info']
  assert type(row['simulation_steps']) is int and row['simulation_steps']>0
  seeds.append(row['actual_reset_seed'])
 assert len(set(seeds))==n
 assert d['successes']==sum(x['success_once'] for x in d['episodes'])
 assert d['simulation_steps']==sum(x['simulation_steps'] for x in d['episodes'])
 assert d['success_rate']==d['successes']/n
 return d

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
 initial_teachers={
  'open_laptop':HERE/'candidate/teacher_dev20.json',
  'dump_bin_bigbin':HERE/'candidate_dump_bin_bigbin/teacher_dev20.json',
  'handover_mic':HERE/'retained_teacher_eval/handover_mic.json',
  'stack_bowls_two':HERE/'retained_teacher_eval/stack_bowls_two.json',
 }
 rows=[];source_paths=[]
 for step in (100,200,300,400,500):
  directory=HERE/('replacement_eval' if step in (100,200,500) else f'replacement_eval{step}')
  paths=[directory/(f'sft{step}_{t}.json' if step in (100,200,500) else t+'.json') for t in TASKS]
  if not args.final and not all(p.exists() and json.loads(p.read_text()).get('completed_episodes')==20 for p in paths):continue
  ds=[load_result(p,t,20) for p,t in zip(paths,TASKS,strict=True)]
  checkpoint=HERE/f'new_student_sft500/checkpoints/global_step_{step}'
  assert all(Path(d['checkpoint']).resolve()==checkpoint.resolve() for d in ds)
  row={'sft_step':step,**{t:d['success_rate'] for t,d in zip(TASKS,ds,strict=True)}};row['macro_sr']=sum(row[t] for t in TASKS)/4;rows.append(row);source_paths.extend(paths)
 with (HERE/'new_initialization_screening.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['sft_step',*TASKS,'macro_sr']);w.writeheader();w.writerows(rows)
 manifest=json.loads((HERE/'pace_manifest.json').read_text());norms={}
 for t in TASKS:
  p=HERE/('candidate/torch/open_laptop/assets/robotwin/norm_stats.json' if t=='open_laptop' else 'candidate_dump_bin_bigbin/torch/dump_bin_bigbin/assets/robotwin/norm_stats.json') if t in ('open_laptop','dump_bin_bigbin') else HERE/'replacement_norms'/t/'norm_stats.json'
  expected=next(m for m in manifest['models'] if m['task']==t)['norm_stats_sha256'];assert digest(p)==expected;norms[t]={'path':str(p),'sha256':expected}
 audit={'screening_checkpoints':[x['sft_step'] for x in rows],'inference':{'prediction_horizon':50,'execution_horizon':25,'ode_steps':10},'norms':norms,'formal_scenes_used':False,'complete':False}
 # Negative checks exercise scene, count, and predicate guards on a real fixture.
 fixture=HERE/'replacement_eval400/open_laptop.json'
 if fixture.exists():
  good=load_result(fixture,'open_laptop',20)
  for kind in ['count','scene','success_type']:
   bad=copy.deepcopy(good)
   if kind=='count':bad['completed_episodes']=19
   elif kind=='scene':bad['episodes'][0]['actual_reset_seed']+=1
   else:bad['episodes'][0]['success_once']=1
   temp=HERE/f'.audit_negative_{kind}.json';temp.write_text(json.dumps(bad))
   try:
    try:load_result(temp,'open_laptop',20)
    except AssertionError:pass
    else:raise RuntimeError(f'Negative audit guard failed: {kind}')
   finally:temp.unlink()
  audit['negative_guards_passed']=['count','scene_identity','boolean_success']
 if args.final:
  assert [x['sft_step'] for x in rows]==[100,200,300,400,500]
  assert (HERE/'new_student_sft500/COMPLETE').is_file()
  confirmed={};fresh={}
  for t in TASKS:
   first_student=load_result(HERE/'replacement_eval400'/f'{t}.json',t,20)
   first_teacher=load_result(initial_teachers[t],t,20)
   extra_student=load_result(HERE/'validation400'/f'student400_{t}.json',t,30)
   extra_teacher=load_result(HERE/'validation400'/f'teachers_{t}.json',t,30)
   ds=[first_student,first_teacher,extra_student,extra_teacher]
   assert len({canonical(d['verified_runtime']) for d in ds})==1
   assert first_student['checkpoint']==extra_student['checkpoint']
   assert first_teacher['checkpoint']==extra_teacher['checkpoint']
   for d in ds:assert d['normalization'][t]==first_student['normalization'][t]
   for a,b in [(first_student,first_teacher),(extra_student,extra_teacher)]:
    assert [(x['actual_reset_seed'],x['instruction'],x['actual_scene_info']) for x in a['episodes']]==[(x['actual_reset_seed'],x['instruction'],x['actual_scene_info']) for x in b['episodes']]
   assert not {x['actual_reset_seed'] for x in first_student['episodes']} & {x['actual_reset_seed'] for x in extra_student['episodes']}
   s=first_student['successes']+extra_student['successes'];q=first_teacher['successes']+extra_teacher['successes']
   confirmed[t]={'student_successes':s,'teacher_successes':q,'episodes_per_model':50,'student_sr':s/50,'teacher_sr':q/50,'teacher_minus_student_pp':2*(q-s)}
   fresh[t]={'student_successes':extra_student['successes'],'teacher_successes':extra_teacher['successes'],'episodes_per_model':30}
   source_paths.extend([initial_teachers[t],HERE/'validation400'/f'student400_{t}.json',HERE/'validation400'/f'teachers_{t}.json'])
  launch=json.loads((HERE/'new_student_sft500/launch.json').read_text());assert launch['data']['episodes']=={t:45 for t in TASKS}
  assert launch['data']['episode_ids']=={t:list(range(45)) for t in TASKS}
  metrics=[json.loads(l) for l in (HERE/'new_student_sft500/metrics.jsonl').read_text().splitlines()]
  assert len(metrics)==500 and [x['step'] for x in metrics]==list(range(1,501))
  for m in metrics:
   assert all(m[t+'/samples']==32 for t in TASKS)
   assert math.isfinite(m['loss']) and math.isfinite(m['grad_norm'])
  weights=HERE/'new_student_sft500/checkpoints/global_step_400/actor/model_state_dict/full_weights.pt'
  audit.update(complete=True,confirmed_50_per_task=confirmed,additional_30_only=fresh,student_macro_sr=sum(x['student_sr'] for x in confirmed.values())/4,teacher_macro_sr=sum(x['teacher_sr'] for x in confirmed.values())/4,checkpoint={'path':str(weights.parent.parent.parent),'weights':str(weights),'bytes':weights.stat().st_size,'sha256':digest(weights)},source_results={str(p):digest(p) for p in source_paths})
 (HERE/('final_initialization_audit400.json' if args.final else 'screening_audit400.json')).write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps(audit,indent=2)[:2500])
if __name__=='__main__':main()
