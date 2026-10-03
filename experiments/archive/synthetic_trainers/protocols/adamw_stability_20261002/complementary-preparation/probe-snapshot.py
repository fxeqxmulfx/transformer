"""Small implementation probes for future complementary AdamW comparisons."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import torch
from experiments.gpt_mini import GPTMini
from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
from experiments.synthetic_trainers.corpus import corpus_report, problem_key, study_pool
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.metrics import evaluate
from experiments.synthetic_trainers.oracles import validate_example
from experiments.synthetic_trainers.specs import TaskSpec
from experiments.synthetic_trainers.training import train_run

root=Path(__file__).parent
protocol=Path('experiments/synthetic_trainers/protocols/adamw_stability_20261002')
model=ModelSpec(width=8,layers=1,heads=1,init_std=.02)
config=TrainConfig(study='double_descent',steps=2,eval_every=1,batch_size=2,
                   train_examples=4,validation_examples=4,test_examples=4,
                   optimizer='adamw',beta1=.9,beta2=.98,learning_rate=.001,
                   weight_decay=.1,grad_clip=None,split_policy='disjoint',device='cpu',target=1.0)
specs=[TaskSpec(task='mqar',length=24,symbols=8,pairs=4,queries=2),
       TaskSpec(task='lookup',length=24,symbols=8,pairs=4,queries=2,hops=1),
       TaskSpec(task='lookup',length=24,symbols=8,pairs=4,queries=2,hops=2),
       TaskSpec(task='copy',length=4,symbols=8,number_limit=32),
       TaskSpec(task='parity',length=4,number_limit=32),
       TaskSpec(task='parity',length=4,scratchpad='running',number_limit=32),
       TaskSpec(task='crasp',length=8,formula_depth=2,formula_seed=0)]
manifest={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'model':asdict(model),'training':asdict(config),'cases':[asdict(s) for s in specs],'scope':'CPU_implementation_smokes_only; not_frozen_scientific_architecture_selection_or_confirmation'}
(root/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
def without_times(value):
    if isinstance(value,dict):return {k:without_times(v) for k,v in value.items() if k!='generation_seconds'}
    return value
rows=[]
for i,spec in enumerate(specs):
    label=f'{spec.task}-hops{spec.hops}' if spec.task=='lookup' else f'{spec.task}-{spec.scratchpad}'
    case=root/label
    cfg=replace(config,eval_lengths=(spec.length*2,))
    result=train_run(spec,model,cfg,case)
    history=[json.loads(s) for s in (case/'history.jsonl').read_text().splitlines()]
    assert [r['step'] for r in history]==[0,1,2]
    assert result['steps_completed']==2 and result['examples_seen']==4 and result['peak_cuda_bytes'] is None
    description=result['provenance']['optimizer']
    assert description=={'name':'adamw','betas':[.9,.98],'epsilon':1e-8,'bias_correction':True,'maximum_second_moment':False,'decoupled_decay':True,'decay_scope':'matrices','gradient_clip':None}
    pool=study_pool(spec,cfg)
    assert all(r['unique_inputs']==0 for r in corpus_report(pool)['overlaps'].values())
    for split in pool.values():
        for example in split.examples:validate_example(example,spec)
    seen={problem_key(r) for r in pool['train'].examples}
    assert all(problem_key(r) not in seen for r in pool['validation'].examples)
    transfer=f'length-{spec.length*2}'
    assert set(result['test'])==set(result['test_final'])=={'in_distribution',transfer}
    support=result['study']['generalization_transition']
    assert support['novel_validation_examples']==4 and support['novel_probe_examples'][transfer]==4
    assert all(r['validation_novel'] is not None and transfer in r['validation_ood_novel'] for r in history)
    provenance=result['provenance']
    for checkpoint,reported in [('final.pt','test_final'),('best.pt','test')]:
        loaded=GPTMini(model.reference_config(provenance['vocab_size'],provenance['context_length']))
        loaded.load_state_dict(torch.load(case/checkpoint,weights_only=True))
        for name,test_spec in [('in_distribution',spec),(transfer,replace(spec,length=spec.length*2,min_length=None))]:
            examples=pool['test'].examples if name=='in_distribution' else build_split(test_spec,'test',cfg.data_seed,cfg.test_examples).examples
            actual=evaluate(loaded,examples,cfg.batch_size,spec=test_spec)
            assert without_times(actual)==without_times(result[reported][name])
    rows.append({'name':label,'task':asdict(spec),'config':asdict(cfg),'optimizer':description,'parameters':result['parameters'],'canonical_observations':len(history),'steps_completed':2,'examples_seen':4,'novel_ID_validation_examples':support['novel_validation_examples'],'novel_length_probe_examples':support['novel_probe_examples'],'split_fingerprints':result['split_fingerprints'],'final_and_selected_checkpoints_reloaded_and_all_metrics_equal':True,'generative':spec.generative,'final_test_splits':sorted(result['test_final']),'selected_test_splits':sorted(result['test']),'result_path':str(case/'result.json'),'history_sha256':hashlib.sha256((case/'history.jsonl').read_bytes()).hexdigest()})
    print(label,'passed',flush=True)
paths=['experiments/synthetic_trainers/config.py','experiments/synthetic_trainers/runtime.py','experiments/synthetic_trainers/training.py','experiments/synthetic_trainers/cli.py','experiments/synthetic_trainers/studies.py','experiments/synthetic_trainers/corpus.py','experiments/synthetic_trainers/specs.py','experiments/synthetic_trainers/metrics.py']
frozen={}
for name in ['optimizer-pair-plan.json','larger-modulus-plan.json']:
    plan=json.loads((protocol/name).read_text())
    hashes={p:v for group in ['source_hashes','analysis_and_driver_source_hashes','papers'] for p,v in plan[group].items()}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    frozen[name]={'checked_files':len(hashes),'all_unchanged':True}
validation={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'scope':manifest['scope'],'cpu_only':True,'scientific_architecture_experiment_launched':False,'native_AdamW_scope':'matrices; differs_from_modular_all_parameter_decay','warmup_scope':'constant_learning_rate; differs_from_modular_ten_update_warmup','novel_ID_meaning':'unseen_complete_inputs_at_training_lengths; not_unseen_token_identifiers','learning_or_grokking_claim':False,'model':asdict(model),'cases':rows,'source_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},'frozen_live_and_queued_fingerprints':frozen,'probe_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
print('all_seven_CPU_cases_verified',flush=True)
