"""Exercise portable verification against actual completed CPU archives."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_scheduled_case import PROTOCOL, verify
from experiments.synthetic_trainers.scheduled_layout import digest
from experiments.synthetic_trainers.stability_report import write_json


stage = Path('experiments/runs/adamw_stability_20261002/bootstrap/schedule_pair_CPU_pipeline')
root = Path('experiments/runs/adamw_stability_20261002/bootstrap/scheduled-case-verification-CPU-20261003-verified')
if root.exists():
    raise FileExistsError(root)
root.mkdir(parents=True)
results, rejections = {}, []
for name, label in (('adamw-constant', 'constant'), ('adamw-cosine-tail', 'cosine')):
    archive = PROTOCOL / 'scheduled-pair-preparation' / name
    output = root / label
    result = verify(stage, name, archive, output)
    assert result['scientific_run'] is False
    assert result['observation_counts'] == {'canonical': 9, 'probes.jsonl': 16, 'diagnostics.jsonl': 24, 'gradients.jsonl': 40}
    assert result['assessment']['stable_grokking'] is False
    assert result['Torch_imported'] is False and result['peak_memory_checked_against_recorded_segments'] is False
    replay = root / (label + '-fresh-replay')
    command = [sys.executable, str(output / 'verification-snapshot.py'), name,
               '--stage', str(stage), '--archive', str(archive), '--output', str(replay)]
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(Path.cwd())
    process = subprocess.run(command, capture_output=True, text=True, env=environment)
    if process.returncode:
        raise RuntimeError(process.stderr)
    (root / (label + '-replay.stdout')).write_text(process.stdout)
    names = ('verification-snapshot.py', 'recovery-metrics.json', 'recovery-series.json')
    assert all((output / filename).read_bytes() == (replay / filename).read_bytes() for filename in names)
    before = digest(output / 'artifact-hashes.json')
    try:
        verify(stage, name, archive, output)
    except FileExistsError as error:
        assert digest(output / 'artifact-hashes.json') == before
        rejections.append({'case': label + '-existing-receipt', 'exception': str(error), 'receipt_preserved': True})
    else:
        raise AssertionError('An existing receipt must be preserved')
    results[label] = {'completed_updates': result['completed_updates'], 'parameters': result['parameters'],
        'scientific_run': False, 'Torch_imported': False, 'observation_counts': result['observation_counts'],
        'stable_grokking': False, 'fresh_replay_three_core_files_byte_exact': True,
        'checkpoint_sha256': result['checkpoint_sha256']}

live = Path(json.loads((PROTOCOL / 'scheduled-pair-plan.json').read_text())['output_directory'])
rejected = root / 'live-incomplete-case-receipt'
try:
    verify(live, 'adamw-constant', PROTOCOL / 'scheduled-pair-preparation' / 'adamw-constant', rejected)
except ValueError as error:
    assert 'complete frozen case' in str(error) and not rejected.exists()
    rejections.append({'case': 'actual-live-incomplete-scientific-case', 'exception': str(error), 'no_output_created': True})
else:
    raise AssertionError('The incomplete live case must be rejected')

with tempfile.TemporaryDirectory(prefix='scheduled-case-forgeries-') as directory:
    temporary = Path(directory)
    forged_plan = temporary / 'scientific-plan'
    forged_plan.mkdir()
    plan = json.loads((live / 'plan.json').read_text())
    plan['forged_annotation'] = 'uncommitted-plan-field'
    write_json(forged_plan / 'plan.json', plan)
    rejected = root / 'changed-scientific-plan-receipt'
    try:
        verify(forged_plan, 'adamw-constant', PROTOCOL / 'scheduled-pair-preparation' / 'adamw-constant', rejected)
    except ValueError as error:
        assert 'exact committed prospective plan' in str(error) and not rejected.exists()
        rejections.append({'case': 'changed-scientific-plan', 'exception': str(error), 'no_output_created': True})
    else:
        raise AssertionError('The changed scientific plan must be rejected')
    for kind in ('canonical', 'gradients', 'recovery', 'checkpoint'):
        altered = temporary / kind
        shutil.copytree(stage, altered)
        case = altered / 'adamw-constant'
        if kind in ('canonical', 'gradients'):
            filename = 'history.jsonl' if kind == 'canonical' else 'gradients.jsonl'
            path = case / filename
            rows = path.read_text().splitlines()
            path.write_text('\n'.join(rows[:-1]) + '\n')
        elif kind == 'recovery':
            path = case / 'recovery-metrics.json'
            value = json.loads(path.read_text())
            value['forged_derivation'] = True
            write_json(path, value)
        else:
            with (case / 'checkpoint.pt').open('ab') as stream:
                stream.write(b'changed-checkpoint')
        rejected = root / ('changed-' + kind + '-receipt')
        try:
            verify(altered, 'adamw-constant', PROTOCOL / 'scheduled-pair-preparation' / 'adamw-constant', rejected)
        except ValueError as error:
            assert not rejected.exists()
            rejections.append({'case': 'changed-raw-' + kind, 'exception': str(error), 'no_output_created': True})
        else:
            raise AssertionError(kind + ' corruption must be rejected')
assert 'torch' not in sys.modules
receipt = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'completed_CPU_archive_verification_preparation; no_scientific_learning_result',
    'actual_completed_CPU_cases_verified': 2, 'updates_per_existing_CPU_case': 40,
    'new_optimizer_updates': 0, 'Torch_imported': False, 'GPU_context_created': False,
    'actual_fresh_snapshot_replays': 2, 'six_core_files_byte_exact_on_fresh_replay': True,
    'actual_rejections': rejections, 'case_receipts': results,
    'verification_program_sha256': digest(PROTOCOL / 'verify_scheduled_case.py'),
    'verification_exercise_sha256': digest(__file__),
    'scientific_pair_still_in_progress': True, 'scientific_gate_opened': False}
write_json(root / 'validation.json', receipt)
print(json.dumps(receipt, indent=2))
