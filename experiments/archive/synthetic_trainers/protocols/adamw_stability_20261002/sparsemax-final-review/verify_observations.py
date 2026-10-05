"""Independently recompute the preliminary mod-193 inspection through the current lab.

EXPERIMENT_PLAN.md, step 5: read the source in commit 441f47b before trusting
the archived observations. This review uses lab-built models and locally
wrapped normalizers, not the removed historical trainers or their inspector.
It preserves the frozen partition order, supervised columns and batches.
Entropy is in natural units, and both numeric and EOS rows are grouped by
numeric correctness, matching the archived record's definitions. Fixed-weight
normalizer swaps and local derivatives cannot establish a training cause.
"""

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import runpy
import subprocess
import time

import torch
from torch.nn import functional as F

from lab.dsl import Softmax, Sparsemax, substitute
from lab.infrastructure.benchmarks.modular import make_corpus
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.legacy import import_state


ROOT = Path('/home/misha/lean_projects/transformer')
ARCHIVE = ROOT / 'experiments/archive/synthetic_trainers'
RAW = ARCHIVE / 'runs/adamw_stability_20261002/attention_mod193_fraction25_lr0003_budget300k'
OBSERVATIONS = ARCHIVE / 'runs/adamw_stability_20261002/bootstrap/sparsemax-generalization-inspection-20261003/observations.json'
OUTPUT = Path('/tmp/mod193-archive-independent-review.json')
ROUNDING_FIELDS = {'entropy_sum', 'mean_entropy', 'numerator_mass_sum', 'denominator_mass_sum',
                   'score_gradient_L2', 'CPU_loss'}
DIFFERENCES = []


def sha(path):
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


@contextmanager
def capture(model, gradients=False):
    """Wrap each actual normalizer, retaining its score tensor only for backward probes."""
    pairs, originals = [], []
    try:
        for block in model.blocks:
            attention = block.attention
            original = attention.weights
            originals.append((attention, original))

            def recorded(scores, original=original):
                weights = original(scores)
                if gradients:
                    scores.retain_grad()
                pairs.append((scores, weights))
                return weights

            attention.weights = recorded
        yield pairs
    finally:
        for attention, original in originals:
            attention.weights = original


@torch.no_grad()
def partition(model, rows):
    totals = defaultdict(lambda: defaultdict(float))
    numeric, complete, stops = 0, 0, 0
    for start in range(0, len(rows), 512):
        batch = torch.tensor(rows[start:start + 512])
        with capture(model) as pairs:
            logits = model(batch[:, :-1])[:, 4:, :]
        assert len(pairs) == 2
        correct = logits.argmax(-1) == batch[:, 5:]
        numeric += int(correct[:, 0].sum())
        complete += int(correct.all(-1).sum())
        stops += int(correct[:, 1].sum())
        for layer, (scores, weights) in enumerate(pairs):
            length = scores.shape[-1]
            future = torch.ones(length, length, dtype=torch.bool).triu(1)
            assert bool(torch.isfinite(scores[..., ~future]).all())
            assert bool((weights >= 0).all()) and not bool(weights[..., future].any())
            assert float((weights.sum(-1) - 1).abs().max()) < 1e-6
            ordered = scores.masked_fill(future, -torch.inf).sort(descending=True, dim=-1).values
            gap = ordered[..., 0] - ordered[..., 1]
            support = (weights > 0).sum(-1)
            probabilities = weights.double()
            entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
            for head in range(weights.shape[1]):
                for position, label in ((4, 'numeric'), (5, 'EOS')):
                    for flag, group in ((True, 'numeric_correct'), (False, 'numeric_wrong')):
                        chosen = correct[:, 0] == flag
                        count = int(chosen.sum())
                        if not count:
                            continue
                        row = totals[(layer, head, label, group)]
                        row['rows'] += count
                        row['singleton_rows'] += int((support[chosen, head, position] == 1).sum())
                        row['strict_unit_gap_rows'] += int((gap[chosen, head, position] > 1).sum())
                        operands = weights[chosen, head, position][:, [1, 3]]
                        row['numerator_zero_rows'] += int((operands[:, 0] == 0).sum())
                        row['denominator_zero_rows'] += int((operands[:, 1] == 0).sum())
                        row['both_operand_zero_rows'] += int((operands == 0).all(-1).sum())
                        row['numerator_mass_sum'] += float(operands[:, 0].double().sum())
                        row['denominator_mass_sum'] += float(operands[:, 1].double().sum())
                        row['entropy_sum'] += float(entropy[chosen, head, position].sum())
    routing = [{'layer': layer, 'head': head, 'position': position, 'group': group,
                **dict(row), 'mean_entropy': row['entropy_sum'] / row['rows']}
               for (layer, head, position, group), row in sorted(totals.items())]
    return {'examples': len(rows), 'numeric_correct': numeric, 'complete_RHS_correct': complete,
            'EOS_correct': stops, 'accuracy': complete / len(rows), 'routing': routing}


def derivative(model, rows):
    batch = torch.tensor(rows[:512])
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    with capture(model, gradients=True) as pairs:
        logits = model(batch[:, :-1])[:, 4:, :]
        target = batch[:, 5:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        loss.backward()
    observed = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    correct = logits.argmax(-1)[:, 0] == target[:, 0]
    result = []
    for layer, (scores, weights) in enumerate(pairs):
        support = (weights.detach() > 0).sum(-1)
        gradient = scores.grad.detach()
        assert bool(torch.isfinite(gradient).all())
        assert not bool(gradient[weights.detach() == 0].any())
        for position, label in ((4, 'numeric'), (5, 'EOS')):
            for flag, group in ((True, 'numeric_correct'), (False, 'numeric_wrong')):
                chosen = correct == flag
                rows_gradient = gradient[chosen, :, position]
                result.append({'layer': layer, 'position': label, 'group': group,
                               'rows': rows_gradient.shape[0] * rows_gradient.shape[1],
                               'singleton_rows': int((support[chosen, :, position] == 1).sum()),
                               'entire_zero_score_gradient_rows': int((rows_gradient == 0).all(-1).sum()),
                               'score_gradient_L2': float(rows_gradient.double().norm())})
    model.zero_grad(set_to_none=True)
    ordinary_logits = model(batch[:, :-1])[:, 4:, :]
    ordinary_loss = F.cross_entropy(ordinary_logits.reshape(-1, ordinary_logits.shape[-1]), target.reshape(-1))
    ordinary_loss.backward()
    assert torch.equal(logits, ordinary_logits) and torch.equal(loss, ordinary_loss)
    assert all(torch.equal(parameter, before[name]) and torch.equal(parameter.grad, observed[name])
               for name, parameter in model.named_parameters())
    return {'examples': len(batch), 'CPU_loss': float(loss.detach()), 'numeric_correct': int(correct.sum()),
            'rows': result, 'all_inactive_score_gradient_entries_exactly_zero': True,
            'instrumented_logits_loss_and_every_parameter_gradient_match': True}


def compare(actual, expected, path='results'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            compare(actual[key], value, f'{path}.{key}')
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            compare(a, b, f'{path}[{index}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        field = path.rsplit('.', 1)[-1]
        if field in ROUNDING_FIELDS:
            assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-12), (path, actual, expected)
        else:
            assert actual == expected, (path, actual, expected)
        if actual != expected:
            DIFFERENCES.append({'field': path, 'actual': actual, 'archived': expected,
                                'absolute_difference': abs(actual - expected)})
    else:
        assert actual == expected, (path, actual, expected)


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    archived = json.loads(OBSERVATIONS.read_text())
    program = subprocess.run(['git', 'show', '441f47b:experiments/archive/synthetic_trainers/protocols/adamw_stability_20261002/inspect_sparsemax_generalization.py'],
                             cwd=ROOT, check=True, capture_output=True).stdout
    assert hashlib.sha256(program).hexdigest() == archived['program_sha256']
    study = runpy.run_path(str(ROOT / 'experiments/mod193_stability/experiment.py'))['experiments']
    corpus = make_corpus(study['base'].benchmark, 0)
    assert corpus.summary() == archived['corpus']
    results, checkpoint_hashes = {}, {}
    for trained in ('sparsemax', 'softmax'):
        checkpoint_path = RAW / f'adamw-{trained}' / 'checkpoint.pt'
        checkpoint_hashes[trained] = sha(checkpoint_path)
        assert checkpoint_hashes[trained] == archived['checkpoint_hashes'][trained]
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == 300000
        for forward in (trained, 'softmax' if trained == 'sparsemax' else 'sparsemax'):
            spec = substitute(study['base'].model, Softmax, Sparsemax()) if forward == 'sparsemax' else study['base'].model
            model = build_model(spec, len(corpus.tokens), 0).eval()
            model.load_state_dict(import_state(checkpoint['model']), strict=True)
            assert sum(parameter.numel() for parameter in model.parameters()) == 436104
            before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            key = f'weights-{trained}_forward-{forward}'
            result = {name: partition(model, rows) for name, rows in (('train', corpus.train), ('heldout', corpus.heldout))}
            if trained == forward:
                result['derivative_probes'] = {name: derivative(model, rows) for name, rows in
                                              (('train', corpus.train), ('heldout', corpus.heldout))}
            assert all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())
            compare(result, archived['results'][key], key)
            results[key] = result
            print(json.dumps({'case': key, 'counts_match': True, 'train_accuracy': result['train']['accuracy'],
                              'heldout_accuracy': result['heldout']['accuracy']}), flush=True)
    assert not torch.cuda.is_initialized()
    source = Path(__file__).read_text()
    record = {'reviewed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'scope': 'independent_CPU_recomputation_of_archived_fixed_weight_routing_and_local_derivatives; no_training_cause_established',
              'program_source': source, 'program_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'archived_observations_sha256': sha(OBSERVATIONS), 'archived_program_sha256': archived['program_sha256'],
              'checkpoint_hashes': checkpoint_hashes, 'corpus': corpus.summary(),
              'optimizer_updates_performed': 0, 'CUDA_initialized': False,
              'all_integer_fields_match': True, 'rounding_policy': {'relative': 1e-6, 'absolute': 1e-12},
              'rounding_differences': DIFFERENCES, 'results': results,
              'lab_source_hashes': {str(path.relative_to(ROOT)): sha(path) for path in
                  sorted((ROOT / 'python/src/lab/infrastructure/nn').glob('*.py')) +
                  [ROOT / 'python/src/lab/infrastructure/benchmarks/modular.py', ROOT / 'experiments/mod193_stability/experiment.py']}}
    OUTPUT.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'review': str(OUTPUT), 'all_integer_fields_match': True,
                      'rounding_differences': len(DIFFERENCES)}), flush=True)


if __name__ == '__main__':
    main()
