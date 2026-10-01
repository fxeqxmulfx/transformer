"""Exact causal oracles on raw input, independent of generator metadata.

Dyck-1: arXiv:2506.16055v3, Example 2.3 / Appendix A.2.
E_k: the neutral-letter language of Appendix F (app:tlclpos).
Lookup composition is the new experimental extension described in TASKS.md.
"""

from . import vocabulary as v


def mqar_targets(tokens, spec):
    targets = [v.IGNORE] * len(tokens)
    end = 1 + 2 * spec.pairs
    if not tokens or tokens[0] != v.BOS or len(tokens) < end:
        raise ValueError("Incomplete MQAR prefix")
    table = {}
    for position in range(1, end, 2):
        key, value = tokens[position:position + 2]
        if not v.IDENTITY_BASE <= key < v.IDENTITY_BASE + spec.symbols:
            raise ValueError("Invalid MQAR key")
        if not v.IDENTITY_BASE + spec.symbols <= value < spec.vocab_size:
            raise ValueError("Invalid MQAR value")
        if key in table:
            raise ValueError("Duplicate MQAR key")
        table[key] = value
    seen = set()
    for position in range(end, len(tokens)):
        token = tokens[position]
        if v.IDENTITY_BASE <= token < v.IDENTITY_BASE + spec.symbols:
            if position < end + spec.query_gap or token not in table or token in seen:
                raise ValueError("MQAR query must repeat one distinct prior key")
            targets[position] = table[token]
            seen.add(token)
        elif not v.IDENTITY_BASE + spec.symbols <= token < spec.vocab_size:
            raise ValueError("MQAR fillers must be value tokens")
    return targets


def lookup_targets(tokens, hops):
    if hops < 1 or not tokens or tokens[0] != v.BOS:
        raise ValueError("Lookup needs BOS and positive hops")
    targets = [v.IGNORE] * len(tokens)
    table, position = {}, 1
    while position < len(tokens) and tokens[position] == v.KEY:
        if position + 3 >= len(tokens) or tokens[position + 2] != v.VALUE:
            raise ValueError("Incomplete key/value record")
        key, value = tokens[position + 1], tokens[position + 3]
        if min(key, value) < v.IDENTITY_BASE or key in table:
            raise ValueError("Invalid or duplicate lookup key")
        table[key] = value
        position += 4
    if position >= len(tokens) or tokens[position] != v.END_TABLE:
        raise ValueError("Queries need a completed prior table")
    position += 1
    seen = set()
    while position < len(tokens):
        if tokens[position] == v.FILL:
            position += 1
            continue
        if tokens[position] != v.QUERY or position + 1 >= len(tokens):
            raise ValueError("Invalid lookup query record")
        start = tokens[position + 1]
        if start in seen:
            raise ValueError("Lookup queries must be distinct")
        seen.add(start)
        current, visited = start, {start}
        for _ in range(hops):
            if current not in table:
                raise ValueError("A query path lacks a prior link")
            current = table[current]
            if current in visited:
                raise ValueError("A query path repeats a vertex")
            visited.add(current)
        targets[position + 1] = current
        position += 2
    return targets


def prefix_targets(tokens, task, blocks=3):
    if not tokens or tokens[0] != v.BOS or task not in ("dyck", "blocks") or blocks < 1:
        raise ValueError("Invalid prefix task or missing BOS")
    targets = [v.IGNORE]
    balance, invalid, runs, first, previous = 0, False, 0, None, None
    for token in tokens[1:]:
        allowed = (v.OPEN, v.CLOSE, v.NEUTRAL) if task == "dyck" else (v.A, v.B, v.NEUTRAL)
        if token not in allowed:
            raise ValueError("Invalid prefix input token")
        if task == "dyck":
            balance += (token == v.OPEN) - (token == v.CLOSE)
            invalid = invalid or balance < 0
            targets.append(v.INVALID if invalid else v.BALANCED if balance == 0 else v.INCOMPLETE)
        else:
            if token != v.NEUTRAL:
                if first is None:
                    first = token
                runs += previous != token
                previous = token
            targets.append(v.ACCEPT if first == v.A and runs == blocks else v.REJECT)
    return targets


def oracle_targets(tokens, spec):
    if spec.task == "random_lm":
        raise ValueError("Random payloads have no deterministic raw-prompt oracle")
    if spec.generative:
        from .sequence_oracles import generation_answer

        # Validate teacher-forcing tokens by locating the unique prompt separator.
        if v.SEP not in tokens:
            raise ValueError("Missing answer separator")
        end = tokens.index(v.SEP) + 1
        prompt = tokens[:end]
        answer = generation_answer(prompt, spec)
        if tuple(tokens) != (*prompt, *answer[:-1]):
            raise ValueError("Teacher-forcing input disagrees with the raw-prompt answer")
        return [v.IGNORE] * (end - 1) + list(answer)
    if spec.task == "mqar":
        return mqar_targets(tokens, spec)
    if spec.task == "lookup":
        return lookup_targets(tokens, spec.hops)
    if spec.task == "dyck2":
        from .typed_dyck import typed_targets

        return typed_targets(tokens, spec.bracket_types)
    if spec.task == "crasp":
        from .crasp import crasp_targets

        return crasp_targets(tokens, spec)
    return prefix_targets(tokens, spec.task, spec.blocks)


def validate_example(example, spec):
    if spec.task == "random_lm":
        from .random_control import validate_random

        validate_random(example, spec)
        return
    minimum = spec.min_length or spec.length
    if spec.generative:
        from .sequence_oracles import generation_answer, generation_limit, generation_problem_size

        size = generation_problem_size(example.prompt, spec)
        if example.answer != generation_answer(example.prompt, spec):
            raise ValueError("Generated answer disagrees with the raw-prompt oracle")
        if example.generation_limit != generation_limit(example.prompt, spec):
            raise ValueError("Generation limit must depend only on the prompt format")
    else:
        size = len(example.tokens)
        if example.prompt:
            raise ValueError("A prefix task cannot contain a generated answer")
    if example.task != spec.task or not minimum <= size <= spec.length:
        raise ValueError("Example task or length does not match the specification")
    if any(not 0 <= token < spec.vocab_size for token in example.tokens):
        raise ValueError("Input token is outside the vocabulary")
    expected = oracle_targets(example.tokens, spec)
    if tuple(expected) != example.targets:
        raise ValueError("Targets disagree with the raw-input oracle")
    if len(example.tokens) > spec.context_length:
        raise ValueError("Serialized input exceeds the context bound")
    if spec.task == "addition" and spec.carry_length is not None:
        from .arithmetic import addition_inputs, longest_carry

        left, right, _ = addition_inputs(example.prompt, spec)
        if longest_carry(left, right) != spec.carry_length:
            raise ValueError("Addition input violates the requested carry length")
    if spec.task == "boolean_and":
        from .sequence_oracles import bit_inputs

        bits, _ = bit_inputs(example.prompt, spec)
        zeros = [i for i, token in enumerate(bits) if token == v.ZERO]
        cut = len(bits) - max(1, len(bits) // 4)
        if len(zeros) > 1 or spec.and_shift and any((i < cut) != (spec.and_region == "early") for i in zeros):
            raise ValueError("AND input violates its zero-position distribution")
    if spec.task in ("mqar", "lookup") and sum(x != v.IGNORE for x in expected) != spec.queries:
        raise ValueError("Wrong number of supervised queries")
    if spec.task == "lookup":
        if example.tokens.count(v.KEY) != spec.pairs:
            raise ValueError("Wrong number of lookup records")
        end = example.tokens.index(v.END_TABLE) + 1
        first_query = example.tokens.index(v.QUERY)
        if first_query < end + spec.query_gap:
            raise ValueError("Lookup query violates query_gap")
    previous = None
    for target, change in zip(expected, example.changes):
        if change != (target != v.IGNORE and (previous is None or previous != target)):
            raise ValueError("Incorrect state-change mask")
        if target != v.IGNORE:
            previous = target
