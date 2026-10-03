"""MQAR control and composed associative lookup generators."""

from . import vocabulary as v
from .oracles import oracle_targets
from .records import example_from_targets
from .sampling import distribute, weighted_positions


def mqar_example(spec, rng, length):
    keys = rng.sample(range(v.IDENTITY_BASE, v.IDENTITY_BASE + spec.symbols), spec.pairs)
    values = [rng.randrange(v.IDENTITY_BASE + spec.symbols, spec.vocab_size) for _ in keys]
    tokens = [v.BOS] + [rng.randrange(v.IDENTITY_BASE + spec.symbols, spec.vocab_size)
                        for _ in range(length - 1)]
    for row, (key, value) in enumerate(zip(keys, values)):
        tokens[1 + 2 * row:3 + 2 * row] = [key, value]
    end = 1 + 2 * spec.pairs
    positions = weighted_positions(range(end + spec.query_gap, length), spec.queries, spec.alpha, rng)
    for position, key in zip(positions, rng.sample(keys, spec.queries)):
        tokens[position] = key
    return example_from_targets(spec.task, tokens, oracle_targets(tokens, spec))


def cycle_table(identities, rng):
    cycle = rng.sample(identities, len(identities))
    return {key: cycle[(index + 1) % len(cycle)] for index, key in enumerate(cycle)}


def lookup_pair(spec, rng, length):
    identities = rng.sample(range(v.IDENTITY_BASE, spec.vocab_size), spec.pairs)
    table = cycle_table(identities, rng)
    rows = rng.sample(identities, spec.pairs)
    queries = rng.sample(identities, spec.queries)
    spare = length - (2 + 4 * spec.pairs + 2 * spec.queries) - spec.query_gap
    gaps = distribute(spare, spec.queries + 1, rng)
    gaps[0] += spec.query_gap

    def encode(mapping):
        tokens = [v.BOS]
        for key in rows:
            tokens.extend((v.KEY, key, v.VALUE, mapping[key]))
        tokens.append(v.END_TABLE)
        tokens.extend([v.FILL] * gaps[0])
        for query, gap in zip(queries, gaps[1:]):
            tokens.extend((v.QUERY, query))
            tokens.extend([v.FILL] * gap)
        return example_from_targets(spec.task, tokens, oracle_targets(tokens, spec))

    first = encode(table)
    if spec.pairs == 2:
        # Only one two-vertex cycle exists. This smallest case is a control.
        return first, first
    # Conjugate the cycle by a transposition that fixes the first query and
    # changes its endpoint. This guarantees a different answer without retries.
    start = queries[0]
    endpoint = start
    for _ in range(spec.hops):
        endpoint = table[endpoint]
    alternative = rng.choice([token for token in identities if token not in (start, endpoint)])

    def swap(token):
        return alternative if token == endpoint else endpoint if token == alternative else token

    second = encode({swap(key): swap(value) for key, value in table.items()})
    return first, second
