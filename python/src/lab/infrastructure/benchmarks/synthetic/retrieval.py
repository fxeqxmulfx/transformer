"""Associative recall, and lookup composed over several hops.

A port of `experiments/synthetic_trainers/lookup.py` and of the oracles of
`oracles.py`; composed lookup is the experimental extension of
`experiments/archive/synthetic_trainers/TASKS.md`, and the rewrites of MQAR
extend the port (`lab.domain.tasks.MQAR`).
"""

from .base import Generator
from .sampling import distribute, weighted_positions
from .vocabulary import BOS, END_TABLE, FILL, IDENTITY_BASE, IGNORE, KEY, QUERY, VALUE


class MQAR(Generator):
    name = "mqar"

    @property
    def vocab(self):
        return IDENTITY_BASE + 2 * self.task.symbols

    def controls(self):
        task = self.task
        # A split without rewrites keeps the historical seed and fingerprint.
        overwrites = {"overwrites": task.overwrites} if task.overwrites else {}
        return {"symbols": task.symbols, "pairs": task.pairs, "queries": task.queries,
                "query_gap": task.query_gap, "alpha": float(task.alpha), **overwrites}

    def pair(self, rng, length):
        return self.sample(rng, length), self.sample(rng, length)

    def sample(self, rng, length):
        """Without rewrites, every draw is the historical one, in the historical order."""
        task = self.task
        distinct = rng.sample(range(IDENTITY_BASE, IDENTITY_BASE + task.symbols), task.pairs - task.overwrites)
        keys = distinct + [rng.choice(distinct) for _ in range(task.overwrites)]
        if task.overwrites:
            rng.shuffle(keys)
        values, current = [], {}
        for key in keys:
            # A rewrite draws uniformly among the values other than its key's current one.
            value = rng.randrange(IDENTITY_BASE + task.symbols, self.vocab - (key in current))
            if key in current and value >= current[key]:
                value += 1
            current[key] = value
            values.append(value)
        tokens = [BOS] + [rng.randrange(IDENTITY_BASE + task.symbols, self.vocab) for _ in range(length - 1)]
        for row, (key, value) in enumerate(zip(keys, values)):
            tokens[1 + 2 * row:3 + 2 * row] = [key, value]
        end = 1 + 2 * task.pairs
        positions = weighted_positions(range(end + task.query_gap, length), task.queries, task.alpha, rng)
        for position, key in zip(positions, rng.sample(distinct, task.queries)):
            tokens[position] = key
        return self.example(tokens)

    def writes(self, tokens):
        """The key and value of every write, in order."""
        end = 1 + 2 * self.task.pairs
        return [tuple(tokens[position:position + 2]) for position in range(1, end, 2)]

    def targets(self, tokens):
        task = self.task
        targets = [IGNORE] * len(tokens)
        end = 1 + 2 * task.pairs
        if not tokens or tokens[0] != BOS or len(tokens) < end:
            raise ValueError("Incomplete MQAR prefix")
        table = {}
        for key, value in self.writes(tokens):
            if not IDENTITY_BASE <= key < IDENTITY_BASE + task.symbols:
                raise ValueError("Invalid MQAR key")
            if not IDENTITY_BASE + task.symbols <= value < self.vocab:
                raise ValueError("Invalid MQAR value")
            if key in table and not task.overwrites:
                raise ValueError("Duplicate MQAR key")
            table[key] = value
        seen = set()
        for position in range(end, len(tokens)):
            token = tokens[position]
            if IDENTITY_BASE <= token < IDENTITY_BASE + task.symbols:
                if position < end + task.query_gap or token not in table or token in seen:
                    raise ValueError("MQAR query must repeat one distinct prior key")
                targets[position] = table[token]
                seen.add(token)
            elif not IDENTITY_BASE + task.symbols <= token < self.vocab:
                raise ValueError("MQAR fillers must be value tokens")
        return targets

    def check(self, example, expected):
        task = self.task
        if sum(target != IGNORE for target in expected) != task.queries:
            raise ValueError("Wrong number of supervised queries")
        current = {}
        for key, value in self.writes(example.tokens):
            if current.get(key) == value:
                raise ValueError("A rewrite must change its key's value")
            current[key] = value
        if len(current) != task.pairs - task.overwrites:
            raise ValueError("The writes must bind pairs - overwrites distinct keys")

    def content(self, example, target):
        return range(IDENTITY_BASE + self.task.symbols, self.vocab)


def cycle_table(identities, rng):
    cycle = rng.sample(identities, len(identities))
    return {key: cycle[(index + 1) % len(cycle)] for index, key in enumerate(cycle)}


class Lookup(Generator):
    name = "lookup"

    @property
    def vocab(self):
        return IDENTITY_BASE + self.task.symbols

    def controls(self):
        task = self.task
        return {"symbols": task.symbols, "pairs": task.pairs, "queries": task.queries,
                "query_gap": task.query_gap, "hops": task.hops}

    def pair(self, rng, length):
        task = self.task
        identities = rng.sample(range(IDENTITY_BASE, self.vocab), task.pairs)
        table = cycle_table(identities, rng)
        rows = rng.sample(identities, task.pairs)
        queries = rng.sample(identities, task.queries)
        spare = length - (2 + 4 * task.pairs + 2 * task.queries) - task.query_gap
        gaps = distribute(spare, task.queries + 1, rng)
        gaps[0] += task.query_gap

        def encode(mapping):
            tokens = [BOS]
            for key in rows:
                tokens.extend((KEY, key, VALUE, mapping[key]))
            tokens.append(END_TABLE)
            tokens.extend([FILL] * gaps[0])
            for query, gap in zip(queries, gaps[1:]):
                tokens.extend((QUERY, query))
                tokens.extend([FILL] * gap)
            return self.example(tokens)

        first = encode(table)
        if task.pairs == 2:
            # Only one two-vertex cycle exists. This smallest case is a control.
            return first, first
        # Conjugate the cycle by a transposition that fixes the first query and
        # changes its endpoint. This guarantees a different answer without retries.
        start = queries[0]
        endpoint = start
        for _ in range(task.hops):
            endpoint = table[endpoint]
        alternative = rng.choice([token for token in identities if token not in (start, endpoint)])

        def swap(token):
            return alternative if token == endpoint else endpoint if token == alternative else token

        return first, encode({swap(key): swap(value) for key, value in table.items()})

    def targets(self, tokens):
        hops = self.task.hops
        if hops < 1 or not tokens or tokens[0] != BOS:
            raise ValueError("Lookup needs BOS and positive hops")
        targets = [IGNORE] * len(tokens)
        table, position = {}, 1
        while position < len(tokens) and tokens[position] == KEY:
            if position + 3 >= len(tokens) or tokens[position + 2] != VALUE:
                raise ValueError("Incomplete key/value record")
            key, value = tokens[position + 1], tokens[position + 3]
            if min(key, value) < IDENTITY_BASE or key in table:
                raise ValueError("Invalid or duplicate lookup key")
            table[key] = value
            position += 4
        if position >= len(tokens) or tokens[position] != END_TABLE:
            raise ValueError("Queries need a completed prior table")
        position += 1
        seen = set()
        while position < len(tokens):
            if tokens[position] == FILL:
                position += 1
                continue
            if tokens[position] != QUERY or position + 1 >= len(tokens):
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

    def check(self, example, expected):
        task = self.task
        if sum(target != IGNORE for target in expected) != task.queries:
            raise ValueError("Wrong number of supervised queries")
        if example.tokens.count(KEY) != task.pairs:
            raise ValueError("Wrong number of lookup records")
        if example.tokens.index(QUERY) < example.tokens.index(END_TABLE) + 1 + task.query_gap:
            raise ValueError("Lookup query violates query_gap")

    def content(self, example, target):
        return range(IDENTITY_BASE, self.vocab)
