"""Prefix languages: the status of every prefix, with neutral letters between the active ones.

A port of `experiments/synthetic_trainers/prefixes.py`, `typed_dyck.py` and
the prefix oracles of `oracles.py`. A third of the pairs are two random
words; the rest are a matched pair, two words with the same letter counts
and neutral gaps whose labels differ.
"""

from .base import Generator
from .sampling import distribute, insert_gaps, neutral_gaps
from .vocabulary import (A, ACCEPT, B, BALANCED, BOS, CLOSE, IDENTITY_BASE, IGNORE, INCOMPLETE, INVALID,
                         NEUTRAL, OPEN, REJECT, bracket_pairs)


def balanced_word(size, maximum, rng):
    if size < 0 or size % 2 or maximum < 1:
        raise ValueError("A balanced word needs an even size and positive balance bound")
    word, balance = [], 0
    for index in range(size):
        remaining = size - index
        opening = balance == 0 or (balance < maximum and balance < remaining and rng.random() < 0.5)
        word.append(OPEN if opening else CLOSE)
        balance += 1 if opening else -1
    return word


class Prefix(Generator):
    """A prefix language: `words` draws the matched pair, `word` a random word."""

    def pair(self, rng, length):
        if rng.random() < 1 / 3:
            # Broaden support beyond matched pairs and their specific corruptions.
            return self.random(rng, length), self.random(rng, length)
        size = self.task.active_size(length)
        words = self.words(rng, size)
        gaps = neutral_gaps(size, length - 1, rng, self.task.max_neutral_gap)
        return tuple(self.example([BOS, *insert_gaps(word, gaps, NEUTRAL)]) for word in words)

    def random(self, rng, length):
        size = self.task.active_size(length)
        word = self.word(rng, size)
        gaps = neutral_gaps(size, length - 1, rng, self.task.max_neutral_gap)
        return self.example([BOS, *insert_gaps(word, gaps, NEUTRAL)])


class Dyck(Prefix):
    name = "dyck"

    def controls(self):
        task = self.task
        return {"neutral_fraction": float(task.neutral_fraction), "max_neutral_gap": task.max_neutral_gap,
                "max_balance": task.max_balance}

    def words(self, rng, size):
        # The first component varies the location of the first violation.
        maximum = self.task.max_balance
        cut = 2 * rng.randint(1, (size - 4) // 2)
        first = balanced_word(cut, maximum, rng) + balanced_word(size - cut, maximum, rng)
        second = first.copy()
        candidates = [i for i in range(cut + 1, size - 1) if first[i] == CLOSE]
        closing = candidates[0] if maximum == 1 else rng.choice(candidates)
        second[cut], second[closing] = second[closing], second[cut]
        return first, second

    def word(self, rng, size):
        maximum = self.task.max_balance
        word, balance = [], 0
        for _ in range(size):
            opening = balance == -maximum or (balance < maximum and rng.random() < 0.5)
            word.append(OPEN if opening else CLOSE)
            balance += 1 if opening else -1
        return word

    def targets(self, tokens):
        if not tokens or tokens[0] != BOS:
            raise ValueError("Invalid prefix task or missing BOS")
        targets, balance, invalid = [IGNORE], 0, False
        for token in tokens[1:]:
            if token not in (OPEN, CLOSE, NEUTRAL):
                raise ValueError("Invalid prefix input token")
            balance += (token == OPEN) - (token == CLOSE)
            invalid = invalid or balance < 0
            targets.append(INVALID if invalid else BALANCED if balance == 0 else INCOMPLETE)
        return targets

    def content(self, example, target):
        return (BALANCED, INCOMPLETE, INVALID)


class AlternatingBlocks(Prefix):
    name = "blocks"

    def controls(self):
        task = self.task
        return {"neutral_fraction": float(task.neutral_fraction), "max_neutral_gap": task.max_neutral_gap,
                "blocks": task.blocks}

    def words(self, rng, size):
        blocks = self.task.blocks
        if blocks == 1:
            first = [A] * size
            second = first.copy()
            second[rng.randrange(size)] = B
            return first, second
        lengths = [2, 2] + [1] * (blocks - 2)
        extra = distribute(size - sum(lengths), blocks, rng)
        first = []
        for index, (base, added) in enumerate(zip(lengths, extra)):
            first.extend([A if index % 2 == 0 else B] * (base + added))
        # Splitting the first boundary introduces two runs with identical counts.
        second = first.copy()
        boundary = lengths[0] + extra[0]
        second[boundary - 1], second[boundary] = second[boundary], second[boundary - 1]
        return first, second

    def word(self, rng, size):
        runs = rng.randint(1, min(size, 2 * self.task.blocks + 3))
        lengths = distribute(size - runs, runs, rng)
        start = rng.choice((A, B))
        word = []
        for index, added in enumerate(lengths):
            token = start if index % 2 == 0 else B if start == A else A
            word.extend([token] * (1 + added))
        return word

    def targets(self, tokens):
        if not tokens or tokens[0] != BOS:
            raise ValueError("Invalid prefix task or missing BOS")
        targets, runs, first, previous = [IGNORE], 0, None, None
        for token in tokens[1:]:
            if token not in (A, B, NEUTRAL):
                raise ValueError("Invalid prefix input token")
            if token != NEUTRAL:
                if first is None:
                    first = token
                runs += previous != token
                previous = token
            targets.append(ACCEPT if first == A and runs == self.task.blocks else REJECT)
        return targets

    def content(self, example, target):
        return (REJECT, ACCEPT)


class TypedDyck(Prefix):
    name = "dyck2"

    @property
    def vocab(self):
        return IDENTITY_BASE + 2 * (self.task.types - 1)

    def controls(self):
        task = self.task
        return {"neutral_fraction": float(task.neutral_fraction), "max_neutral_gap": task.max_neutral_gap,
                "max_balance": task.max_balance, "bracket_types": task.types}

    def words(self, rng, size):
        types = self.task.types
        pairs = bracket_pairs(types)
        skeleton = balanced_word(size, self.task.max_balance, rng)
        word, stack, opened = [], [], 0
        for token in skeleton:
            if token == OPEN:
                kind = opened if opened < 2 else rng.randrange(types)
                opened += 1
                opening, closing = pairs[kind]
                word.append(opening)
                stack.append(closing)
            else:
                word.append(stack.pop())
        closings = [i for i, token in enumerate(word) if token in {pair[1] for pair in pairs}]
        first = rng.choice(closings)
        second = rng.choice([i for i in closings if word[i] != word[first]])
        corrupted = word.copy()
        corrupted[first], corrupted[second] = corrupted[second], corrupted[first]
        return word, corrupted

    def word(self, rng, size):
        pairs = bracket_pairs(self.task.types)
        maximum = self.task.max_balance
        word, balance = [], 0
        for _ in range(size):
            opening = balance == -maximum or (balance < maximum and rng.random() < 0.5)
            word.append(rng.choice(pairs)[0 if opening else 1])
            balance += 1 if opening else -1
        return word

    def targets(self, tokens):
        if not tokens or tokens[0] != BOS:
            raise ValueError("Typed Dyck needs BOS")
        opens = dict(bracket_pairs(self.task.types))
        closes = set(opens.values())
        stack, invalid, targets = [], False, [IGNORE]
        for token in tokens[1:]:
            if token == NEUTRAL:
                pass
            elif token in opens:
                stack.append(opens[token])
            elif token in closes:
                if not stack or stack.pop() != token:
                    invalid = True
            else:
                raise ValueError("Invalid typed bracket token")
            targets.append(INVALID if invalid else INCOMPLETE if stack else BALANCED)
        return targets

    def content(self, example, target):
        return (BALANCED, INCOMPLETE, INVALID)
