"""Raw-prompt oracles for RASP and RASP-L tasks with generated answers.

RASP: arXiv:2106.06981v2, Section 5. RASP-L: arXiv:2310.16028v1,
Section 5 and Appendix experimental details. Original noncausal RASP outputs
are serialized after the entire input for the causal mini GPT.
"""

from collections import Counter

from . import vocabulary as v
from .arithmetic import addition_answer, addition_inputs


def sequence_body(prompt, spec):
    bos = spec.histogram_bos if spec.task == "histogram" else True
    if not prompt or prompt[-1] != v.SEP or prompt.count(v.SEP) != 1:
        raise ValueError("A prompt needs exactly one final answer separator")
    if bos and prompt[0] != v.BOS:
        raise ValueError("Missing BOS")
    body = tuple(prompt[int(bos):-1])
    if not body:
        raise ValueError("An empty problem is unsupported")
    return body


def bit_inputs(prompt, spec):
    body = sequence_body(prompt, spec)
    if spec.task == "parity" and spec.index_hints:
        if len(body) % 2:
            raise ValueError("Index hints must precede each bit")
        hints, bits = body[::2], body[1::2]
        if any(not spec.number_base <= x <= spec.number_base + spec.number_limit for x in hints):
            raise ValueError("Invalid bit index hint")
        if any(b != a + 1 for a, b in zip(hints, hints[1:])):
            raise ValueError("Bit hints must form a contiguous ascending interval")
    else:
        hints, bits = (), body
    if any(x not in (v.ZERO, v.ONE) for x in bits):
        raise ValueError("Invalid bit input")
    return bits, hints


def count_interval(prompt, spec):
    body = sequence_body(prompt, spec)
    if len(body) != 2 or any(not spec.number_base + 1 <= x <= spec.number_base + spec.number_limit
                             for x in body) or body[1] < body[0]:
        raise ValueError("Count needs a nonempty ascending interval of atomic positive integers")
    return body


def generation_problem_size(prompt, spec):
    if spec.task == "random_lm":
        from .random_control import requested_length

        return requested_length(prompt, spec)
    if spec.task == "addition":
        return len(addition_inputs(prompt, spec)[0]) - 1
    if spec.task == "count":
        start, end = count_interval(prompt, spec)
        return end - start + 1
    if spec.task in ("parity", "boolean_and"):
        return len(bit_inputs(prompt, spec)[0])
    body = sequence_body(prompt, spec)
    if any(not v.IDENTITY_BASE <= x < v.IDENTITY_BASE + spec.symbols for x in body):
        raise ValueError("Invalid symbol in sequence input")
    if spec.task in ("copy", "reverse", "sort") and spec.unique and len(set(body)) != len(body):
        raise ValueError("A unique-token problem contains a repeated symbol")
    return len(body)


def generation_limit(prompt, spec):
    """A bound computed only from the prompt and format, never its answer."""
    n = generation_problem_size(prompt, spec)
    if spec.task in ("copy", "reverse", "sort", "count", "random_lm"):
        return n + 1
    if spec.task in ("histogram", "histogram2", "most_freq"):
        bos = spec.histogram_bos if spec.task == "histogram" else True
        return n + 1 + int(bos)
    if spec.task == "mode":
        return 2 + (2 * min(n, spec.symbols) if spec.scratchpad != "none" else 0)
    if spec.task == "addition":
        return (2 if spec.index_hints else 1) * (n + 1) + 1
    if spec.task == "parity" and spec.scratchpad != "none":
        return (2 if spec.index_hints else 1) * n + 2
    return 2


def generation_answer(prompt, spec):
    if spec.task == "random_lm":
        raise ValueError("Random payloads have no deterministic raw-prompt oracle")
    generation_problem_size(prompt, spec)  # Validate the raw input alphabet and format.
    if spec.task == "addition":
        return addition_answer(prompt, spec)
    if spec.task == "count":
        start, end = count_interval(prompt, spec)
        return (*range(start, end + 1), v.EOS)
    if spec.task in ("parity", "boolean_and"):
        bits, hints = bit_inputs(prompt, spec)
        if spec.task == "boolean_and":
            return (v.ACCEPT if all(x == v.ONE for x in bits) else v.REJECT, v.EOS)
        parity = 0
        answer = [v.EVEN] if spec.scratchpad != "none" else []
        for index, bit in enumerate(bits):
            parity ^= bit == v.ONE
            if spec.scratchpad == "running" or spec.scratchpad == "ones" and bit == v.ONE:
                if hints:
                    answer.append(hints[index])
                answer.append(v.ODD if parity else v.EVEN)
        if spec.scratchpad == "none":
            answer.append(v.ODD if parity else v.EVEN)
        return (*answer, v.EOS)
    word = sequence_body(prompt, spec)
    counts = Counter(word)
    if spec.task == "copy":
        answer = word
    elif spec.task == "reverse":
        answer = word[::-1]
    elif spec.task == "sort":
        answer = sorted(word)
    elif spec.task in ("histogram", "histogram2"):
        frequency_classes = Counter(counts.values())
        numbers = [counts[token] if spec.task == "histogram" else frequency_classes[counts[token]]
                   for token in word]
        bos = spec.histogram_bos if spec.task == "histogram" else True
        answer = ([v.BOS] if bos else []) + [spec.number_base + count for count in numbers]
    elif spec.task == "most_freq":
        order = sorted(counts, key=lambda x: (-counts[x], word.index(x)))
        answer = [v.BOS, *order, *([v.BOS] * (len(word) - len(order)))]
    elif spec.task == "mode":
        highest = max(counts.values())
        winners = [token for token, count in counts.items() if count == highest]
        if len(winners) != 1:
            raise ValueError("Mode requires a unique most frequent symbol")
        answer = []
        if spec.scratchpad != "none":
            order = sorted(counts, key=lambda x: (counts[x], word.index(x))) if spec.scratchpad == "counts" else counts
            for token in order:
                answer.extend((spec.number_base + counts[token], token))
        answer.append(winners[0])
    else:
        raise ValueError(f"No generated-answer oracle for {spec.task}")
    return (*answer, v.EOS)
