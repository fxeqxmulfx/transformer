"""Exact rational premise checks for the complete trace, manuscript L2.

Decimal square roots only propose witnesses. Every scale is checked again
by rational arithmetic; no Decimal or floating transcendental is trusted.
Analytic consequences and trace composition are independently proved in Lean.
"""

from decimal import Decimal, localcontext
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

from backend_contract_arithmetic import exact, exp_polynomial, rotation_polynomial


ROOT = Path(__file__).resolve().parent
ETA = Q(1, 10**6)
EPS = Q(1, 8388608)
SHAPES = {name: [1, 2, width] for name, width in (
    ("fused", 12), ("heads", 4), ("projected", 4), ("mid", 4), ("activated", 11),
    ("product", 11), ("ffn_projected", 4), ("output", 4), ("initial", 4), ("pre", 4),
    ("attn_output", 4), ("ffn_pre", 4), ("ffn_output", 4), ("normalized", 4),
    ("gate", 11), ("value", 11))}
SHAPES.update({name: [1, 1, 2, 4] for name in ("direction", "average", "qnorm", "knorm", "query", "key")})
SHAPES.update(scores=[1, 1, 2, 2], logits=[1, 2, 3])
PARAMETERS = {"token_embedding_table.weight": [3, 4], "blocks.0.sa.c_attn.weight": [12, 4],
              "blocks.0.sa.c_proj.weight": [4, 4], "blocks.0.ffwd.c_fc.weight": [22, 4],
              "blocks.0.ffwd.c_proj.weight": [4, 11]}


def rows(record, shape):
    if record["shape"] != shape:
        raise ValueError("wrong tensor shape")
    count = 1
    for n in shape:
        count *= n
    if len(record["values"]) != count:
        raise ValueError("wrong tensor length")
    values = [exact(x) for x in record["values"]]
    width = shape[-1]
    return [values[i:i + width] for i in range(0, count, width)]


def dot(x, y):
    return sum((a * b for a, b in zip(x, y, strict=True)), Q(0))


def scale_witness(square):
    with localcontext() as context:
        context.prec = 60
        value = (Decimal(square.numerator) / Decimal(square.denominator)).sqrt()
        return Q(int((1 / value) * 10**15), 10**15)


def exponential_witness(x):
    return Q(int(exp_polynomial(x) * 10**15), 10**15)


def check(report):
    """Check every ModelAccurate conjunct; return the explicit proof witnesses."""
    if report["config"] != dict(width=4, heads=1, depth=1, ffn=11, vocab=3, length=2, rope_base=10000):
        raise ValueError("wrong architecture")
    if report["tokens"] != [0, 1] or report["rms_epsilon"] != "1/8388608" or report["l2_epsilon"] != "1/1000000000000":
        raise ValueError("wrong tokens or epsilon")
    if set(report["trace"]) != set(SHAPES) or set(report["parameters"]) != set(PARAMETERS) or set(report["tables"]) != {"cos", "sin"}:
        raise ValueError("missing or extra operation/parameter/table")
    t = {name: rows(report["trace"][name], shape) for name, shape in SHAPES.items()}
    p = {name: rows(report["parameters"][name], shape) for name, shape in PARAMETERS.items()}
    tables = {name: rows(report["tables"][name], [2, 2]) for name in ("cos", "sin")}
    checked, witnesses = {}, {}

    def require(label, condition):
        if not condition:
            raise ValueError(f"{label}: certificate failed")

    def box(label, actual, expected, allowance=ETA):
        errors = [abs(a - b) for a, b in zip(actual, expected, strict=True)]
        require(label, max(errors) <= allowance)
        checked[label] = str(max(errors))

    def normalize(name, inputs, l2=False):
        scales = []
        for i, x in enumerate(inputs):
            square = dot(x, x) if l2 else dot(x, x) / 4 + EPS
            require(name, not l2 or square >= Q(1, 10**24))
            a = scale_witness(square)
            require(name, a >= 0 and abs(a*a*square - 1) <= ETA)
            box(f"{name}.{i}", t[name][i], [a * c for c in x])
            scales.append(str(a))
        witnesses[name] = scales

    embedding = p["token_embedding_table.weight"]
    normalize("initial", [embedding[i] for i in report["tokens"]])
    normalize("pre", t["initial"])
    qkv = p["blocks.0.sa.c_attn.weight"]
    for i in range(2):
        box(f"fused.{i}", t["fused"][i], [dot(row, t["pre"][i]) for row in qkv])
    q, k, v = ([row[start:start+4] for row in t["fused"]] for start in (0, 4, 8))
    normalize("qnorm", q)
    normalize("knorm", k)
    for i in range(2):
        phases = (Q(i), Q(i, 100))
        box(f"cos.{i}", tables["cos"][i], [rotation_polynomial(x)[0] for x in phases])
        box(f"sin.{i}", tables["sin"][i], [rotation_polynomial(x)[1] for x in phases])
        for name, norm in (("query", "qnorm"), ("key", "knorm")):
            x, c, s = t[norm][i], tables["cos"][i], tables["sin"][i]
            box(f"{name}.{i}", t[name][i], [c[j]*x[j]-s[j]*x[j+2] for j in range(2)] +
                [s[j]*x[j]+c[j]*x[j+2] for j in range(2)])
        box(f"scores.{i}", t["scores"][i], [dot(t["query"][i], t["key"][j])/2 for j in range(2)])
        masses = [exponential_witness(s) if j <= i else Q(0) for j, s in enumerate(t["scores"][i])]
        require("average", sum(masses) >= Q(1, 2))
        for j in range(i+1):
            require("average.exp", abs(t["scores"][i][j]) <= 1 and
                    abs(masses[j] - exp_polynomial(t["scores"][i][j])) <= Q(1, 10**10))
        require("average.values", all(abs(c) <= 1 for row in v for c in row))
        box(f"average.{i}", t["average"][i], [sum(masses[j]*v[j][c] for j in range(2))/sum(masses) for c in range(4)])
        witnesses[f"average.{i}"] = [str(x) for x in masses]
    normalize("direction", v, l2=True)
    for i in range(2):
        a, u = t["average"][i], t["direction"][i]
        box(f"heads.{i}", t["heads"][i], [a[c]-dot(a,u)*u[c] if i else Q(0) for c in range(4)])
        box(f"projected.{i}", t["projected"][i], [dot(row, t["heads"][i]) for row in p["blocks.0.sa.c_proj.weight"]])
    normalize("attn_output", t["projected"])
    for i in range(2):
        box(f"mid.{i}", t["mid"][i], [x+y for x,y in zip(t["initial"][i], t["attn_output"][i], strict=True)])
    normalize("ffn_pre", t["mid"])
    for i in range(2):
        fc = p["blocks.0.ffwd.c_fc.weight"]
        box(f"gate.{i}", t["gate"][i], [dot(row, t["ffn_pre"][i]) for row in fc[:11]])
        box(f"value.{i}", t["value"][i], [dot(row, t["ffn_pre"][i]) for row in fc[11:]])
        masses = [exponential_witness(-x) for x in t["gate"][i]]
        for x,e in zip(t["gate"][i], masses, strict=True):
            require("activated.exp", abs(x) <= 1 and e >= 0 and abs(e-exp_polynomial(-x)) <= Q(1,10**10))
        box(f"activated.{i}", t["activated"][i], [x/(1+e) for x,e in zip(t["gate"][i], masses, strict=True)])
        witnesses[f"activated.{i}"] = [str(e) for e in masses]
        box(f"product.{i}", t["product"][i], [a*b for a,b in zip(t["activated"][i], t["value"][i], strict=True)])
        box(f"ffn_projected.{i}", t["ffn_projected"][i], [dot(row, t["product"][i]) for row in p["blocks.0.ffwd.c_proj.weight"]])
    normalize("ffn_output", t["ffn_projected"])
    for i in range(2):
        box(f"output.{i}", t["output"][i], [x+y for x,y in zip(t["mid"][i], t["ffn_output"][i], strict=True)])
    normalize("normalized", t["output"])
    for i in range(2):
        box(f"logits.{i}", t["logits"][i], [dot(t["normalized"][i], e) for e in embedding])
    return {"maximum_rational_residuals": checked, "witnesses": witnesses,
            "uniform_operation_allowance": "1/10000", "table_allowance": "1/500000"}


def main():
    capture = ROOT / "model_trace_capture.json"
    result = check(json.loads(capture.read_text()))
    result.update(capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT / "model_trace_certificates.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Checked {len(result['maximum_rational_residuals'])} complete-trace vector/scalar sites")


if __name__ == "__main__":
    main()
