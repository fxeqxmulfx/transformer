"""Parameter names of the historical models, mapped to the modules built here.

`rename` turns a state dictionary of `experiments/gpt_mini.py` GPTMini (also
after `sparsemax_attention.replace_attention`), of the openai/grok reference
`Transformer`, or of the convex MQAR `RopeTransformer` (also as
`SparsemaxTransformer`) into one that `nn.Transformer` loads. Buffers that a
spec determines (position tables, the causal mask) are dropped.
"""

import re

RULES = [
    # GPTMini
    (r"blocks\.(\d+)\.attn\.qkv\.", r"blocks.\1.attention.projections.qkv."),
    (r"blocks\.(\d+)\.attn\.proj\.", r"blocks.\1.attention.output."),
    (r"blocks\.(\d+)\.attn\.log_alpha$", r"blocks.\1.attention.scores.log_alpha"),
    (r"blocks\.(\d+)\.ffn\.w_in\.", r"blocks.\1.ffn.input."),
    (r"blocks\.(\d+)\.ffn\.w_out\.", r"blocks.\1.ffn.output."),
    (r"unembed\.", "readout."),
    # openai/grok reference
    (r"embedding\.", "embed."),
    (r"decoder\.blocks\.(\d+)\.self_attn\.attn_heads\.(\d+)\.Wq\.",
     r"blocks.\1.attention.projections.heads.\2.query."),
    (r"decoder\.blocks\.(\d+)\.self_attn\.attn_heads\.(\d+)\.Wk\.",
     r"blocks.\1.attention.projections.heads.\2.key."),
    (r"decoder\.blocks\.(\d+)\.self_attn\.attn_heads\.(\d+)\.Wv\.",
     r"blocks.\1.attention.projections.heads.\2.value."),
    (r"decoder\.blocks\.(\d+)\.self_attn\.Wo\.", r"blocks.\1.attention.output."),
    (r"decoder\.blocks\.(\d+)\.self_attn_norm\.", r"blocks.\1.attention_norm."),
    (r"decoder\.blocks\.(\d+)\.ffn\.ffn\.0\.", r"blocks.\1.ffn.input."),
    (r"decoder\.blocks\.(\d+)\.ffn\.ffn\.2\.", r"blocks.\1.ffn.output."),
    (r"decoder\.blocks\.(\d+)\.ffn_norm\.", r"blocks.\1.ffn_norm."),
    (r"linear\.", "readout."),
    # convex MQAR RopeTransformer
    (r"blocks\.(\d+)\.norm1\.", r"blocks.\1.attention_norm."),
    (r"blocks\.(\d+)\.norm2\.", r"blocks.\1.ffn_norm."),
    (r"blocks\.(\d+)\.attention\.qkv\.", r"blocks.\1.attention.projections.qkv."),
    (r"blocks\.(\d+)\.mlp\.0\.", r"blocks.\1.ffn.input."),
    (r"blocks\.(\d+)\.mlp\.2\.", r"blocks.\1.ffn.output."),
]
# Names a historical model shares with the current one.
CURRENT = re.compile(r"embed\.weight$|blocks\.\d+\.attention\.output\.|final_norm\.")
DERIVED = {"position_encoding", "self_attn_mask"}


def rename(name):
    for pattern, replacement in RULES:
        renamed, count = re.subn("^" + pattern, replacement, name)
        if count:
            return renamed
    if CURRENT.match(name):
        return name
    raise KeyError(f"No current name for historical parameter {name}")


def import_state(state):
    """A historical state dictionary under current names."""
    return {rename(name): value for name, value in state.items() if name not in DERIVED}
