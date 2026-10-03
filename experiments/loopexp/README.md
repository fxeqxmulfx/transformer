# Looped against flat pre-norm GPT: how quantization error propagates

Does a model that loops its blocks amplify the error of a quantized block
more than a flat stack does? [`loopexp.py`](loopexp.py) trains a
scaled-down parameter-golf block on FineWeb sp1024, looped or flat, and
evaluates it with every block quantized to 8, 6, 5 and 4 bits. Then, layer
by layer, it measures the stream norm, the error a quantized block injects
at the clean state, the propagated error, the gain along the actual error
direction and the top singular value of the layer's Jacobian, and compares
the propagated error with the Gronwall recursion over those gains.

The block has resid_mix with x0 injection, per-channel attention and MLP
scales, zero-initialized output maps, q/k RMS norm, a relu² MLP, tied
embeddings and a logit softcap, trained under Muon, without U-Net skips.
The script predates the [lab](../../python/README.md) and is not written in
its language: a run is configured by flags.

## Running

It needs CUDA, and parameter golf's FineWeb sp1024 data under `$PG_DATA`,
by default `~/.cache/param-golf-data`:
`datasets/fineweb10B_sp1024/fineweb_{train,val}_000000.bin` and
`tokenizers/fineweb_1024_bpe.model`. It imports numpy, sentencepiece and
torch, the dependencies of the root `pyproject.toml`, which was added for
it; with that environment active:

```sh
python experiments/loopexp/loopexp.py --name looped --virtual 0,1,2,3,2,3,2,3,4,5
```

`--virtual` lists the physical block of every virtual layer, so a repeated
index loops its block; `--help` lists the other flags. A run writes to
`runs/<name>/` here, which git ignores, or under `--out`.

The repository records no run of it.
