//! Reading `model.bin`, the flat weight file the analytic compiler emits.
//!
//! The format is fixed by `save_weights` in `transformer_vm/model/weights.py`
//! and read back by `transformer_vm/model/transformer.cpp`.  Everything is
//! little-endian, the matrices are row-major `f64`, and there is no version
//! field, so the header sizes are the only integrity check available.

use std::io::{self, Read};
use std::path::Path;

/// The shapes a weight file declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shapes {
    pub vocab: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ffn: usize,
    pub stop_token: usize,
}

impl Shapes {
    /// The `f64` count of one layer: `qkv`, `out`, `ff_in`, `ff_out`.
    fn per_layer(&self) -> usize {
        let (d, f) = (self.d_model, self.d_ffn);
        3 * d * d + d * d + 2 * f * d + d * f
    }

    fn total_weights(&self) -> usize {
        self.vocab * self.d_model + self.n_layers * self.per_layer() + self.vocab * self.d_model
    }
}

/// One layer's four matrices, row-major, as stored.
#[derive(Clone, Debug)]
pub struct RawLayer {
    /// `[3 * d_model, d_model]`, the query, key and value projections stacked.
    pub qkv: Vec<f64>,
    /// `[d_model, d_model]`.
    pub out: Vec<f64>,
    /// `[2 * d_ffn, d_model]`, the gate and the value halves stacked.
    pub ff_in: Vec<f64>,
    /// `[d_model, d_ffn]`.
    pub ff_out: Vec<f64>,
}

/// A whole weight file, before it is handed to a tensor backend.
#[derive(Clone, Debug)]
pub struct RawModel {
    pub shapes: Shapes,
    pub tokens: Vec<String>,
    /// `[vocab, d_model]`.
    pub embedding: Vec<f64>,
    pub layers: Vec<RawLayer>,
    /// `[vocab, d_model]`.
    pub head: Vec<f64>,
    /// Per head, whether ties resolve to the latest write rather than the mean.
    /// Absent in older files, in which case every head averages.
    pub latest_heads: Option<Vec<Vec<bool>>>,
}

struct Reader<R> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn i32(&mut self) -> io::Result<i32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(self.i32()? as u32)
    }

    fn string(&mut self, len: usize) -> io::Result<String> {
        let mut b = vec![0u8; len];
        self.inner.read_exact(&mut b)?;
        String::from_utf8(b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    fn f64s(&mut self, n: usize) -> io::Result<Vec<f64>> {
        let mut bytes = vec![0u8; n * 8];
        self.inner.read_exact(&mut bytes)?;
        Ok(bytes.as_chunks::<8>().0.iter().map(|&c| f64::from_le_bytes(c)).collect())
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

impl RawModel {
    pub fn load(path: impl AsRef<Path>) -> io::Result<RawModel> {
        let file = std::fs::File::open(path)?;
        RawModel::read(io::BufReader::new(file))
    }

    pub fn read(source: impl Read) -> io::Result<RawModel> {
        let mut r = Reader { inner: source };

        let head: Vec<i32> = (0..6).map(|_| r.i32()).collect::<io::Result<_>>()?;
        if head.iter().take(5).any(|&v| v <= 0) {
            return Err(invalid("weight file declares an empty dimension"));
        }
        let shapes = Shapes {
            vocab: head[0] as usize,
            d_model: head[1] as usize,
            n_layers: head[2] as usize,
            n_heads: head[3] as usize,
            d_ffn: head[4] as usize,
            stop_token: head[5].max(0) as usize,
        };
        if shapes.d_model != 2 * shapes.n_heads {
            return Err(invalid("every head is two-dimensional, so d_model must be 2 * n_heads"));
        }

        let mut tokens = Vec::with_capacity(shapes.vocab);
        for _ in 0..shapes.vocab {
            let len = r.u32()? as usize;
            tokens.push(r.string(len)?);
        }

        let mut w = r.f64s(shapes.total_weights())?;
        let mut take = |n: usize| -> Vec<f64> { w.drain(..n).collect() };

        let (d, f, v) = (shapes.d_model, shapes.d_ffn, shapes.vocab);
        let embedding = take(v * d);
        let layers: Vec<RawLayer> = (0..shapes.n_layers)
            .map(|_| RawLayer {
                qkv: take(3 * d * d),
                out: take(d * d),
                ff_in: take(2 * f * d),
                ff_out: take(d * f),
            })
            .collect();
        let head = take(v * d);

        // The erase lists record which residual slots a reused value overwrites.
        // Neither reference runtime consults them — the subtraction is already
        // folded into the weights — so they are read past, not kept.
        if r.i32().unwrap_or(0) == 1 {
            for _ in 0..shapes.n_layers {
                for _ in 0..2 {
                    let n = r.i32()?;
                    for _ in 0..n {
                        r.i32()?;
                    }
                }
            }
        }

        let latest_heads = if r.i32().unwrap_or(0) == 1 {
            let mut all = Vec::with_capacity(shapes.n_layers);
            for _ in 0..shapes.n_layers {
                let mut layer = Vec::with_capacity(shapes.n_heads);
                for _ in 0..shapes.n_heads {
                    layer.push(r.i32()? == 1);
                }
                all.push(layer);
            }
            Some(all)
        } else {
            None
        };

        Ok(RawModel { shapes, tokens, embedding, layers, head, latest_heads })
    }

    /// The token id for a name, as the runner needs when tokenizing a program.
    pub fn token_id(&self, name: &str) -> Option<usize> {
        self.tokens.iter().position(|t| t == name)
    }
}
