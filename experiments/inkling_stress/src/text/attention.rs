use super::*;

/// Kernel taps are indexed by lag. Padding is confined to the left boundary.
pub(super) struct Convolution {
    taps: Tensor,
    dynamic: Option<Tensor>,
    heads: i64,
}
impl Convolution {
    pub(super) fn new(p: &nn::Path, channels: i64, dynamic: Option<(i64, i64)>) -> Self {
        let heads = dynamic.map_or(1, |(_, h)| h);
        let taps = if dynamic.is_some() {
            p.zeros("taps", &[channels, 4])
        } else {
            weights(p, "taps", &[channels, 4])
        };
        Self {
            taps,
            dynamic: dynamic.map(|(width, h)| weights(p, "dynamic", &[width, 4 * h])),
            heads,
        }
    }
    pub(super) fn apply(&self, x: &Tensor, input: &Tensor) -> Tensor {
        let [b, t, d] = <[i64; 3]>::try_from(x.size()).unwrap();
        let windows = x
            .transpose(1, 2)
            .constant_pad_nd([3, 0])
            .unfold(-1, 4, 1)
            .flip([-1])
            .transpose(1, 2);
        let (sum, denominator) = if let Some(w) = &self.dynamic {
            let taps = self.taps.view([self.heads, d / self.heads, 4])
                + input.matmul(w).view([b, t, self.heads, 1, 4]);
            let applied = (windows.reshape([b, t, self.heads, d / self.heads, 4]) * taps)
                .sum_dim_intlist(-1, false, Kind::Float)
                .view([b, t, d]);
            (applied, 3.)
        } else {
            (
                (windows * &self.taps).sum_dim_intlist(-1, false, Kind::Float),
                5f64.sqrt(),
            )
        };
        (x + sum) / denominator
    }
}

fn clip(x: &Tensor, threshold: f64) -> Tensor {
    let norm = x.norm_scalaropt_dim(2., [-1], true);
    x * ((norm + EPS).reciprocal() * threshold).clamp_max(1.)
}

pub(super) struct Attention {
    heads: i64,
    kv_heads: i64,
    head_dim: i64,
    full: bool,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    out_gate: Tensor,
    sink: Tensor,
    q_scale: Tensor,
    rel_scale: Tensor,
    rel: Tensor,
    bank: Tensor,
    keys: Convolution,
    values: Convolution,
}
impl Attention {
    pub(super) fn new(p: &nn::Path, width: i64, heads: i64, full: bool) -> Self {
        let kv_heads = 1;
        let head_dim = width / heads;
        let channels = kv_heads * head_dim;
        let dynamic = full.then_some((width, kv_heads));
        Self {
            heads,
            kv_heads,
            head_dim,
            full,
            q: weights(p, "q", &[width, width]),
            k: weights(p, "k", &[width, channels]),
            v: weights(p, "v", &[width, channels]),
            out: weights(p, "out", &[width, width]),
            out_gate: weights(p, "out_gate", &[width, width]),
            sink: weights(p, "sink", &[width, heads]),
            q_scale: p.ones("q_scale", &[heads, head_dim]),
            rel_scale: p.ones("rel_scale", &[heads, 16]),
            rel: weights(p, "rel", &[width, heads * 16]),
            bank: weights(p, "distance", &[16, 1024]),
            keys: Convolution::new(&(p / "keys"), channels, dynamic),
            values: Convolution::new(&(p / "values"), channels, dynamic),
        }
    }
    pub(super) fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
        let [b, t, w] = <[i64; 3]>::try_from(query.size()).unwrap();
        let (h, d) = (self.heads, self.head_dim);
        let positions = Tensor::arange(t, (Kind::Int64, query.device()));
        let delta = positions.unsqueeze(1) - positions.unsqueeze(0);
        let log_scale = if self.full && t > 8192 {
            Some(
                (((&positions + 1).to_kind(Kind::Float) / 8192.)
                    .clamp_min(1.)
                    .log()
                    * 0.1
                    + 1.)
                    .view([1, t, 1, 1]),
            )
        } else {
            None
        };
        let scale_position = |x: Tensor| match &log_scale {
            Some(s) => x * s,
            None => x,
        };
        let q = scale_position(clip(
            &(query.matmul(&self.q).view([b, t, h, d]) * &self.q_scale),
            2. * (d as f64).sqrt(),
        ))
        .transpose(1, 2);
        let k = self
            .keys
            .apply(&key.matmul(&self.k), key)
            .view([b, t, self.kv_heads, d]);
        let v = self
            .values
            .apply(&value.matmul(&self.v), value)
            .view([b, t, self.kv_heads, d]);
        let repeat = |s: Tensor| s.transpose(1, 2).expand([b, h, t, d], false);
        let scores = q.matmul(&repeat(clip(&k, 2. * (d as f64).sqrt())).transpose(-2, -1))
            / (d as f64).sqrt();
        let extent = t.min(1024);
        let relative = scale_position(clip(
            &(query.matmul(&self.rel).view([b, t, h, 16]) * &self.rel_scale),
            8.,
        ))
        .matmul(&self.bank.narrow(1, 0, extent))
        .transpose(1, 2);
        let indices = delta
            .clamp(0, extent - 1)
            .view([1, 1, t, t])
            .expand([b, h, t, t], false);
        let bias = relative
            .gather(-1, &indices, false)
            .masked_fill(&delta.ge(1024), 0.);
        let hidden = if self.full {
            delta.lt(0)
        } else {
            delta.lt(0).logical_or(&delta.ge(1024))
        };
        let visible = (scores + bias).masked_fill(&hidden, f64::NEG_INFINITY);
        let sink = query.matmul(&self.sink).transpose(1, 2).unsqueeze(-1);
        let weights = Tensor::cat(&[visible, sink], -1)
            .softmax(-1, Kind::Float)
            .narrow(-1, 0, t);
        let values = weights
            .matmul(&repeat(rms(&v)))
            .transpose(1, 2)
            .reshape([b, t, w]);
        (values * query.matmul(&self.out_gate).sigmoid()).matmul(&self.out)
    }
}
