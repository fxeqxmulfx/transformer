use super::attention::{Attention, Convolution};
use super::*;

struct Experts {
    router: Tensor,
    gate_up: Tensor,
    down: Tensor,
}
impl Experts {
    fn new(p: &nn::Path, d: i64) -> Self {
        Self {
            router: weights(p, "router", &[d, EXPERTS]),
            gate_up: weights(p, "gate_up", &[d, EXPERTS * d]),
            down: weights(p, "down", &[EXPERTS, d / 2, d]),
        }
    }
    fn forward(&self, x: &Tensor) -> Tensor {
        let [b, t, w] = <[i64; 3]>::try_from(x.size()).unwrap();
        let x = x.reshape([-1, w]);
        let scores = x.matmul(&self.router).sigmoid();
        let (selected, indices) = scores.topk(TOP, -1, true, false);
        let selected = &selected / (selected.sum_dim_intlist(-1, true, Kind::Float) + 1e-20) * 8.;
        if RELEASE {
            return super::routing::dispatch(&x, &self.gate_up, &self.down, &indices, &selected)
                .view([b, t, w]);
        }
        let mix = Tensor::zeros_like(&scores).scatter(-1, &indices, &selected);
        // Evaluate the small bank as two GEMMs; only the selected experts contribute.
        let gate_up = x.matmul(&self.gate_up).view([-1, EXPERTS, w]);
        let gate = gate_up.narrow(-1, 0, w / 2);
        let up = gate_up.narrow(-1, w / 2, w / 2);
        let hidden = softcap(&gate, 4.) * gate.sigmoid() * softcap(&up, 25.);
        let values = hidden.transpose(0, 1).bmm(&self.down).transpose(0, 1);
        (values * mix.unsqueeze(-1))
            .sum_dim_intlist(1, false, Kind::Float)
            .view([b, t, w])
    }
}

pub(super) struct Layer {
    boundary: bool,
    attention_query: Option<Tensor>,
    mlp_query: Tensor,
    query_gain: Tensor,
    key_gain: Option<Tensor>,
    value_gain: Option<Tensor>,
    mlp_gain: Tensor,
    pub(super) reader: Option<(Tensor, [Gate; 4])>,
    attn: Attention,
    experts: Experts,
    post_attn: Convolution,
    post_mlp: Convolution,
}
impl Layer {
    pub(super) fn new(p: &nn::Path, i: usize, w: i64, heads: i64) -> Self {
        let full = (i + 1).is_multiple_of(GROUP);
        Self {
            boundary: i.is_multiple_of(GROUP),
            attention_query: (i > 0)
                .then(|| p.zeros("attention_query", &[if full { 3 } else { 1 }, w])),
            mlp_query: p.zeros("mlp_query", &[1, w]),
            query_gain: p.ones("query_gain", &[w]),
            key_gain: full.then(|| p.ones("key_gain", &[w])),
            value_gain: full.then(|| p.ones("value_gain", &[w])),
            mlp_gain: p.ones("mlp_gain", &[w]),
            reader: (BEGIN..END).contains(&i).then(|| {
                (
                    p.zeros("message_queries", &[4, w]),
                    std::array::from_fn(|r| Gate::new(&(p / format!("gate{r}")), w)),
                )
            }),
            attn: Attention::new(&(p / "attn"), w, heads, full),
            experts: Experts::new(&(p / "experts"), w),
            post_attn: Convolution::new(&(p / "post_attn"), w, None),
            post_mlp: Convolution::new(&(p / "post_mlp"), w, None),
        }
    }
    fn merge(&self, role: usize, x: Tensor, messages: Option<&Tensor>) -> Tensor {
        match messages {
            Some(m) => {
                self.reader.as_ref().unwrap().1[role].forward(&x, &m.select(-2, role as i64))
            }
            None => x,
        }
    }
    pub(super) fn apply(
        &self,
        h: Tensor,
        blocks: &mut Vec<Tensor>,
        messages: Option<&Tensor>,
    ) -> Tensor {
        let input = self
            .attention_query
            .as_ref()
            .map(|q| read_states(blocks, &h, q));
        let component = |r: i64| match &input {
            Some(a) => a.select(-2, if a.size()[2] == 1 { 0 } else { r }),
            None => h.shallow_clone(),
        };
        let q = rms(&self.merge(0, component(0), messages)) * &self.query_gain;
        let k = rms(&self.merge(1, component(1), messages))
            * self.key_gain.as_ref().unwrap_or(&self.query_gain);
        let v = rms(&self.merge(2, component(2), messages))
            * self.value_gain.as_ref().unwrap_or(&self.query_gain);
        let attention = self.attn.forward(&q, &k, &v);
        let attention = self.post_attn.apply(&attention, &attention);
        let residual = if self.boundary {
            blocks.push(h);
            attention
        } else {
            h + attention
        };
        let mix = read_states(blocks, &residual, &self.mlp_query).squeeze_dim(-2);
        let input = rms(&self.merge(3, mix, messages)) * &self.mlp_gain;
        let mlp = self.experts.forward(&input);
        residual + self.post_mlp.apply(&mlp, &mlp)
    }
}
