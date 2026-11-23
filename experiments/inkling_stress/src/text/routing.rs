//! Batched computation on the selected experts. The address of a route is
//! discrete, while gradients still flow through its weight and expert output.
use super::*;

pub(super) fn dispatch(
    x: &Tensor,
    gate_up: &Tensor,
    down: &Tensor,
    indices: &Tensor,
    mix: &Tensor,
) -> Tensor {
    let [tokens, width] = <[i64; 2]>::try_from(x.size()).unwrap();
    let experts = down.size()[0];
    let k = indices.size()[1];
    let picked = Vec::<i64>::try_from(&indices.reshape([-1])).unwrap();
    let tile = picked.len().div_ceil(experts as usize).clamp(1, 32);
    let mut assigned = vec![Vec::new(); experts as usize];
    for (slot, e) in picked.into_iter().enumerate() {
        assigned[e as usize].push(slot as i64);
    }
    let mut expert_ids = Vec::new();
    let mut inputs = Vec::new();
    let mut valid = Vec::new();
    let mut destinations = Vec::new();
    let mut routes = Vec::new();
    for (e, slots) in assigned.iter().enumerate() {
        for chunk in slots.chunks(tile) {
            expert_ids.push(e as i64);
            for &slot in chunk {
                valid.push(inputs.len() as i64);
                inputs.push(slot / k);
                destinations.push(slot / k);
                routes.push(slot);
            }
            inputs.resize(inputs.len() + tile - chunk.len(), 0);
        }
    }
    let experts_used = Tensor::from_slice(&expert_ids);
    let input = x
        .index_select(0, &Tensor::from_slice(&inputs))
        .view([-1, tile as i64, width]);
    let matrices = gate_up
        .view([width, experts, width])
        .permute([1, 0, 2])
        .index_select(0, &experts_used);
    let two = input.bmm(&matrices);
    let gate = two.narrow(-1, 0, width / 2);
    let up = two.narrow(-1, width / 2, width / 2);
    let hidden = softcap(&gate, 4.) * gate.sigmoid() * softcap(&up, 25.);
    let values = hidden
        .bmm(&down.index_select(0, &experts_used))
        .view([-1, width])
        .index_select(0, &Tensor::from_slice(&valid));
    let weights = mix
        .reshape([-1])
        .index_select(0, &Tensor::from_slice(&routes))
        .unsqueeze(1);
    Tensor::zeros([tokens, width], (x.kind(), x.device())).index_add(
        0,
        &Tensor::from_slice(&destinations),
        &(values * weights),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_routes_match_a_dense_mixture_and_all_four_gradients() {
        tch::set_num_threads(1);
        let leaf = |shape: &[i64]| {
            (Tensor::arange(
                shape.iter().product::<i64>(),
                (Kind::Float, tch::Device::Cpu),
            ) * 0.07)
                .sin()
                .reshape(shape)
                .set_requires_grad(true)
        };
        let (x, g, u, w) = (
            leaf(&[7, 4]),
            leaf(&[4, 20]),
            leaf(&[5, 2, 4]),
            leaf(&[7, 2]),
        );
        let ids =
            Tensor::from_slice(&[0i64, 3, 3, 0, 1, 3, 3, 1, 0, 1, 1, 3, 3, 0]).reshape([7, 2]);
        let packed = dispatch(&x, &g, &u, &ids, &w);
        let both = x.matmul(&g).view([7, 5, 4]);
        let gate = both.narrow(-1, 0, 2);
        let hidden = softcap(&gate, 4.) * gate.sigmoid() * softcap(&both.narrow(-1, 2, 2), 25.);
        let all = hidden.transpose(0, 1).bmm(&u).transpose(0, 1);
        let mixed = Tensor::zeros([7, 5], (Kind::Float, tch::Device::Cpu)).scatter(-1, &ids, &w);
        let dense = (all * mixed.unsqueeze(-1)).sum_dim_intlist(1, false, Kind::Float);
        let leaves = [&x, &g, &u, &w];
        let g1 = Tensor::run_backward(&[packed.sum(Kind::Float)], &leaves, false, false);
        let g2 = Tensor::run_backward(&[dense.sum(Kind::Float)], &leaves, false, false);
        assert!((&packed - &dense).abs().max().double_value(&[]) < 2e-6);
        for (a, b) in g1.iter().zip(&g2) {
            assert!((a - b).abs().max().double_value(&[]) < 5e-6)
        }
    }
}
