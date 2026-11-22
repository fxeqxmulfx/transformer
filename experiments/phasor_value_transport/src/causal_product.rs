//! Learned coordinate products at the normalized embedding input.
//! Transformer.Phasor.CausalProductInput gives the forward and finite-sequence adjoint.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, i: u32) {
        let addr: u64;
        mad.wide.u32(addr, i, 4, p);
        ld.global.f32(v, [addr]);
    }
    fn store(p: u64, i: u32, v: f32) {
        let addr: u64;
        mad.wide.u32(addr, i, 4, p);
        st.global.f32([addr], v);
    }
    // Projected rows are [query, current, lag], each with 32 coordinates.
    fn operands(a: f32, b: f32, projected: u64, query: u32, position: u32) {
        let at: u32;
        load(a, projected, query);
        add.u32(at, query, 32);
        load(b, projected, at);
        let has_previous: pred;
        setp.ne.u32(has_previous, position, 0);
        if has_previous {
            let previous: f32;
            sub.u32(at, query, 32);
            load(previous, projected, at);
            add.rn.f32(b, b, previous);
        }
    }
    pub unsafe extern "ptx-kernel" fn forward(
        product: *mut f32,
        projected: *const f32,
        length: u32,
    ) {
        let (row, lane, position, at): (u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        rem.u32(position, row, length);
        mad.lo.u32(at, row, 96, lane);
        let (a, b): (f32, f32);
        operands(a, b, projected, at, position);
        mul.rn.f32(a, a, b);
        mad.lo.u32(at, row, 32, lane);
        store(product, at, a);
    }
    // A thread writes all three cotangents of its own row. The lag reads
    // the next product's cotangent only within this sequence.
    pub unsafe extern "ptx-kernel" fn backward(
        dprojected: *mut f32,
        projected: *const f32,
        upstream: *const f32,
        length: u32,
    ) {
        let (row, lane, position, at, idx, next): (u32, u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        rem.u32(position, row, length);
        mad.lo.u32(at, row, 96, lane);
        mad.lo.u32(idx, row, 32, lane);
        let (a, b, g, grad): (f32, f32, f32, f32);
        operands(a, b, projected, at, position);
        load(g, upstream, idx);
        mul.rn.f32(grad, g, b);
        store(dprojected, at, grad);
        add.u32(at, at, 32);
        mul.rn.f32(grad, g, a);
        store(dprojected, at, grad);
        add.u32(at, at, 32);
        add.u32(position, position, 1);
        let has_next: pred;
        setp.lt.u32(has_next, position, length);
        mov.f32(grad, 0.0);
        if has_next {
            add.u32(next, at, 32);
            load(a, projected, next);
            add.u32(next, idx, 32);
            load(g, upstream, next);
            mul.rn.f32(grad, g, a);
        }
        store(dprojected, at, grad);
    }
}
