//! Sixteen independent four-dimensional Householder states. Four contiguous
//! warp lanes own a state. A learned slope supplies each nonconstant normal coordinate.
pub const CHANNELS: usize = 16;
pub const INPUTS: usize = 3 * CHANNELS;
pub const FEATURES: usize = 4 * CHANNELS;

pub fn parameter_count(width: usize) -> usize {
    (4 + INPUTS + FEATURES) * width + FEATURES
}

#[kernite_macros::kernels]
pub mod kernels {
    fn load(value: f32, base: u64, index: u32) {
        let ptr: u64;
        mad.wide.u32(ptr, index, 4, base);
        ld.global.f32(value, [ptr]);
    }
    fn store(base: u64, index: u32, value: f32) {
        let ptr: u64;
        mad.wide.u32(ptr, index, 4, base);
        st.global.f32([ptr], value);
    }
    fn dot4(sum: f32, a: f32, b: f32) {
        let other: f32;
        mul.rn.f32(sum, a, b);
        for k in 0..2 {
            shfl.sync.bfly.b32(other, sum, 1 << k, 31, 0xffffffff);
            add.rn.f32(sum, sum, other);
        }
    }
    fn normal(value: f32, input_at: u32, projected: u64, row: u32, lane: u32) {
        let (channel, coord): (u32, u32);
        shr.u32(channel, lane, 2);
        and.b32(coord, lane, 3);
        mov.f32(value, 1.0);
        mov.u32(input_at, 0);
        let slope: pred;
        setp.ne.u32(slope, coord, 0);
        if slope {
            mad.lo.u32(input_at, row, 48, coord);
            mad.lo.u32(input_at, channel, 3, input_at);
            sub.u32(input_at, input_at, 1);
            load(value, projected, input_at);
        }
    }

    pub unsafe extern "ptx-kernel" fn forward(
        read: *mut f32,
        previous: *mut f32,
        projected: *const f32,
        initial: *const f32,
        length: u32,
    ) {
        let (row, end, lane, at, input_at): (u32, u32, u32, u32, u32);
        mov.u32(lane, tid.x);
        mov.u32(row, ctaid.x);
        mul.lo.u32(row, row, length);
        add.u32(end, row, length);
        let (state, start, n, d, u, centered): (f32, f32, f32, f32, f32, f32);
        load(start, initial, lane);
        mov.f32(state, start);
        label(scan);
        mad.lo.u32(at, row, 64, lane);
        store(previous, at, state);
        normal(n, input_at, projected, row, lane);
        dot4(d, n, n);
        dot4(u, n, state);
        div.rn.f32(u, u, d);
        mul.rn.f32(u, u, -2.0);
        fma.rn.f32(state, u, n, state);
        sub.rn.f32(centered, state, start);
        store(read, at, centered);
        add.u32(row, row, 1);
        let more: pred;
        setp.lt.u32(more, row, end);
        if more {
            bra(scan);
        }
    }

    pub unsafe extern "ptx-kernel" fn backward(
        dprojected: *mut f32,
        partial_initial: *mut f32,
        previous: *const f32,
        projected: *const f32,
        dread: *const f32,
        length: u32,
    ) {
        let (seq, row, first, lane, coord, at, input_at): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(seq, ctaid.x);
        mul.lo.u32(first, seq, length);
        add.u32(row, first, length);
        mov.u32(lane, tid.x);
        and.b32(coord, lane, 3);
        let (g, sum, local, state, n, d, u, v, normal_grad, coeff): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        mov.f32(g, 0.0);
        mov.f32(sum, 0.0);
        label(scan);
        sub.u32(row, row, 1);
        mad.lo.u32(at, row, 64, lane);
        load(local, dread, at);
        add.rn.f32(g, g, local);
        add.rn.f32(sum, sum, local);
        load(state, previous, at);
        normal(n, input_at, projected, row, lane);
        dot4(d, n, n);
        dot4(u, n, state);
        dot4(v, n, g);
        div.rn.f32(u, u, d);
        div.rn.f32(v, v, d);
        // Only three normal coordinates come from the learned projection.
        let slope: pred;
        setp.ne.u32(slope, coord, 0);
        if slope {
            mul.rn.f32(normal_grad, u, g);
            fma.rn.f32(normal_grad, v, state, normal_grad);
            mul.rn.f32(coeff, u, v);
            mul.rn.f32(coeff, coeff, -2.0);
            fma.rn.f32(normal_grad, coeff, n, normal_grad);
            mul.rn.f32(normal_grad, normal_grad, -2.0);
            store(dprojected, input_at, normal_grad);
        }
        mul.rn.f32(v, v, -2.0);
        fma.rn.f32(g, v, n, g);
        let more: pred;
        setp.gt.u32(more, row, first);
        if more {
            bra(scan);
        }
        // Every centered read also subtracts the shared initial state directly.
        sub.rn.f32(g, g, sum);
        mad.lo.u32(at, seq, 64, lane);
        store(partial_initial, at, g);
    }
}
