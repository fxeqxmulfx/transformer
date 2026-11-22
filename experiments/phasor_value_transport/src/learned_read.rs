//! Coordinate-wise learned mixing of the four contextual streams.
//! See Transformer.Phasor.LearnedStreamRead for the forward and adjoint.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, p: u64, i: u32) {
        let addr: u64;
        mad.wide.u32(addr, i, 4, p);
        ld.global.f32(x, [addr]);
    }
    fn store(p: u64, i: u32, x: f32) {
        let addr: u64;
        mad.wide.u32(addr, i, 4, p);
        st.global.f32([addr], x);
    }
    // The mean and offset are accumulated separately, preserving the original
    // FP32 additions at zero offsets. BACK writes the four state gradients.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn mix<const BACK: bool>(
        out: *mut f32,
        source: *const f32,
        offsets: *const f32,
        width: u32,
        count: u32,
    ) {
        let (i, row, channel, item, at): (u32, u32, u32, u32, u32);
        let past: pred;
        let (block, threads, lane): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(threads, ntid.x);
        mov.u32(lane, tid.x);
        mad.lo.u32(i, block, threads, lane);
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(row, i, width);
        rem.u32(channel, i, width);
        let (mean, correction, x, weight): (f32, f32, f32, f32);
        if BACK {
            load(x, source, i);
        } else {
            mov.f32(mean, 0.0);
            mov.f32(correction, 0.0);
        }
        for s in 0..4 {
            mad.lo.u32(item, s, width, channel);
            mad.lo.u32(at, row, 4, s);
            mad.lo.u32(at, at, width, channel);
            load(weight, offsets, item);
            if BACK {
                add.rn.f32(weight, weight, 0.25);
                mul.rn.f32(weight, x, weight);
                store(out, at, weight);
            } else {
                load(x, source, at);
                add.rn.f32(mean, mean, x);
                fma.rn.f32(correction, weight, x, correction);
            }
        }
        if !BACK {
            fma.rn.f32(mean, mean, 0.25, correction);
            store(out, i, mean);
        }
    }

    // One warp sums all rows for a single parameter. There are no atomic
    // updates and inactive positions contribute their upstream zero.
    pub unsafe extern "ptx-kernel" fn weights_back(
        out: *mut f32,
        states: *const f32,
        upstream: *const f32,
        width: u32,
        rows: u32,
    ) {
        let (item, channel, lane, row, at): (u32, u32, u32, u32, u32);
        mov.u32(item, ctaid.x);
        rem.u32(channel, item, width);
        mov.u32(lane, tid.x);
        mov.u32(row, lane);
        let (sum, x, g): (f32, f32, f32);
        mov.f32(sum, 0.0);
        let done: pred;
        label(rows_loop);
        setp.ge.u32(done, row, rows);
        if done {
            bra(reduce);
        }
        shl.b32(at, row, 2);
        mad.lo.u32(at, at, width, item);
        load(x, states, at);
        mad.lo.u32(at, row, width, channel);
        load(g, upstream, at);
        fma.rn.f32(sum, g, x, sum);
        add.u32(row, row, 32);
        bra(rows_loop);
        label(reduce);
        for j in 0..5 {
            shfl.sync.bfly.b32(x, sum, 1 << j, 31, 0xffffffff);
            add.rn.f32(sum, sum, x);
        }
        setp.ne.u32(done, lane, 0);
        if done {
            ret();
        }
        store(out, item, sum);
    }
}
