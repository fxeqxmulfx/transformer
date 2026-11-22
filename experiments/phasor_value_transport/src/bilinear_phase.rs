//! Two-register scan of ordered products from dense projections. The local
//! read affects only the current observation. See Phasor.BilinearPhaseRead.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(value: f32, base: u64, index: u32) {
        let address: u64;
        mad.wide.u32(address, index, 4, base);
        ld.global.f32(value, [address]);
    }
    fn store(base: u64, index: u32, value: f32) {
        let address: u64;
        mad.wide.u32(address, index, 4, base);
        st.global.f32([address], value);
    }

    // The shared model passes eight latent coordinates to its frequency
    // matrix. The independent model immediately observes its 32 coordinates.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn forward<const SHARED: bool>(
        read: *mut f32,
        projected: *mut f32,
        length: u32,
    ) {
        let (row, end, lane, base, left_at, right_at, local_at, out): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        mov.u32(row, ctaid.x);
        mul.lo.u32(row, row, length);
        add.u32(end, row, length);
        mov.u32(lane, tid.x);
        let (channels, stride): (u32, u32);
        if SHARED {
            mov.u32(channels, ntid.x);
        } else {
            mov.u32(channels, 32);
        }
        shl.b32(stride, channels, 2);
        let (left_sum, phase, step, left, right, local, angle, sin, cos): (
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
        mov.f32(left_sum, 0.0);
        mov.f32(phase, 0.0);
        label(scan);
        mad.lo.u32(base, row, stride, lane);
        add.u32(left_at, base, channels);
        add.u32(right_at, left_at, channels);
        add.u32(local_at, right_at, channels);
        load(step, projected, base);
        load(left, projected, left_at);
        load(right, projected, right_at);
        load(local, projected, local_at);
        add.rn.f32(phase, phase, step);
        fma.rn.f32(phase, left_sum, right, phase);
        add.rn.f32(angle, phase, local);
        store(projected, left_at, left_sum);
        add.rn.f32(left_sum, left_sum, left);
        if SHARED {
            mad.lo.u32(out, row, channels, lane);
            store(read, out, angle);
        } else {
            store(projected, base, angle);
            sin.approx.f32(sin, angle);
            cos.approx.f32(cos, angle);
            sub.rn.f32(cos, cos, 1.0);
            mad.lo.u32(out, row, 64, lane);
            store(read, out, sin);
            add.u32(out, out, 32);
            store(read, out, cos);
        }
        add.u32(row, row, 1);
        let more: pred;
        setp.lt.u32(more, row, end);
        if more {
            bra(scan);
        }
    }

    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn backward<const SHARED: bool>(
        gradient: *mut f32,
        projected: *const f32,
        dread: *const f32,
        length: u32,
    ) {
        let (row, start, lane, base, left_at, right_at, local_at, out): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        mov.u32(start, ctaid.x);
        mul.lo.u32(start, start, length);
        add.u32(row, start, length);
        mov.u32(lane, tid.x);
        let (channels, stride): (u32, u32);
        if SHARED {
            mov.u32(channels, ntid.x);
        } else {
            mov.u32(channels, 32);
        }
        shl.b32(stride, channels, 2);
        let (dleft, dphase, gs, gc, angle, sin, cos, local, right_grad, right, left_sum): (
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
            f32,
        );
        mov.f32(dleft, 0.0);
        mov.f32(dphase, 0.0);
        label(scan);
        sub.u32(row, row, 1);
        mad.lo.u32(base, row, stride, lane);
        add.u32(left_at, base, channels);
        add.u32(right_at, left_at, channels);
        add.u32(local_at, right_at, channels);
        load(left_sum, projected, left_at);
        load(right, projected, right_at);
        if SHARED {
            mad.lo.u32(out, row, channels, lane);
            load(local, dread, out);
        } else {
            load(angle, projected, base);
            sin.approx.f32(sin, angle);
            cos.approx.f32(cos, angle);
            mad.lo.u32(out, row, 64, lane);
            load(gs, dread, out);
            add.u32(out, out, 32);
            load(gc, dread, out);
            mul.rn.f32(local, gc, sin);
            neg.f32(local, local);
            fma.rn.f32(local, gs, cos, local);
        }
        store(gradient, local_at, local);
        add.rn.f32(dphase, dphase, local);
        store(gradient, base, dphase);
        mul.rn.f32(right_grad, dphase, left_sum);
        store(gradient, right_at, right_grad);
        // The current left token enters future states only.
        store(gradient, left_at, dleft);
        fma.rn.f32(dleft, dphase, right, dleft);
        let more: pred;
        setp.gt.u32(more, row, start);
        if more {
            bra(scan);
        }
    }
}
