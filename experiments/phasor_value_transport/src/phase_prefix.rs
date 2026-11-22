//! Each lane scans one projected channel. The single-state rotation and
//! reflection share the sine/cosine read and have opposite carry signs.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, i: u32) {
        let at: u64;
        mad.wide.u32(at, i, 4, p);
        ld.global.f32(v, [at]);
    }
    fn store(p: u64, i: u32, v: f32) {
        let at: u64;
        mad.wide.u32(at, i, 4, p);
        st.global.f32([at], v);
    }

    // The input projection buffer is reused for the exact FP32 prefix
    // angles needed by the pullback. Each angle has a single writer.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn forward<const REFLECT: bool>(
        read: *mut f32,
        angles: *mut f32,
        length: u32,
    ) {
        let (row, end, lane, angle_at, read_at): (u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mul.lo.u32(row, row, length);
        add.u32(end, row, length);
        mov.u32(lane, tid.x);
        let (a, increment, s, c): (f32, f32, f32, f32);
        mov.f32(a, 0.0);
        label(scan);
        mad.lo.u32(angle_at, row, 32, lane);
        load(increment, angles, angle_at);
        if REFLECT {
            sub.rn.f32(a, increment, a);
        } else {
            add.rn.f32(a, a, increment);
        }
        store(angles, angle_at, a);
        sin.approx.f32(s, a);
        cos.approx.f32(c, a);
        sub.rn.f32(c, c, 1.0);
        mad.lo.u32(read_at, row, 64, lane);
        store(read, read_at, s);
        add.u32(read_at, read_at, 32);
        store(read, read_at, c);
        add.u32(row, row, 1);
        let more: pred;
        setp.lt.u32(more, row, end);
        if more {
            bra(scan);
        }
    }

    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn backward<const REFLECT: bool>(
        dincrements: *mut f32,
        angles: *const f32,
        dread: *const f32,
        length: u32,
    ) {
        let (row, start, lane, angle_at, read_at): (u32, u32, u32, u32, u32);
        mov.u32(start, ctaid.x);
        mul.lo.u32(start, start, length);
        add.u32(row, start, length);
        mov.u32(lane, tid.x);
        let (a, s, c, gs, gc, local, carry): (f32, f32, f32, f32, f32, f32, f32);
        mov.f32(carry, 0.0);
        label(scan);
        sub.u32(row, row, 1);
        mad.lo.u32(angle_at, row, 32, lane);
        load(a, angles, angle_at);
        sin.approx.f32(s, a);
        cos.approx.f32(c, a);
        mad.lo.u32(read_at, row, 64, lane);
        load(gs, dread, read_at);
        add.u32(read_at, read_at, 32);
        load(gc, dread, read_at);
        mul.rn.f32(local, gc, s);
        neg.f32(local, local);
        fma.rn.f32(local, gs, c, local);
        if REFLECT {
            sub.rn.f32(carry, local, carry);
        } else {
            add.rn.f32(carry, carry, local);
        }
        store(dincrements, angle_at, carry);
        let more: pred;
        setp.gt.u32(more, row, start);
        if more {
            bra(scan);
        }
    }
}
