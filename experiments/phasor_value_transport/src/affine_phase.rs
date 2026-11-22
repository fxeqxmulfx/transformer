//! Moving-frame scan and pullback for Transformer.Phasor.AffinePhaseRead.
//! Each lane owns one channel in one sequence. The second projection writes
//! into the real coordinate after the current rotation.
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

    // Projected rows hold the 32 angle increments and 32 real writes. Save
    // the inclusive angle in place. The write stays available for backward.
    pub unsafe extern "ptx-kernel" fn forward(read: *mut f32, projected: *mut f32, length: u32) {
        let (row, end, lane, index, second): (u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mul.lo.u32(row, row, length);
        add.u32(end, row, length);
        mov.u32(lane, tid.x);
        let (angle, frame_re, frame_im, step, write, sin, cos): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let (real, imag, temp): (f32, f32, f32);
        mov.f32(angle, 0.0);
        mov.f32(frame_re, 1.0);
        mov.f32(frame_im, 0.0);
        label(scan);
        mad.lo.u32(index, row, 64, lane);
        add.u32(second, index, 32);
        load(step, projected, index);
        load(write, projected, second);
        add.rn.f32(angle, angle, step);
        store(projected, index, angle);
        sin.approx.f32(sin, angle);
        cos.approx.f32(cos, angle);
        fma.rn.f32(frame_re, write, cos, frame_re);
        neg.f32(temp, sin);
        fma.rn.f32(frame_im, write, temp, frame_im);
        mul.rn.f32(real, temp, frame_im);
        fma.rn.f32(real, cos, frame_re, real);
        mul.rn.f32(imag, cos, frame_im);
        fma.rn.f32(imag, sin, frame_re, imag);
        sub.rn.f32(real, real, 1.0);
        store(read, index, imag);
        store(read, second, real);
        add.u32(row, row, 1);
        let more: pred;
        setp.lt.u32(more, row, end);
        if more {
            bra(scan);
        }
    }

    pub unsafe extern "ptx-kernel" fn backward(
        gradient: *mut f32,
        projected: *const f32,
        read: *const f32,
        dread: *const f32,
        length: u32,
    ) {
        let (row, start, lane, index, second): (u32, u32, u32, u32, u32);
        mov.u32(start, ctaid.x);
        mul.lo.u32(start, start, length);
        add.u32(row, start, length);
        mov.u32(lane, tid.x);
        let (frame_re, frame_im, angle_grad): (f32, f32, f32);
        let (angle, sin, cos, write, real, imag, gr, gi): (f32, f32, f32, f32, f32, f32, f32, f32);
        let (local, temp, write_grad): (f32, f32, f32);
        mov.f32(frame_re, 0.0);
        mov.f32(frame_im, 0.0);
        mov.f32(angle_grad, 0.0);
        label(scan);
        sub.u32(row, row, 1);
        mad.lo.u32(index, row, 64, lane);
        add.u32(second, index, 32);
        load(angle, projected, index);
        load(write, projected, second);
        load(imag, read, index);
        load(real, read, second);
        add.rn.f32(real, real, 1.0);
        load(gi, dread, index);
        load(gr, dread, second);
        sin.approx.f32(sin, angle);
        cos.approx.f32(cos, angle);
        // Accumulate cotangents for the real and imaginary moving-frame sums.
        mul.rn.f32(temp, gi, sin);
        fma.rn.f32(temp, gr, cos, temp);
        add.rn.f32(frame_re, frame_re, temp);
        mul.rn.f32(temp, gr, sin);
        neg.f32(temp, temp);
        fma.rn.f32(temp, gi, cos, temp);
        add.rn.f32(frame_im, frame_im, temp);
        mul.rn.f32(temp, frame_im, sin);
        neg.f32(temp, temp);
        fma.rn.f32(write_grad, frame_re, cos, temp);
        store(gradient, second, write_grad);
        // Differentiate both the output rotation and the write's inverse
        // rotation. Each earlier increment contributes to this prefix angle.
        mul.rn.f32(temp, gr, imag);
        neg.f32(temp, temp);
        fma.rn.f32(local, gi, real, temp);
        mul.rn.f32(temp, frame_re, sin);
        fma.rn.f32(temp, frame_im, cos, temp);
        neg.f32(temp, temp);
        fma.rn.f32(local, write, temp, local);
        add.rn.f32(angle_grad, angle_grad, local);
        store(gradient, index, angle_grad);
        let more: pred;
        setp.gt.u32(more, row, start);
        if more {
            bra(scan);
        }
    }
}
