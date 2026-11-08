//! LF3 mean pairwise margin loss. Source launch/reduction layout: original
//! rotate90_complex ops.rs at source SHA 14cff3b5; changed objective and stable
//! softplus/sigmoid. No encoder forward or derivative is hidden in this kernel.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        ld.global.f32(x, [p]);
    }
    fn store(base: u64, i: u32, x: f32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        st.global.f32([p], x);
    }
    fn load_u(x: u32, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        ld.global.u32(x, [p]);
    }
    fn sum32(x: f32) {
        let other: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(other, x, 1 << j, 31, 0xffffffff);
            add.rn.f32(x, x, other);
        }
    }
    pub unsafe extern "ptx-kernel" fn pairwise(
        gradients: *mut f32,
        losses: *mut f32,
        logits: *const f32,
        labels: *const u32,
        batch: u32,
    ) {
        let (block, thread, threads, i, row, lane, target, at, cls): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        mov.u32(block, ctaid.x);
        mov.u32(thread, tid.x);
        mov.u32(threads, ntid.x);
        mad.lo.u32(i, block, threads, thread);
        shr.u32(row, i, 5);
        and.b32(lane, i, 31);
        let past: pred;
        setp.ge.u32(past, row, batch);
        if past {
            ret();
        }
        load_u(target, labels, row);
        mad.lo.u32(at, row, 97, target);
        let (chosen, value, total, qsum, q, z, e, den, positive, normalizer): (
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
        load(chosen, logits, at);
        mov.f32(total, 0.0);
        mov.f32(qsum, 0.0);
        cvt.rn.f32.u32(normalizer, batch);
        mul.rn.f32(normalizer, normalizer, 96.0);
        for j in 0..4 {
            add.u32(cls, lane, 32 * j);
            mad.lo.u32(at, row, 97, cls);
            mov.f32(q, 0.0);
            let (valid, negative, work): (pred, pred, pred);
            setp.lt.u32(valid, cls, 97);
            setp.ne.u32(negative, cls, target);
            and.pred(work, valid, negative);
            if work {
                load(value, logits, at);
                sub.rn.f32(z, value, chosen);
                add.rn.f32(z, z, 1.0);
                abs.f32(e, z);
                mul.rn.f32(e, e, -1.4426950408889634);
                ex2.approx.f32(e, e);
                add.rn.f32(den, e, 1.0);
                let nonnegative: pred;
                setp.ge.f32(nonnegative, z, 0.0);
                selp.f32(q, 1.0, e, nonnegative);
                div.rn.f32(q, q, den);
                add.rn.f32(qsum, qsum, q);
                lg2.approx.f32(den, den);
                mul.rn.f32(den, den, 0.6931471805599453);
                max.f32(positive, z, 0.0);
                add.rn.f32(den, den, positive);
                add.rn.f32(total, total, den);
                div.rn.f32(q, q, normalizer);
            }
            if valid {
                store(gradients, at, q);
            }
        }
        sum32(total);
        sum32(qsum);
        setp.eq.u32(past, lane, 0);
        if past {
            div.rn.f32(total, total, 96.0);
            store(losses, row, total);
            neg.f32(qsum, qsum);
            div.rn.f32(qsum, qsum, normalizer);
            mad.lo.u32(at, row, 97, target);
            store(gradients, at, qsum);
        }
    }
}
