//! FP32 kernels for the complex decoder and its independent real coordinates.
#[kernite_macros::kernels]
pub(super) mod device {
    fn index(i: u32) {
        let (block, lane, threads): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        mov.u32(threads, ntid.x);
        mad.lo.u32(i, block, threads, lane);
    }
    fn load(x: f32, base: u64, i: u32) {
        let ptr: u64;
        mad.wide.u32(ptr, i, 4, base);
        ld.global.f32(x, [ptr]);
    }
    fn put(base: u64, i: u32, x: f32) {
        let ptr: u64;
        mad.wide.u32(ptr, i, 4, base);
        st.global.f32([ptr], x);
    }
    fn accumulate(base: u64, i: u32, x: f32) {
        let old: f32;
        load(old, base, i);
        add.rn.f32(old, old, x);
        put(base, i, old);
    }
    fn atomic_add(base: u64, i: u32, x: f32) {
        let ptr: u64;
        mad.wide.u32(ptr, i, 4, base);
        let old: f32;
        atom.global.add.f32(old, [ptr], x);
    }
    fn complex_load(re: f32, im: f32, base: u64, i: u32) {
        let ptr: u64;
        mad.wide.u32(ptr, i, 8, base);
        ld.global.f32(re, [ptr]);
        ld.global.f32(im, [ptr + 4]);
    }
    fn complex_put(base: u64, i: u32, re: f32, im: f32, add: pred) {
        let ptr: u64;
        mad.wide.u32(ptr, i, 8, base);
        if add {
            let (a, b): (f32, f32);
            ld.global.f32(a, [ptr]);
            ld.global.f32(b, [ptr + 4]);
            add.rn.f32(re, re, a);
            add.rn.f32(im, im, b);
        }
        st.global.f32([ptr], re);
        st.global.f32([ptr + 4], im);
    }
    fn reduce_sum(x: f32) {
        let peer: f32;
        for step in 0..5 {
            shfl.sync.bfly.b32(peer, x, 1 << step, 31, 0xffffffff);
            add.rn.f32(x, x, peer);
        }
    }
    fn reduce_max(x: f32) {
        let peer: f32;
        for step in 0..5 {
            shfl.sync.bfly.b32(peer, x, 1 << step, 31, 0xffffffff);
            max.f32(x, x, peer);
        }
    }
    fn exp(x: f32) {
        mul.rn.f32(x, x, 1.4426950408889634);
        ex2.approx.f32(x, x);
    }
    fn mode<const BIT: u32>(pred_value: pred, flags: u32) {
        let x: u32;
        and.b32(x, flags, BIT);
        setp.ne.u32(pred_value, x, 0);
    }
    // Flags: transpose A/B/Y, conjugate A/B/Y, accumulate into Y.
    pub unsafe extern "ptx-kernel" fn matmul(
        y: *mut f32, a: *const f32, b: *const f32,
        m: u32, n: u32, k: u32, tiles_m: u32, tiles_n: u32, flags: u32,
    ) {
        let a_tile: shared<[f32; 512], 16>;
        let b_tile: shared<[f32; 512], 16>;
        let (ta, tb, ty, ca, cb, cy, add): (pred, pred, pred, pred, pred, pred, pred);
        mode::<1>(ta, flags); mode::<2>(tb, flags); mode::<4>(ty, flags);
        mode::<8>(ca, flags); mode::<16>(cb, flags); mode::<32>(cy, flags);
        mode::<64>(add, flags);
        let (lane, x, t, row, col, block, batch, tile_m, tile_n): (u32,u32,u32,u32,u32,u32,u32,u32,u32);
        mov.u32(lane, tid.x);
        and.b32(x, lane, 15);
        shr.u32(t, lane, 4);
        mov.u32(block, ctaid.x);
        rem.u32(tile_n, block, tiles_n);
        div.u32(block, block, tiles_n);
        rem.u32(tile_m, block, tiles_m);
        div.u32(batch, block, tiles_m);
        mad.lo.u32(row, tile_m, 16, t);
        mad.lo.u32(col, tile_n, 16, x);
        let (ai, bi, ci, aa, ab, ba, bb, ap, bp): (u32,u32,u32,u32,u32,u32,u32,u32,u32);
        mul.lo.u32(ai, m, k);
        mul.lo.u32(ai, ai, batch);
        mul.lo.u32(bi, n, k);
        mul.lo.u32(bi, bi, batch);
        mul.lo.u32(ci, m, n);
        mul.lo.u32(ci, ci, batch);
        selp.u32(aa, 1, k, ta); selp.u32(ab, m, 1, ta);
        selp.u32(ba, 1, n, tb); selp.u32(bb, k, 1, tb);
        mad.lo.u32(ai, row, aa, ai);
        mad.lo.u32(bi, col, bb, bi);
        mov.u32(ap, a_tile);
        mad.lo.u32(ap, lane, 8, ap);
        mov.u32(bp, b_tile);
        mad.lo.u32(bp, lane, 8, bp);
        let (re, im): (f32, f32);
        mov.f32(re, 0.0); mov.f32(im, 0.0);
        let (start, ar, br, q): (u32,u32,u32,u32);
        mov.u32(start, 0);
        let (valid_row, valid_col, valid, within): (pred,pred,pred,pred);
        setp.lt.u32(valid_row, row, m);
        setp.lt.u32(valid_col, col, n);
        label(tile);
        add.u32(ar, start, x);
        add.u32(br, start, t);
        let (a0, a1, b0, b1): (f32,f32,f32,f32);
        mov.f32(a0,0.0); mov.f32(a1,0.0); mov.f32(b0,0.0); mov.f32(b1,0.0);
        setp.lt.u32(within, ar, k);
        and.pred(valid, within, valid_row);
        if valid {
            mad.lo.u32(q, ar, ab, ai);
            complex_load(a0,a1,a,q);
        }
        setp.lt.u32(within, br, k);
        and.pred(valid, within, valid_col);
        if valid {
            mad.lo.u32(q, br, ba, bi);
            complex_load(b0,b1,b,q);
        }
        if ca { neg.f32(a1,a1); }
        if cb { neg.f32(b1,b1); }
        st.shared.f32([ap], a0); st.shared.f32([ap + 4], a1);
        st.shared.f32([bp], b0); st.shared.f32([bp + 4], b1);
        bar.sync(0);
        let (ax, bx): (u32,u32);
        mov.u32(ax, a_tile);
        mad.lo.u32(ax, t, 128, ax);
        mov.u32(bx, b_tile);
        mad.lo.u32(bx, x, 8, bx);
        for j in 0..16 {
            ld.shared.f32(a0,[ax + j * 8]);
            ld.shared.f32(a1,[ax + j * 8 + 4]);
            ld.shared.f32(b0,[bx + j * 128]);
            ld.shared.f32(b1,[bx + j * 128 + 4]);
            fma.rn.f32(re,a0,b0,re);
            fma.rn.f32(im,a1,b0,im);
            fma.rn.f32(im,a0,b1,im);
            neg.f32(a1,a1);
            fma.rn.f32(re,a1,b1,re);
        }
        bar.sync(0);
        add.u32(start, start, 16);
        setp.lt.u32(within, start, k);
        if within { bra(tile); }
        and.pred(valid, valid_row, valid_col);
        if !valid { ret(); }
        selp.u32(aa, 1, n, ty); selp.u32(ab, m, 1, ty);
        mad.lo.u32(ci, row, aa, ci);
        mad.lo.u32(ci, col, ab, ci);
        if cy { neg.f32(im, im); }
        complex_put(y,ci,re,im,add);
    }

    // One warp per row, looping over its two real coordinates per complex number.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn rms<const BACK: bool>(
        y: *mut f32, x: *const f32, dy: *const f32, width: u32, gain: f32,
    ) {
        let (row,lane,j,base,at): (u32,u32,u32,u32,u32);
        mov.u32(row,ctaid.x); mov.u32(lane,tid.x);
        mul.lo.u32(base,row,width);
        mov.u32(j,lane);
        let (square,dot,v,g): (f32,f32,f32,f32);
        mov.f32(square,0.0); mov.f32(dot,0.0);
        let active: pred;
        label(reduce);
        setp.ge.u32(active,j,width);
        if active { bra(finish_reduce); }
        add.u32(at,base,j);
        load(v,x,at);
        fma.rn.f32(square,v,v,square);
        if BACK { load(g,dy,at); fma.rn.f32(dot,v,g,dot); }
        add.u32(j,j,32);
        bra(reduce);
        label(finish_reduce);
        reduce_sum(square);
        sqrt.rn.f32(square,square);
        let denom: f32;
        add.rn.f32(denom,square,1e-6);
        div.rn.f32(gain,gain,denom);
        if BACK {
            reduce_sum(dot);
            setp.gt.f32(active,square,0.0);
            if active {
                mul.rn.f32(denom,denom,square);
                div.rn.f32(dot,dot,denom);
            } else { mov.f32(dot,0.0); }
        }
        mov.u32(j,lane);
        label(write);
        setp.ge.u32(active,j,width);
        if active { ret(); }
        add.u32(at,base,j);
        load(v,x,at);
        if BACK {
            load(g,dy,at);
            neg.f32(v,v);
            fma.rn.f32(v,v,dot,g);
        }
        mul.rn.f32(v,v,gain);
        if BACK { accumulate(y,at,v); } else { put(y,at,v); }
        add.u32(j,j,32);
        bra(write);
    }

    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn product<const BACK: bool>(
        y: *mut f32, a: *mut f32, b: *mut f32, dy: *const f32,
        da: *mut f32, db: *mut f32, size: u32,
    ) {
        let i: u32; index(i);
        let flag: pred;
        setp.ge.u32(flag,i,size);
        if flag { ret(); }
        let (ar,ai,br,bi,yr,yi,gr,gi,t): (f32,f32,f32,f32,f32,f32,f32,f32,f32);
        complex_load(ar,ai,a,i); complex_load(br,bi,b,i);
        if BACK {
            complex_load(gr,gi,dy,i);
            setp.eq.u32(flag,0,0);
            mul.rn.f32(yr,gr,br); fma.rn.f32(yr,gi,bi,yr);
            mul.rn.f32(yi,gi,br); neg.f32(t,bi); fma.rn.f32(yi,gr,t,yi);
            complex_put(da,i,yr,yi,flag);
            mul.rn.f32(yr,gr,ar); fma.rn.f32(yr,gi,ai,yr);
            mul.rn.f32(yi,gi,ar); neg.f32(t,ai); fma.rn.f32(yi,gr,t,yi);
            complex_put(db,i,yr,yi,flag);
        } else {
            mul.rn.f32(yr,ai,bi); neg.f32(yr,yr); fma.rn.f32(yr,ar,br,yr);
            mul.rn.f32(yi,ai,br); fma.rn.f32(yi,ar,bi,yi);
            complex_put(y,i,yr,yi,flag);
        }
    }

    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn sum<const BACK: bool>(
        y: *mut f32, a: *const f32, b: *const f32, dy: *const f32,
        da: *mut f32, db: *mut f32, size: u32, scale: f32,
    ) {
        let i: u32; index(i);
        let past: pred; setp.ge.u32(past,i,size);
        if past { ret(); }
        let (u,v): (f32,f32);
        if BACK {
            load(u,dy,i);
            accumulate(da,i,u);
            mul.rn.f32(u,u,scale);
            accumulate(db,i,u);
        } else {
            load(u,a,i); load(v,b,i);
            fma.rn.f32(u,v,scale,u);
            put(y,i,u);
        }
    }

    // The forward load is token-major, and the same index becomes the
    // backward destination. FLIP reverses the packing operation.
    #[instances((false,false),(true,false),(false,true),(true,true))]
    pub unsafe extern "ptx-kernel" fn heads<const BACK: bool, const FLIP: bool>(
        y: *mut f32, x: *const f32, size: u32, length: u32, heads: u32, width: u32,
    ) {
        let (i,j,col,head,time,batch): (u32,u32,u32,u32,u32,u32);
        index(i);
        let past: pred; setp.ge.u32(past,i,size);
        if past { ret(); }
        rem.u32(col,i,width); div.u32(j,i,width);
        rem.u32(head,j,heads); div.u32(j,j,heads);
        rem.u32(time,j,length); div.u32(batch,j,length);
        mad.lo.u32(j,batch,heads,head);
        mad.lo.u32(j,j,length,time);
        mad.lo.u32(j,j,width,col);
        let v: f32;
        if BACK != FLIP {
            load(v,x,j);
            if BACK { accumulate(y,i,v); } else { put(y,i,v); }
        } else {
            load(v,x,i);
            if BACK { accumulate(y,j,v); } else { put(y,j,v); }
        }
    }

    // Store (cos(theta), sin(theta), theta) for each position and frequency.
    pub unsafe extern "ptx-kernel" fn rope_table(
        table: *mut f32, alpha: *const f32, width: u32, count: u32, base: f32,
    ) {
        let (i,pos,col,k): (u32,u32,u32,u32);
        index(i);
        let past: pred; setp.ge.u32(past,i,count);
        if past { ret(); }
        div.u32(pos,i,width); rem.u32(col,i,width);
        let (x,a,p,c,s): (f32,f32,f32,f32,f32);
        cvt.rn.f32.u32(x,col); cvt.rn.f32.u32(p,pos);
        load(a,alpha,col);
        fma.rn.f32(a,x,base,a);
        exp(a);
        mul.rn.f32(a,a,p);
        cos.approx.f32(c,a); sin.approx.f32(s,a);
        mul.lo.u32(k,i,3);
        put(table,k,c); add.u32(k,k,1); put(table,k,s); add.u32(k,k,1); put(table,k,a);
    }

    #[instances(false,true)]
    pub unsafe extern "ptx-kernel" fn rotate<const BACK: bool>(
        y: *mut f32, x: *const f32, dy: *const f32, table: *const f32,
        dalpha: *mut f32, width: u32, length: u32, heads: u32, size: u32,
    ) {
        let (i,j,col,row): (u32,u32,u32,u32);
        index(i);
        let past: pred; setp.ge.u32(past,i,size);
        if past { ret(); }
        rem.u32(col,i,width);
        div.u32(row,i,width); div.u32(row,row,heads); rem.u32(row,row,length);
        mad.lo.u32(j,row,width,col); mul.lo.u32(j,j,3);
        let (re,im,cos,sin,angle,u,v,t): (f32,f32,f32,f32,f32,f32,f32,f32);
        load(cos,table,j); add.u32(j,j,1); load(sin,table,j);
        if BACK {
            add.u32(j,j,1); load(angle,table,j);
            complex_load(re,im,dy,i);
            mul.rn.f32(u,re,cos); fma.rn.f32(u,im,sin,u);
            mul.rn.f32(v,im,cos); neg.f32(t,sin); fma.rn.f32(v,re,t,v);
            setp.eq.u32(past,0,0);
            complex_put(y,i,u,v,past);
            complex_load(u,v,x,i);
            mul.rn.f32(t,re,v); neg.f32(t,t); fma.rn.f32(t,im,u,t);
            mul.rn.f32(t,t,angle);
            atomic_add(dalpha,col,t);
        } else {
            complex_load(re,im,x,i);
            mul.rn.f32(u,im,sin); neg.f32(u,u); fma.rn.f32(u,re,cos,u);
            mul.rn.f32(v,im,cos); fma.rn.f32(v,re,sin,v);
            complex_put(y,i,u,v,past);
        }
    }

    /// Out-of-place trial weights for a PAGE line search.
    pub unsafe extern "ptx-kernel" fn page_trial(
        out: *mut f32, start: *const f32, estimate: *const f32, count: u32, rate: f32,
    ) {
        let i: u32; index(i);
        let past: pred; setp.ge.u32(past,i,count);
        if past { ret(); }
        let (w,g): (f32,f32);
        load(w,start,i); load(g,estimate,i);
        neg.f32(g,g); fma.rn.f32(w,rate,g,w);
        put(out,i,w);
    }

    /// Both gradients are of the same component, at consecutive accepted weights.
    pub unsafe extern "ptx-kernel" fn page_correct(
        estimate: *mut f32, after: *const f32, before: *const f32, count: u32,
    ) {
        let i: u32; index(i);
        let past: pred; setp.ge.u32(past,i,count);
        if past { ret(); }
        let (g,a,b): (f32,f32,f32);
        load(g,estimate,i); load(a,after,i); load(b,before,i);
        sub.rn.f32(a,a,b); add.rn.f32(g,g,a); put(estimate,i,g);
    }

    /// SGD with Polyak momentum on the globally clipped gradient.
    pub unsafe extern "ptx-kernel" fn sgd_momentum(
        p: *mut f32,
        momentum: *mut f32,
        g: *const f32,
        norm: *const f32,
        count: u32,
        lr: f32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (value, velocity, gradient, scale, displacement): (f32, f32, f32, f32, f32);
        ld.global.f32(scale, [norm]);
        max.f32(scale, scale, 1.0);
        load(gradient, g, i);
        div.rn.f32(gradient, gradient, scale);
        load(velocity, momentum, i);
        fma.rn.f32(velocity, 0.9, velocity, gradient);
        load(value, p, i);
        mul.rn.f32(displacement, lr, velocity);
        sub.rn.f32(value, value, displacement);
        put(p, i, value);
        put(momentum, i, velocity);
    }

    /// Adds `lam * p`, the gradient of the penalty `(lam / 2) * |p|^2`.
    pub unsafe extern "ptx-kernel" fn add_penalty(
        g: *mut f32,
        p: *const f32,
        count: u32,
        lam: f32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (value, gradient): (f32, f32);
        load(value, p, i);
        load(gradient, g, i);
        fma.rn.f32(gradient, lam, value, gradient);
        put(g, i, gradient);
    }

    /// Zeroes `count` coordinates, as `clear` in
    /// experiments/phasor_value_transport/src/slot_copy.rs.
    pub unsafe extern "ptx-kernel" fn clear(out: *mut f32, count: u32) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let zero: f32;
        mov.f32(zero, 0.0);
        put(out, i, zero);
    }

    #[instances(false,true)]
    pub unsafe extern "ptx-kernel" fn real_readout<const BACK: bool>(
        y: *mut f32, x: *const f32, size: u32,
    ) {
        let (i,j): (u32,u32);
        index(i);
        let past: pred; setp.ge.u32(past,i,size);
        if past { ret(); }
        let v: f32;
        mul.lo.u32(j,i,2);
        if BACK {
            load(v,x,i); mul.rn.f32(v,v,0.5); put(y,j,v);
            add.u32(j,j,1); mov.f32(v,0.0); put(y,j,v);
        } else {
            load(v,x,j); mul.rn.f32(v,v,0.5); put(y,i,v);
        }
    }
}
