//! Separate causal softmaxes for the real and imaginary score coordinates.
#[kernite_macros::kernels]
pub(super) mod device {
    fn sum(v: f32) {
        let peer: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(peer,v,1 << j,31,0xffffffff);
            add.rn.f32(v,v,peer);
        }
    }
    fn max(v: f32) {
        let peer: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(peer,v,1 << j,31,0xffffffff);
            max.f32(v,v,peer);
        }
    }
    fn load(r: f32, i: f32, p: u64, index: u32) {
        let at: u64;
        mad.wide.u32(at,index,8,p);
        ld.global.f32(r,[at]); ld.global.f32(i,[at+4]);
    }
    fn put(p: u64, index: u32, r: f32, i: f32) {
        let at: u64;
        mad.wide.u32(at,index,8,p);
        st.global.f32([at],r); st.global.f32([at+4],i);
    }
    fn score(r: f32, i: f32, a: f32, b: f32, p: u64, index: u32) {
        let (u,v): (f32,f32);
        load(u,v,p,index);
        mul.rn.f32(r,b,v); neg.f32(r,r); fma.rn.f32(r,a,u,r);
        mul.rn.f32(i,a,v); fma.rn.f32(i,b,u,i);
    }
    fn exp(x: f32, m: f32) {
        sub.rn.f32(x,x,m);
        mul.rn.f32(x,x,1.4426950408889634);
        ex2.approx.f32(x,x);
    }
    #[instances(false,true)]
    pub unsafe extern "ptx-kernel" fn weights<const BACK: bool>(
        p: *mut f32, s: *const f32, raw: *const f32, dprob: *const f32,
        draw: *mut f32, ds: *mut f32, length: u32, gain: f32,
    ) {
        let (row,lane,limit,base,j,index): (u32,u32,u32,u32,u32,u32);
        mov.u32(row,ctaid.x); mov.u32(lane,tid.x);
        rem.u32(limit,row,length); add.u32(limit,limit,1);
        mul.lo.u32(base,row,length);
        let (a,b,g,r,i,u,v,ra,ia,dr,di): (f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32);
        cvt.rn.f32.u32(g,limit);
        lg2.approx.f32(g,g); mul.rn.f32(g,g,gain);
        ld.global.f32(a,[s]); ld.global.f32(b,[s+4]);
        mul.rn.f32(a,a,g); mul.rn.f32(b,b,g);
        mov.u32(j,lane);
        let done: pred;
        if BACK {
            mov.f32(ra,0.0); mov.f32(ia,0.0);
        } else {
            mov.f32(ra,f32::NEG_INFINITY); mov.f32(ia,f32::NEG_INFINITY);
        }
        label(reduce);
        setp.ge.u32(done,j,limit);
        if done { bra(reduced); }
        add.u32(index,base,j);
        if BACK {
            load(r,i,p,index); load(u,v,dprob,index);
            fma.rn.f32(ra,r,u,ra);
            fma.rn.f32(ia,i,v,ia);
        } else {
            score(r,i,a,b,raw,index);
            max.f32(ra,ra,r); max.f32(ia,ia,i);
        }
        add.u32(j,j,32); bra(reduce);
        label(reduced);
        if BACK {
            sum(ra); sum(ia);
            mov.f32(dr,0.0); mov.f32(di,0.0);
        } else {
            max(ra); max(ia);
            mov.f32(dr,0.0); mov.f32(di,0.0);
            mov.u32(j,lane);
            label(total);
            setp.ge.u32(done,j,limit);
            if done { bra(total_ready); }
            add.u32(index,base,j);
            score(r,i,a,b,raw,index);
            exp(r,ra); exp(i,ia);
            add.rn.f32(dr,dr,r); add.rn.f32(di,di,i);
            put(p,index,r,i);
            add.u32(j,j,32); bra(total);
            label(total_ready);
            sum(dr); sum(di);
        }
        mov.u32(j,lane);
        label(write);
        setp.ge.u32(done,j,length);
        if done { bra(finish); }
        add.u32(index,base,j);
        let masked: pred;
        setp.ge.u32(masked,j,limit);
        mov.f32(r,0.0); mov.f32(i,0.0);
        if !masked {
            load(r,i,p,index);
            if BACK {
                load(u,v,dprob,index);
                sub.rn.f32(u,u,ra); sub.rn.f32(v,v,ia);
                mul.rn.f32(u,u,r); mul.rn.f32(v,v,i);
                load(r,i,raw,index);
                fma.rn.f32(dr,u,r,dr); fma.rn.f32(dr,v,i,dr);
                fma.rn.f32(di,v,r,di); neg.f32(i,i); fma.rn.f32(di,u,i,di);
                mul.rn.f32(r,u,a); fma.rn.f32(r,v,b,r);
                mul.rn.f32(i,v,a); neg.f32(u,u); fma.rn.f32(i,u,b,i);
            } else {
                div.rn.f32(r,r,dr); div.rn.f32(i,i,di);
            }
        }
        if BACK { put(draw,index,r,i); } else { put(p,index,r,i); }
        add.u32(j,j,32); bra(write);
        label(finish);
        if BACK {
            sum(dr); sum(di);
            setp.eq.u32(done,lane,0);
            if done {
                mul.rn.f32(dr,dr,g); mul.rn.f32(di,di,g);
                let old: f32;
                atom.global.add.f32(old,[ds],dr);
                atom.global.add.f32(old,[ds+4],di);
            }
        }
    }
}
