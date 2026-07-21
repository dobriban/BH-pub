#!/usr/bin/env python3
from __future__ import annotations
import math
import numpy as np
from scipy.special import log_ndtr, ndtri
W=np.array([163/167,1/167,3/167],float)
MU=np.array([0.,37/20,59/12],float)
RHO=np.array([3/10,2/11,20/21],float)
SD=np.sqrt(1-RHO*RHO)
ALPHAS=np.arange(1,101,dtype=float)/1000

def c_grid():
    c0=float(ndtri(1-.1/2))
    return np.unique(np.r_[c0,
      np.arange(c0,15,.005),
      np.arange(15,30.0001,.02)])
CGRID=c_grid()
LOGU_GRID=math.log(2)+log_ndtr(-CGRID)

def logQ(c,a,s):
    return np.logaddexp(log_ndtr((a-c)/s),log_ndtr((-a-c)/s))

def logG(c,z):
    # c and z broadcast to a common leading shape
    out=None
    for w,m,r,s in zip(W,MU,RHO,SD):
        v=math.log(w)+logQ(c,np.abs(m+r*z),s)
        out=v if out is None else np.logaddexp(out,v)
    return out

def batch_curve(z, alphas=ALPHAS, refine=24):
    z=np.asarray(z,dtype=float)
    B=z.size; A=len(alphas); C=len(CGRID)
    # Required nominal level at each candidate c:
    # alpha >= u(c) and alpha >= u(c)/G(c) = 1/S(c).
    lg=logG(CGRID[None,:],z[:,None])
    required=np.maximum(np.exp(LOGU_GRID)[None,:],np.exp(LOGU_GRID[None,:]-lg))
    bins=np.full(required.shape,101,dtype=np.int32)
    finite=np.isfinite(required)
    bins[finite]=np.ceil(required[finite]*1000-1e-12).astype(np.int32)
    mask=(bins>=1)&(bins<=100)
    earliest=np.full((B,101),C,dtype=np.int32)
    combined=np.arange(B,dtype=np.int64)[:,None]*101+np.clip(bins,0,100)
    cidx=np.broadcast_to(np.arange(C,dtype=np.int32),(B,C))
    np.minimum.at(earliest.ravel(),combined[mask],cidx[mask])
    idx=np.minimum.accumulate(earliest[:,1:],axis=1)
    has=idx<C
    safe=np.minimum(idx,C-1)
    hi=CGRID[safe]
    prev=np.maximum(safe-1,0)
    lo=CGRID[prev]
    ca=ndtri(1-alphas/2)[None,:]
    lo=np.maximum(lo,ca)
    hi=np.maximum(hi,ca)
    # If c_alpha itself is feasible, the first crossing is c_alpha.
    sca=np.log(alphas)[None,:]+logG(ca,z[:,None])-(math.log(2)+log_ndtr(-ca))
    at_start=sca>=0
    hi=np.where(at_start,ca,hi)
    lo=np.where(at_start,ca,lo)
    # Refine all located crossings simultaneously.
    for _ in range(refine):
        mid=(lo+hi)/2
        score=np.log(alphas)[None,:]+logG(mid,z[:,None])-(math.log(2)+log_ndtr(-mid))
        feasible=(score>=0)&(mid>=ca)
        hi=np.where(feasible,mid,hi)
        lo=np.where(feasible,lo,mid)
    cs=(lo+hi)/2
    lgc=logG(cs,z[:,None])
    l0=math.log(W[0])+logQ(cs,np.abs(RHO[0]*z[:,None]),SD[0])
    fdp=np.where(has,np.clip(np.exp(l0-lgc),0,1),0.0)
    return fdp,cs,has

if __name__=='__main__':
    import time
    z=np.linspace(-8,8,201)
    t=time.time();fdp,cs,h=batch_curve(z);print('shape',fdp.shape,'sec',time.time()-t,'vals',fdp[:,[9,49,99]].mean(0),'maxc',np.nanmax(np.where(h,cs,np.nan)))
