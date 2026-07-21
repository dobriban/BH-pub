#!/usr/bin/env python3
"""Outward-rounded certificate for the 163N+N+3N three-block model.

The only non-rigorous computation is generation of candidate feasible witnesses
in a separate JSON file.  Every witness and every inequality used by this
checker is verified with exact rational inputs and outward-rounded Arb balls.

A key analytic fact proved in the manuscript is that q(c;a,s) is unimodal in c
whenever 0<s<1.  The checker rigorously brackets each component mode using the
sign of log(q/h), where q is a Gaussian tail ratio and h is its boundary
likelihood ratio.  This provides sharp componentwise maxima on rational
c-intervals without unstable interval subtraction of nearly equal hazards.
"""
from __future__ import annotations
import argparse,json,math,time
from decimal import Decimal
from pathlib import Path
from flint import arb,ctx,fmpq
ctx.dps=100
SQRT2=arb(2).sqrt(); SQRT2PI=(2*arb.pi()).sqrt()

def qrat(x:str)->fmpq:
    if '/' in x:
        a,b=x.split('/');return fmpq(int(a),int(b))
    d=Decimal(x);sign,digits,exp=d.as_tuple();n=0
    for v in digits:n=10*n+v
    if sign:n=-n
    return fmpq(n*10**max(exp,0),10**max(-exp,0))

def iv(lo:fmpq,hi:fmpq)->arb:return arb((lo+hi)/2,(hi-lo)/2)
def ntail(x:arb)->arb:return (x/SQRT2).erfc()/2
def pthreshold(c:arb)->arb:return 2*ntail(c)
def qratio(c:arb,a:arb,s:arb)->arb:
    return (ntail((c-a)/s)+ntail((c+a)/s))/pthreshold(c)
def boundary_h(c:arb,a:arb,s:arb)->arb:
    # [phi_s(c-a)+phi_s(c+a)]/[2 phi(c)]
    t1=(-(c-a)*(c-a)/(2*s*s)+c*c/2).exp()
    t2=(-(c+a)*(c+a)/(2*s*s)+c*c/2).exp()
    return (t1+t2)/(2*s)
def mode_sign(c:fmpq,a:fmpq,s:arb)->arb:
    # sign(q-h), evaluated as log(q/h) for numerical stability
    cc=arb(c);aa=arb(a)
    return qratio(cc,aa,s).log()-boundary_h(cc,aa,s).log()
def absrange(mu:fmpq,rho:fmpq,zl:fmpq,zh:fmpq):
    x=mu+rho*zl;y=mu+rho*zh
    mn=fmpq(0) if ((x<=0<=y) or (y<=0<=x)) else min(abs(x),abs(y))
    return mn,max(abs(x),abs(y))

W=(fmpq(163,167),fmpq(1,167),fmpq(3,167))
MU=(fmpq(0),fmpq(37,20),fmpq(59,12))
RHO=(fmpq(3,10),fmpq(2,11),fmpq(20,21))
S=tuple((arb(1)-arb(r)**2).sqrt() for r in RHO)
SET={
 '0.01':dict(zden=100,zlo=-600,zhi=600,cstart=fmpq(2575,1000),claim='0.0111',omit=set()),
 '0.05':dict(zden=100,zlo=-600,zhi=600,cstart=fmpq(1959,1000),claim='0.0548',omit=set()),
 '0.10':dict(zden=100,zlo=-600,zhi=600,cstart=fmpq(1644,1000),claim='0.1053',omit=set()),
}

class Mode:
    def __init__(self,a:fmpq,s:arb,cmax:fmpq=fmpq(20),tol:fmpq=fmpq(1,10**6)):
        self.a=a;self.s=s;self.kind='';self.lo=fmpq(0);self.hi=fmpq(0);self.hmax=None
        # Resolve the initial sign.  At an exact stationary boundary, a tiny
        # rational displacement determines the relevant one-sided behavior.
        z=fmpq(0);sg=mode_sign(z,a,s)
        if not (sg>0 or sg<0):
            z=fmpq(1,10**8);sg=mode_sign(z,a,s)
        if sg<0:
            self.kind='decreasing';return
        # q is initially increasing.  By unimodality, a positive sign at cmax
        # proves increase throughout [0,cmax].
        sgmax=mode_sign(cmax,a,s)
        if sgmax>0:
            self.kind='increasing';return
        if not sgmax<0:
            raise RuntimeError(f'unresolved mode sign at cmax for a={a}')
        # Find a rational positive/negative sign bracket by dyadic expansion.
        lo=fmpq(0);hi=fmpq(1)
        slo=sg
        while hi<cmax:
            shi=mode_sign(hi,a,s)
            if shi<0:break
            if not shi>0:
                # Nudge an exactly or numerically stationary trial point.
                hi+=fmpq(1,10**8);shi=mode_sign(hi,a,s)
                if shi<0:break
                if not shi>0:raise RuntimeError(f'unresolved mode scan for a={a}')
            lo=hi;slo=shi;hi*=2
        if hi>cmax:hi=cmax
        shi=mode_sign(hi,a,s)
        if not (slo>0 and shi<0):
            raise RuntimeError(f'failed mode bracket for a={a}, signs {slo}, {shi}')
        while hi-lo>tol:
            mid=(lo+hi)/2;sm=mode_sign(mid,a,s)
            if sm>0:lo=mid
            elif sm<0:hi=mid
            else:
                # Exact rational midpoint too close to the root; retain a
                # certified bracket by moving to neighboring rational points.
                left=mid-tol/8;right=mid+tol/8
                sl=mode_sign(left,a,s);sr=mode_sign(right,a,s)
                if sl>0 and sr<0:lo,hi=left,right;break
                raise RuntimeError(f'unresolved mode bisection for a={a}')
        self.kind='interior';self.lo=lo;self.hi=hi
        # At the unique q-mode c*, q(c*)=h(c*), so this interval evaluation
        # rigorously bounds the global maximum of q.
        self.hmax=boundary_h(iv(lo,hi),arb(a),s).upper()
    def upper(self,lo:fmpq,hi:fmpq)->arb:
        aa=arb(self.a)
        if self.kind=='decreasing':return qratio(arb(lo),aa,self.s).upper()
        if self.kind=='increasing':return qratio(arb(hi),aa,self.s).upper()
        if hi<=self.lo:return qratio(arb(hi),aa,self.s).upper()
        if lo>=self.hi:return qratio(arb(lo),aa,self.s).upper()
        return self.hmax

class Certificate:
 def __init__(self,alpha:str,brackets:dict):
    self.alpha_str=alpha;self.A=arb(qrat(alpha));self.cfg=SET[alpha];self.br=brackets
    self.mode_cache={};self.eval=0;self.nodes=0;self.maxdepth=0;self.mode_count=0
    assert pthreshold(arb(self.cfg['cstart']))>self.A
 def getmode(self,g:int,a:fmpq)->Mode:
    key=(g,str(a));m=self.mode_cache.get(key)
    if m is None:
      m=Mode(a,S[g]);self.mode_cache[key]=m;self.mode_count+=1
    return m
 def infeasible(self,lo:fmpq,hi:fmpq,ah:tuple[fmpq,...])->bool:
    self.eval+=1;sm=arb(0)
    for g,(w,a) in enumerate(zip(W,ah)):
      sm+=arb(w)*self.getmode(g,a).upper(lo,hi)
    return self.A*sm<1
 def first_unverified(self,lo:fmpq,hi:fmpq,ah:tuple[fmpq,...],depth=0)->fmpq:
    self.nodes+=1;self.maxdepth=max(self.maxdepth,depth)
    if self.infeasible(lo,hi,ah):return hi
    if hi-lo<=fmpq(1,5000):return lo
    mid=(lo+hi)/2
    left=self.first_unverified(lo,mid,ah,depth+1)
    if left<mid:return left
    return self.first_unverified(mid,hi,ah,depth+1)
 def feasible(self,c:fmpq,al:tuple[fmpq,...])->bool:
    sm=arb(0)
    for w,a,s in zip(W,al,S):sm+=arb(w)*qratio(arb(c),arb(a),s)
    return self.A*sm>1
 def run(self):
    t0=time.time();cfg=self.cfg;den=self.br['bracket_denominator'];rows={r['z_index']:r for r in self.br['rows']}
    total=arb(0);groups={};records=[]
    for ordinal,k in enumerate(range(cfg['zlo'],cfg['zhi'])):
      if k in cfg['omit']:continue
      zl=fmpq(k,cfg['zden']);zh=fmpq(k+1,cfg['zden'])
      ranges=tuple(absrange(m,r,zl,zh) for m,r in zip(MU,RHO));al=tuple(x[0] for x in ranges);ah=tuple(x[1] for x in ranges)
      b=fmpq(rows[k]['b_num'],den)
      if not self.feasible(b,al):raise RuntimeError(f'not feasible bin {k}, b={b}')
      a=self.first_unverified(cfg['cstart'],b,ah)
      if not pthreshold(arb(a))<self.A:raise RuntimeError(f'prefix did not reach c_alpha in bin {k}, a={a}')
      # q_0 is unimodal, hence its minimum on [a,b] is at an endpoint.
      ql=qratio(arb(a),arb(al[0]),S[0]).lower();qr=qratio(arb(b),arb(al[0]),S[0]).lower();q0=ql if ql<qr else qr
      d=(self.A*arb(W[0])*q0).lower();mass=(ntail(arb(zl))-ntail(arb(zh))).lower();cont=(d*mass).lower();total=(total+cont).lower()
      gl=k//cfg['zden'];lab=f'[{gl},{gl+1}]';groups[lab]=(groups.get(lab,arb(0))+cont).lower()
      records.append({'z_lo':str(zl),'z_hi':str(zh),'a':str(a),'b':str(b),'fdp_lower_ball':str(d),'contribution_lower_ball':str(cont)})
      if (ordinal+1)%50==0:print('progress',ordinal+1,'total',total,'elapsed',time.time()-t0,flush=True)
    claim=arb(qrat(cfg['claim']));assert total>claim,(total,claim);assert total>self.A
    return {'alpha':self.alpha_str,'model':{'block_sizes_per_N':[163,1,3],'weights':['163/167','1/167','3/167'],'means':['0','37/20','59/12'],'loadings':['3/10','2/11','20/21'],'residual_sds':['sqrt(91)/10','3*sqrt(13)/11','sqrt(41)/21']},'advertised_certified_strict_lower_bound':cfg['claim'],'computed_total_lower_ball':str(total),'certified_factor_range':[str(fmpq(cfg['zlo'],cfg['zden'])),str(fmpq(cfg['zhi'],cfg['zden']))],'z_mesh':str(fmpq(1,cfg['zden'])),'terminal_prefix_width':'1/5000','omitted_z_bins':[f'[{fmpq(k,cfg["zden"])},{fmpq(k+1,cfg["zden"])}]' for k in sorted(cfg['omit'])],'grouped_contribution_lower_balls':{k:str(v) for k,v in sorted(groups.items())},'component_modes_certified':self.mode_count,'interval_sign_evaluations':self.eval,'prefix_nodes':self.nodes,'maximum_prefix_depth':self.maxdepth,'elapsed_seconds':time.time()-t0,'bins':records}

def main():
 p=argparse.ArgumentParser();p.add_argument('--alpha',choices=['0.01','0.05','0.10'],required=True);p.add_argument('--brackets',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 base=Path(__file__).resolve().parent
 brackets=a.brackets or base/f'brackets_alpha_{a.alpha}.json'
 output=a.output or base.parents[1]/'reproduced'/'central_three_block'/f'certificate_alpha_{a.alpha}.json'
 output.parent.mkdir(parents=True,exist_ok=True)
 br=json.loads(brackets.read_text());r=Certificate(a.alpha,br).run();output.write_text(json.dumps(r,indent=2)+'\n');print('CERTIFIED',r['alpha'],r['computed_total_lower_ball'],'>',r['advertised_certified_strict_lower_bound'],'elapsed',r['elapsed_seconds'])
if __name__=='__main__':main()
