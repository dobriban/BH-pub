#!/usr/bin/env python3
"""Plot limiting, certified, and finite-sample FDR for the three-block model."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_excess_panel(ax,alpha,certified_excess,fin,styles,axis_fontsize,
                      legend_fontsize):
    ax.axhline(0,color='black',linestyle='--',linewidth=1.15)
    ax.plot(alpha,certified_excess,color='#b45309',linewidth=2.0,
            label='Asymptotic lower bound on the violation')
    for N,g in fin.groupby('N'):
        N=int(N);g=g.sort_values('alpha');col,ls=styles[N]
        ax.plot(g['alpha'],g['fdr']-g['alpha'],color=col,linestyle=ls,
                linewidth=1.1 if N<1000 else 1.65)
        if N==1000:
            ax.fill_between(
                g['alpha'].to_numpy(),
                (g['ci95_lower']-g['alpha']).to_numpy(),
                (g['ci95_upper']-g['alpha']).to_numpy(),
                color=col,alpha=.10,linewidth=0,
                label='Simulation, 95% Confidence Interval'
            )
    for x in (0.01,0.05,0.10):
        ax.axvline(x,color='black',alpha=.13,linewidth=.7)
    ax.set_xlabel(r'Nominal FDR $\alpha$',fontsize=axis_fontsize)
    ax.set_ylabel(r'FDR $-\alpha$',fontsize=axis_fontsize)
    ax.set_xlim(.001,.100);ax.set_ylim(-.0010,.0068)
    ax.set_xticks(np.arange(.01,.101,.01))
    ax.grid(True,alpha=.22,linewidth=.55)
    ax.legend(loc='lower right',fontsize=legend_fontsize,frameon=True)
    ax.tick_params(axis='both',labelsize=axis_fontsize)
    plt.setp(ax.get_xticklabels(),rotation=30,ha='right',rotation_mode='anchor')

def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        '--curve', '--limiting', dest='curve', type=Path, required=True,
        help='CSV containing the deterministic_quadrature column; --limiting is a legacy alias'
    )
    p.add_argument('--certified',type=Path,required=True)
    p.add_argument('--finite',type=Path,required=True)
    p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--violation-output',type=Path)
    a=p.parse_args()
    lim=pd.read_csv(a.curve)
    cert=pd.read_csv(a.certified)
    fin=pd.read_csv(a.finite)
    fin=fin.loc[fin['N'].eq(1000)].copy()
    alpha=lim['alpha'].to_numpy()
    det=lim['deterministic_quadrature'].to_numpy()
    if not np.allclose(alpha,cert['alpha'].to_numpy(),rtol=0,atol=1e-14):
        raise ValueError('curve and certified alpha grids do not agree')
    certified_excess=cert['certified_excess_lower_bound'].to_numpy()
    font_scale=1.10
    axis_fontsize=15*font_scale
    top_legend_fontsize=11.55*font_scale
    excess_legend_fontsize=12.0*font_scale

    fig,axs=plt.subplots(1,2,figsize=(12.8,4.3),sharex=True)
    ax=axs[0]
    ax.plot(alpha,alpha,color='black',linestyle='--',linewidth=1.25,label='Nominal FDR')
    ax.plot(alpha,det,color='#b45309',linewidth=2.15,label='Limiting FDR')
    styles={
      50:('#64748b','-.'),
      100:('#059669','--'),
      500:('#9333ea',(0,(4,1.4))),
      1000:('#dc2626','-'),
    }
    for N,g in fin.groupby('N'):
        N=int(N);g=g.sort_values('alpha');col,ls=styles[N]
        ax.plot(g['alpha'],g['fdr'],color=col,linestyle=ls,
                linewidth=1.25 if N<1000 else 1.75,
                label=fr'Simulation, $m={167*N:,}$')
    ax.set_ylabel('Realized FDR',fontsize=axis_fontsize)
    ax.set_xlim(.001,.100);ax.set_ylim(0,.109)
    ax.grid(True,alpha=.22,linewidth=.55)
    ax.legend(loc='upper left',fontsize=top_legend_fontsize,ncol=1,frameon=True)
    ax.set_xlabel(r'Nominal FDR $\alpha$',fontsize=axis_fontsize)

    plot_excess_panel(axs[1],alpha,certified_excess,fin,styles,axis_fontsize,
                      excess_legend_fontsize)
    for ax in axs:
        ax.tick_params(axis='both',labelsize=axis_fontsize)
        plt.setp(ax.get_xticklabels(),rotation=30,ha='right',rotation_mode='anchor')
    fig.tight_layout()
    a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output_prefix.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(a.output_prefix.with_suffix('.png'),dpi=260,bbox_inches='tight')
    print('wrote',a.output_prefix.with_suffix('.pdf'),'and',a.output_prefix.with_suffix('.png'))
    if a.violation_output:
        standalone,ax=plt.subplots(figsize=(6.6,4.3))
        plot_excess_panel(ax,alpha,certified_excess,fin,styles,axis_fontsize,
                          excess_legend_fontsize)
        standalone.tight_layout()
        a.violation_output.parent.mkdir(parents=True,exist_ok=True)
        standalone.savefig(a.violation_output,bbox_inches='tight')
        print('wrote',a.violation_output)
if __name__=='__main__':
    main()
