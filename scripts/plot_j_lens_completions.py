"""Plot the fixed completion follow-up without model access."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]/'output/2026-09-10-j-lens-completions'
FIG = OUT/'figures'
COLORS = ['#426ba5','#ca793c','#548b58','#9462a2']


def save(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/name,dpi=145,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    m=json.loads((OUT/'metrics.json').read_text())
    r=json.loads((OUT/'readouts.json').read_text())
    rows=json.loads((OUT/'dataset.json').read_text())
    a=np.load(OUT/'analysis_arrays.npz')
    names=[f'{k}_{l}' for k in ['activation','gradient','first_gradient'] for l in [11,17]]
    labels=['Activation\nL11','Activation\nL17','Completion ∇\nL11','Completion ∇\nL17','First-token ∇\nL11','First-token ∇\nL17']
    plt.rcParams.update({'font.size':10,'font.family':['DejaVu Sans','Noto Sans CJK JP'],
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,1,figsize=(11,8))
    x=np.arange(6)
    for ax,key,title in zip(axs,['centroid_accuracy','kmeans_ari'],['Held-out topic classification','Held-out cluster agreement (ARI)']):
        for j,(method,label,color) in enumerate([('pca','Top four PCs','#426ba5'),('full','Full vector','#548b58'),('mean_direction','Mean direction','#ca793c'),('random','Random four (32-draw mean)','#aaa')]):
            values=[np.mean([z[key] for z in m[n]['random']]) if method=='random' else m[n][method][key] for n in names]
            ax.bar(x+(j-1.5)*.2,values,width=.19,label=label,color=color)
        ax.set_xticks(x,labels)
        ax.axhline(.25 if key=='centroid_accuracy' else 0,color='black',ls=':',lw=1)
        ax.set(title=title,ylim=(-.15,1.06))
    axs[0].legend(ncol=2,fontsize=9)
    fig.suptitle('Meaningful completions: does PCA improve on direct vectors?\n32 fit rows → 16 held-out rows (8 content pairs, each in two styles)',fontsize=14)
    save(fig,'performance.png')
    groups=m['input_summary']['groups']
    fit=np.array([z['split']=='fit' for z in rows])
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,name in zip(axs.flat,names[:4]):
        s=a[name+'_scores']
        for j,group in enumerate(groups):
            idx=np.array([z['group']==group for z in rows])
            ax.scatter(s[idx&fit,0],s[idx&fit,1],c=COLORS[j],alpha=.25,s=25)
            for style,marker in [('plain','o'),('note','^')]:
                ix=idx&~fit&np.array([z['style']==style for z in rows])
                ax.scatter(s[ix,0],s[ix,1],c=COLORS[j],marker=marker,s=65,label=group if style=='plain' else None)
        ax.set(title=name.replace('_',' '),xlabel='PC1',ylabel='PC2')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Topic structure across two crossed framings\nFaint: fit · solid circles: plain held-out · triangles: note held-out')
    save(fig,'separation.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for n,label in zip(names,labels):
        vals=np.array(m[n]['eigenvalues'])
        axs[0].plot(range(1,13),np.cumsum(vals[:12])/vals.sum(),label=label.replace('\n',' '))
    axs[0].set(xlabel='PC count',ylabel='Cumulative variance',ylim=(0,1.02))
    axs[0].axvline(4,color='#aaa',ls=':')
    axs[0].legend(fontsize=8)
    axs[1].bar(x,[m[n]['pca']['style_accuracy'] for n in names],color='#9462a2')
    axs[1].axhline(.5,color='black',ls=':')
    axs[1].set_xticks(x,labels,fontsize=8)
    axs[1].set(ylabel='Held-out style classification',ylim=(0,1.05))
    fig.suptitle('Variance and framing remain separate from semantic quality')
    save(fig,'diagnostics.png')
    for layer in [11,17]:
        fig,axs=plt.subplots(1,2,figsize=(14,6.5))
        for ax,kind in zip(axs,['activation','gradient']):
            item=r[f'{kind}_{layer}_jlens']
            order=item['names']
            selected=['mean','PC1+','PC1-','PC2+','PC2-','PC3+','PC3-','PC4+','PC4-','random1+']
            text=[]
            for name in selected:
                tokens=item['tokens'][order.index(name)][:8]
                tokens=[t.replace('\n','\\n').replace('\t','\\t') for t in tokens]
                text.append(name+'   '+', '.join(tokens[:4])+'\n          '+', '.join(tokens[4:]))
            ax.text(0,1,'\n\n'.join(text),va='top',fontsize=10,transform=ax.transAxes)
            ax.set_title('Activations' if kind=='activation' else 'Gradient-derived steering directions')
            ax.axis('off')
        fig.suptitle(f'J-Lens rankings · layer {layer}\nBoth PC signs shown; mean gradient readout uses descent sign',fontsize=14)
        save(fig,f'tokens_l{layer}.png')
    fig,axs=plt.subplots(2,2,figsize=(14,8))
    for ax,name in zip(axs.flat,names[:4]):
        lines=[]
        for method in ['jlens','plain']:
            item=r[name+'_'+method]
            for label in ['astronomy-4-plain','cooking-4-plain','football-4-plain','programming-4-plain']:
                tokens=[t.replace('\n','\\n').replace('\t','\\t') for t in item['tokens'][item['names'].index(label)][:5]]
                lines.append(f'{method} · {label.split("-")[0]}\n'+', '.join(tokens))
        ax.text(0,1,'\n\n'.join(lines),va='top',fontsize=9,transform=ax.transAxes)
        ax.set_title(name.replace('_',' ')); ax.axis('off')
    fig.suptitle('Individual held-out vector baselines: first fixed content pair per topic\nAll 16 individual readouts are retained in readouts.json',fontsize=13)
    save(fig,'individual_readouts.png')


if __name__=='__main__':
    main()
