"""Plot archived pilot metrics without model access."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/2026-09-09-j-lens-pilot'
FIG = OUT/'figures'
COLORS = ['#4169ad','#cd7936','#55925d','#905b9c']


def save(fig, name):
    fig.savefig(FIG/name,dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    m=json.loads((OUT/'metrics.json').read_text())
    d=json.loads((OUT/'dataset.json').read_text())
    r=json.loads((OUT/'readouts.json').read_text())
    a=np.load(OUT/'analysis_arrays.npz')
    names=['gradient_11','gradient_17','activation_11','activation_17']
    titles=['Residual gradients · layer 11','Residual gradients · layer 17',
            'Activations · layer 11','Activations · layer 17']
    fit=np.array([row['split']=='fit' for row in d['rows']])
    labels=np.array([m['groups'].index(row['group']) for row in d['rows']])
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,name,title in zip(axs.flat,names,titles):
        score=a[name+'_scores']
        for i,group in enumerate(m['groups']):
            idx=labels==i
            ax.scatter(score[idx&fit,0],score[idx&fit,1],c=COLORS[i],alpha=.25,s=28)
            ax.scatter(score[idx&~fit,0],score[idx&~fit,1],c=COLORS[i],s=70,marker='^',
                       edgecolors='white',linewidths=.5,label=group)
        ax.set(title=title,xlabel='PC1 score',ylabel='PC2 score')
    axs[0,0].legend(fontsize=9)
    fig.suptitle('Do unseen topics separate?\nFaint circles: covariance fit · triangles: held out',fontsize=16)
    fig.tight_layout()
    save(fig,'separation.png')

    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    x=np.arange(4)
    acc=[m['analyses'][n]['evaluation']['centroid_accuracy'] for n in names]
    rnd=[m['analyses'][n]['random_accuracy_mean'] for n in names]
    ari=[m['analyses'][n]['evaluation']['kmeans_ari'] for n in names]
    rr=[m['analyses'][n]['random_ari_mean'] for n in names]
    for ax,pc,control,title in zip(axs,[acc,ari],[rnd,rr],
            ['Held-out topic classification','Held-out unsupervised cluster agreement']):
        ax.bar(x-.18,pc,width=.36,color='#4169ad',label='Top four PCs')
        ax.bar(x+.18,control,width=.36,color='#aaaaaa',label='Random 4D mean (32 draws)')
        ax.set_xticks(x,['Grad\nL11','Grad\nL17','Act\nL11','Act\nL17'])
        ax.set_title(title,fontsize=12)
    axs[0].set(ylabel='Nearest fit-centroid accuracy',ylim=(0,1.05))
    axs[0].axhline(.25,color='black',linestyle=':',linewidth=1,label='Chance')
    axs[0].legend(fontsize=8)
    axs[1].set(ylabel='Adjusted Rand index',ylim=(-.15,1.05))
    axs[1].axhline(0,color='black',linestyle=':',linewidth=1)
    fig.suptitle('48 fit inputs → 24 unseen inputs · no topic labels used to fit PCs',fontsize=14)
    fig.tight_layout()
    save(fig,'performance.png')

    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for name,title,c in zip(names,titles,COLORS):
        row=m['analyses'][name]
        axs[0].plot(np.arange(1,13),np.cumsum(row['variance_fractions'][:12]),label=title,color=c)
        history=row['streaming']['history']
        axs[1].plot([v['n'] for v in history],
                    [v['truncated_to_current_exact_overlap'] for v in history],marker='o',label=title,color=c)
    axs[0].axvline(4,color='#999999',linestyle=':')
    axs[0].set(xlabel='Number of covariance directions',ylabel='Cumulative variance explained',ylim=(0,1.02))
    axs[1].set(xlabel='Frozen-model feature rows accumulated',ylabel='Rank-4 / exact top-4 subspace overlap',ylim=(0,1.02))
    axs[0].legend(fontsize=8)
    fig.suptitle('Compact covariance is feasible; truncation need not preserve the same directions',fontsize=14)
    fig.tight_layout()
    save(fig,'spectrum_streaming.png')

    fig,axs=plt.subplots(1,2,figsize=(9,4.2))
    for ax,name in zip(axs,['gradient_11','gradient_17']):
        mat=np.array(r[name+'_jlens']['cosine'])[:4,:4]
        im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r')
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if abs(mat[i,j])>.6 else 'black')
        ax.set_xticks(range(4),['PC1','PC2','PC3','PC4'])
        ax.set_yticks(range(4),['PC1','PC2','PC3','PC4'])
        ax.set_title('Layer '+name.split('_')[-1])
    fig.colorbar(im,ax=axs,shrink=.75,label='Cosine of centered vocabulary logits')
    fig.suptitle('Orthogonal input directions can have overlapping J-Lens readouts',fontsize=14)
    save(fig,'readout_overlap.png')

    plt.rcParams['font.family']=['Noto Sans CJK JP','DejaVu Sans']
    for layer in [11,17]:
        fig,ax=plt.subplots(figsize=(13,5.6))
        ax.axis('off')
        read=r[f'gradient_{layer}_jlens']
        rows=[]
        for i in [0,8,1,9,2,10,3,11]:
            tokens=[t.replace('\n',r'\n').replace('\t',r'\t') for t in read['tokens'][i][:5]]
            rows.append([read['order'][i]]+tokens)
        table=ax.table(cellText=rows,colLabels=['Direction','Rank 1','Rank 2','Rank 3','Rank 4','Rank 5'],loc='center',cellLoc='left',colWidths=[.1,.18,.18,.18,.18,.18])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1,2)
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor('#dddddd')
            cell.set_facecolor('#eaf0f8' if i==0 else ('#f8f8f8' if i%2==0 else 'white'))
        fig.suptitle(f'Gradient covariance → J-Lens · layer {layer}\nTop tokens, both signs · unfiltered, no semantic relabeling',fontsize=16)
        save(fig,f'gradient_tokens_l{layer}.png')
    fig,axs=plt.subplots(1,2,figsize=(15,5.8))
    for ax,layer in zip(axs,[11,17]):
        ax.axis('off')
        read=r[f'activation_{layer}_jlens']
        rows=[[read['order'][i]]+[t.replace('\n',r'\n') for t in read['tokens'][i][:3]]
              for i in [0,8,1,9,2,10,3,11]]
        table=ax.table(cellText=rows,colLabels=['Direction','Rank 1','Rank 2','Rank 3'],loc='center',cellLoc='left',colWidths=[.16,.28,.28,.28])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1,2)
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor('#dddddd')
            cell.set_facecolor('#eaf0f8' if i==0 else ('#f8f8f8' if i%2==0 else 'white'))
        ax.set_title(f'Layer {layer}',fontsize=14)
    fig.suptitle('Activation covariance → J-Lens\nBoth signs · unfiltered top tokens',fontsize=17)
    fig.tight_layout()
    save(fig,'activation_tokens.png')
    print('Saved seven evidence plots.',flush=True)


if __name__=='__main__':
    main()
