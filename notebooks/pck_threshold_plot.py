"""Computed schematic: the same keypoints under two normalized tolerances.

Coordinates are constructed teaching data in arbitrary equal-aspect units.
L is the neck-to-pelvis length; all radii, pass/fail colours and scores come
from these arrays. This is not a benchmark-specific skeleton or PCK protocol.
"""
from style import *
from matplotlib.patches import Circle
import json
truth=np.array([[0,1.7],[-.35,1.6],[.35,1.6],[-.50,1.1],[.50,1.1],
                [-.60,.6],[.60,.6],[0,.7],[-.20,.65],[.20,.65],
                [-.20,0],[.20,0],[-.20,-.6],[.20,-.6]])
pred=truth+np.array([.04,.03])
pred[1]=truth[1]+[-.25,.15]
pred[5]=truth[5]+[-.55,.05]
pred[6]=truth[6]+[.12,-.08]
pred[13]=truth[13]+[.32,0]
edges=[(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(0,7),(7,8),(7,9),(8,10),(9,11),(10,12),(11,13)]
L=float(np.linalg.norm(truth[0]-truth[7]));distance=np.linalg.norm(pred-truth,axis=1)
fig,axes=plt.subplots(1,3,figsize=(7.2,4.5))
for i,ax in enumerate(axes):
    for a,b in edges:ax.plot(*truth[[a,b]].T,color='#92959A',lw=1.2,zorder=1)
    ax.scatter(*truth.T,s=15,color='#44474D',zorder=3)
    ax.add_patch(Circle((0,2.03),.16,fill=False,edgecolor='#92959A',lw=1.2))
    if i==0:
        ax.annotate('',(.88,1.7),(.88,.7),arrowprops=dict(arrowstyle='<->',color=NML_PURPLE,lw=1.4))
        ax.plot([0,.84],[1.7,1.7],lw=.6,color=NML_PURPLE,ls=':')
        ax.plot([0,.84],[.7,.7],lw=.6,color=NML_PURPLE,ls=':')
        ax.text(.95,1.2,'L',va='center',color=NML_PURPLE)
        ax.set_title('Reference',pad=12)
    else:
        alpha=[.2,.5][i-1];correct=distance<=alpha*L
        for a,b in edges:ax.plot(*pred[[a,b]].T,color=NML_CYAN,lw=.9,alpha=.6,zorder=2)
        ax.scatter(*pred.T,s=20,c=np.where(correct,NML_CYAN,NML_RED),zorder=4)
        for j in [1,6,13]:ax.add_patch(Circle(truth[j],alpha*L,fill=False,ec=NML_PURPLE,lw=1))
        ax.set_title(f'α = {alpha:.1f}\n{correct.sum()}/{len(correct)} correct',pad=12)
    ax.set(xlim=(-1.3,1.25),ylim=(-1.2,2.25),aspect='equal');ax.axis('off')
fig.subplots_adjust(left=0,right=1,top=.83,bottom=.16,wspace=.12)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#44474D',label='Reference joint'),
                    Line2D([],[],marker='o',ls='',color=NML_CYAN,label='Within tolerance'),
                    Line2D([],[],marker='o',ls='',color=NML_RED,label='Outside tolerance')],
           loc='lower center',bbox_to_anchor=(.5,.02),ncol=1,frameon=False)
assert L==1 and [(distance<=a*L).sum() for a in [.2,.5]]==[11,13]
save_figure(fig,'PCK_threshold_visual')
print(json.dumps({'reference_length':L,'distances':distance.tolist(),'alpha':[.2,.5],'correct':[11,13],'N':len(truth)}))
