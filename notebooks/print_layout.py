"""Reviewed layout treatments for dense book figures at pocket-book size.

Only presentation changes live here. Scientific arrays and scores stay in the
original generators. Dense figures use a consistent 7pt lettering profile.
"""
import re
import textwrap
from matplotlib.text import Text
DENSE = {'BERTScore_matching','CLIP_Score_matrix','Confusion_Matrix_cells',
'D2_abs_comparison','DSG_question_graph','Demographic_Parity','Dice_overlap_examples',
'Diversity_comparison','FID_mechanism','FMI_plane','Homogeneity_cluster_composition',
'IS_mechanism','LPIPS_equal_mse','MAP_precision_strips','MAUVE_frontier','MI_contingency_contributions',
'MOS_same_mean','MRR_ranked_lists','PESQ_alignment','Perplexity_surprisal','R2_explained',
'SSIM_vs_PSNR','STOI_envelopes','VQAScore_vs_CLIP','V_Measure_plane','wMAPE_compare_MAPE',
'Equality_of_Opportunity','Equality_of_Odds','Calibration_within_Groups','PPV_parity'}
def dense_figure(name):
    return name in DENSE or name.startswith('Observability_')
def wrap(label,n):
    return '\n'.join(textwrap.fill(line,n,break_long_words=False,break_on_hyphens=False)
                     for line in label.split('\n'))
def arrange_for_print(fig,name):
    if getattr(fig,'_book_reflowed',False): return
    fig._book_reflowed=True
    # Figure-wide legends require more than one row at physical print size.
    for legend in list(fig.legends):
        labels=[t.get_text() for t in legend.get_texts()]
        handles=legend.legend_handles
        if name == 'Calibration_within_Groups':
            legend.remove()
            fig.axes[0].legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,-.08), ncol=2, frameon=False)
        elif sum(map(len,labels))>65 or name=='Homogeneity_cluster_composition':
            legend.remove()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.035),
                       ncol=1 if name=='Dice_overlap_examples' else 2,frameon=False)
    if dense_figure(name) and name not in {'Demographic_Parity','Equality_of_Opportunity','Equality_of_Odds','Dice_overlap_examples','R2_explained','D2_abs_comparison','V_Measure_plane','FMI_plane'}:
        fig.set_figheight(fig.get_figheight()*1.35)
    if name=='VQAScore_vs_CLIP': fig.set_figheight(fig.get_figheight()*1.5)
    if name=='SSIM_vs_PSNR' and fig._suptitle: fig._suptitle.set_visible(False)
    if name=='DSG_question_graph':
        for ax in fig.axes:
            for t in ax.texts:
                if t.get_text().startswith('prompt:'):t.set_visible(False)
        fig.suptitle('Prompt: a red bicycle leaning against a blue wall',y=1.04)
    if name=='BERTScore_matching':
        fig.axes[1].remove();fig.axes[0].set_position([.17,.12,.80,.70])
        fig.set_size_inches(7.2,5.3)
    for ax in fig.axes:
        if name in {'Pair_Confusion_Matrix', 'Contingency_Matrix_heatmap'} and ax.images:
            # Choose ink from the rendered cell colour, not from its count.
            # The shared colormap is not monotonic in luminance.
            im = ax.images[0]
            for text in ax.texts:
                x, y = map(round, text.get_position())
                rgb = im.cmap(im.norm(im.get_array()[y, x]))[:3]
                linear = [c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in rgb]
                luminance = sum(c*w for c,w in zip(linear, [.2126,.7152,.0722]))
                text.set_color('black' if luminance > .179 else 'white')
        if name in {'MI_contingency_contributions','Homogeneity_cluster_composition','MOS_same_mean','IS_mechanism','wMAPE_compare_MAPE','FID_mechanism'}:
            subtitles=[t for t in ax.texts if t.get_transform()==ax.transAxes and 1.0<t.get_position()[1]<1.1]
            for t in subtitles:
                title=ax.get_title()+'\n'+wrap(t.get_text(),25)
                ax.set_title(title,pad=12)
                t.set_visible(False)
        if name=='Observability_expected_range' and ax == fig.axes[0]:
            for text in ax.texts:
                if text.get_text() == 'weekday\nmean':
                    point = text.get_position()
                    text.set_visible(False)
                    ax.annotate('weekday mean', xy=(6, point[1]), xytext=(5.7, 1900),
                                ha='right', va='center', color='#333333',
                                arrowprops=dict(arrowstyle='-', color='#888888', lw=.8))
        if name=='IS_mechanism':
            ax.set_xticks(range(4),labels=['1','2','3','4'])
            ax.set_xlabel('Class',labelpad=5)
            ax.set_yticks([0,1,2,3,5],labels=['1','2','3','4','p(y)'])
            for t in ax.texts:
                if t.get_text().startswith('mean KL('):t.set_text('mean KL = '+t.get_text().split(' = ')[-1])
        if name=='CLIP_Score_matrix':
            for t in ax.get_xticklabels():t.set_text(wrap(t.get_text(),12))
            ax.set_xticks(ax.get_xticks(),labels=[wrap(t.get_text(),12) for t in ax.get_xticklabels()])
        if name=='PESQ_alignment':
            ax.set_xlabel('Waveform SNR (dB)' if ax==fig.axes[0] else 'PESQ (MOS-LQO)')
        if name=='MAUVE_frontier' and 'KL' in ax.get_xlabel():
            ax.set_xlabel(r'$\exp[-c\,\mathrm{KL}(Q\Vert R_\lambda)]$')
            ax.set_ylabel(r'$\exp[-c\,\mathrm{KL}(P\Vert R_\lambda)]$')
        for t in list(ax.texts):
            label=t.get_text()
            if name=='LPIPS_equal_mse':
                if label.startswith('MSE ') and '(PSNR' in label:
                    t.set_text(label.replace('  (PSNR','\nPSNR').replace(')',''))
                if label.startswith('LPIPS '):t.set_y(-.43)
            if name in {'FMI_plane','V_Measure_plane'} and 'one cluster' in label:
                t.set_text(label.replace(':  ',':\n'));t.set_ha('left')
            if name in {'FMI_plane','V_Measure_plane'} and 'well separated' in label:
                t.set_text(label.replace(':  ',':\n'))
            if name in {'R2_explained','D2_abs_comparison','V_Measure_plane','FMI_plane'}:
                label=re.sub(r'= sum of (?:purple-square areas|red-square areas|purple bar heights|red bar heights) =', '=',label)
                if len(label)>65 and 'outlier' in label: label='Squared errors amplify this outlier more than absolute errors.'
                t.set_text(label)
            if name=='FMI_plane':
                if 'one cluster' in label or 'well separated' in label:
                    t.set_text(label.replace(':  ', ':\n'))
                    t.set_ha('center' if 'one cluster' in label else 'right')
                elif label.startswith('FMI ='):
                    t.set_text(label.split('=')[-1].strip())
                elif label.startswith('F1 ='):
                    t.set_bbox(dict(facecolor='white', edgecolor='none', pad=.5))
                if 'overlapping' in label:
                    t.set_bbox(dict(facecolor='white', edgecolor='none', pad=.5))
            if name=='CG_order_blind' and label.startswith('relevance grade'):
                t.set_visible(False) # caption defines bar height
            if name=='Confusion_Matrix_cells':
                t.set_text(label.replace('across the row:\n','').replace('down the column:\n','').replace('diagonal over everything:\n',''))
            if name=='MRR_ranked_lists':
                t.set_text(label.replace('first relevant (MRR counts this)','first relevant').replace('additional relevant (ignored)','later relevant'))
            if name=='MAP_precision_strips':
                if label.startswith('AP = mean'):t.set_text('AP = '+label.split('=')[-1].strip())
                if label.startswith('number below relevant'):t.set_visible(False)
            if name=='MAUVE_frontier' and label.startswith('share of texts:'):
                t.set_text('human P (gray), model Q');t.set_y(.37)
            if name=='Perplexity_surprisal':
                if 'after “Ohio”' in label or 'p = ' in label:t.set_visible(False)
                if label.startswith('“'):t.set_text(wrap(label,32))
            if name=='VQAScore_vs_CLIP':
                if t.get_fontstyle()=='italic': t.set_visible(False)
                elif '“' in label:t.set_text(wrap(label,27))
            if name.startswith('Observability_') and len(label)>27:
                t.set_text(wrap(label,27))
    if name=='Confusion_Matrix_cells':
        for t in fig.texts:t.set_visible(False) # repeats the caption
