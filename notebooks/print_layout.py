"""Reviewed layout treatments for dense book figures at pocket-book size.

Only presentation changes live here. Scientific arrays and scores stay in the
original generators. Dense figures use a consistent 7.2pt lettering profile.
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
'Predictive_Parity','TER_edit_breakdown','Equality_of_Opportunity','Equality_of_Odds','Calibration_within_Groups','PPV_parity'}
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
    if dense_figure(name) and name not in {'Demographic_Parity','Equality_of_Opportunity','Equality_of_Odds','Dice_overlap_examples','R2_explained','D2_abs_comparison','V_Measure_plane','FMI_plane','Predictive_Parity'}:
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
    if name == 'D2_abs_comparison':
        for t in fig.texts:
            if t.get_text().startswith('on the same data,'):
                t.set_text(t.get_text().split(' — ')[0] + '\nSquared errors amplify this outlier more.')
        for ax in fig.axes:
            if ax.get_title(loc='left').startswith('baseline:'):
                ax.set_title('baseline: predict the median', loc='left')
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
        if name in {'Equality_of_Opportunity','Equality_of_Odds'}:
            for t in ax.texts:
                if 'qualified (top block)' in t.get_text():
                    t.set_text(t.get_text().replace(' (top block)', ' above').replace(' (bottom block)', ' below'))
                    t.set_y(1.4)
                elif t.get_text() in {'Group A', 'Group B'}:
                    t.set_y(4.2)
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
            if name=='TER_edit_breakdown':
                if label.startswith('as plain word edits:'):
                    t.set_text(label.replace('as plain word edits:  ', 'Plain word edits: ').replace(r'   $\rightarrow$   ', '\n'))
                    t.set_y(-.55)
                elif label.startswith('with a shift:'):
                    t.set_text(label.replace('with a shift:  ', 'With a shift: ').replace(r'   $\rightarrow$   ', '\n'))
                    t.set_y(-1.65)
            if name=='LPIPS_equal_mse':
                if label.startswith('MSE ') and '(PSNR' in label:
                    t.set_text(label.replace('  (PSNR','\nPSNR').replace(')',''))
                if label.startswith('LPIPS '):t.set_y(-.43)
            if name in {'FMI_plane','V_Measure_plane'} and 'one cluster' in label:
                t.set_text(label.replace(':  ',':\n'));t.set_ha('left')
            if name in {'FMI_plane','V_Measure_plane'} and 'well separated' in label:
                t.set_text(label.replace(':  ',':\n'))
            if name in {'R2_explained','D2_abs_comparison','V_Measure_plane','FMI_plane','Predictive_Parity'}:
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
                if label.startswith('VQAScore  ') or label == 'CLIP-S': t.set_x(.45)
                if t.get_fontstyle()=='italic': t.set_visible(False)
                elif '“' in label:t.set_text(wrap(label,27))
            if name.startswith('Observability_') and len(label)>27:
                t.set_text(wrap(label,27))
    if name=='Confusion_Matrix_cells':
        for t in fig.texts:t.set_visible(False) # repeats the caption
    hardcover_labels(fig, name)


def hardcover_labels(fig, name):
    """Separate labels at the 109.7 mm hardcover text width; preserve data."""
    if name == 'D2_abs_comparison':
        for t in fig.texts:
            if t.get_text().startswith('on the same data,'): t.set_y(-.18)
    if name == 'Diversity_comparison':
        ax = fig.axes[0]
        labels = [t for t in ax.texts if t.get_text() in ['Action','Comedy','Drama','Sci-Fi','Horror','Romance']]
        swatches = [p for p in ax.patches if p.get_y() == -1.0]
        for j,(t,p) in enumerate(zip(labels, swatches)):
            x = .4 + (j%3)*3.2; y = -1.0 - (j//3)*.7
            p.set_xy((x,y));p.set_clip_on(False);t.set_clip_on(False);t.set_position((x+.55,y+.2))
    if name == 'Observability_schema':
        ax=fig.axes[1]; y=.95
        for t in ax.texts:
            label=t.get_text()
            if label.startswith('~'):
                label=label.replace(' → ', '\n→ ')
                t.set_text(label)
            t.set_y(y);y -= .18 if label.startswith('scan') else .21 if '\n' in label else .14
    for ax in fig.axes:
        if name == 'BLEU_ngram_precision' and ax.get_xlabel() == '':
            ticks=ax.get_xticks()
            if len(ticks)==4:
                ax.set_xticks(ticks,labels=['1','2','3','4'])
                ax.set_xlabel('n-gram length (n)')
        if name == 'Homogeneity_cluster_composition':
            labels=[t.get_text().replace('cluster ', '') for t in ax.get_xticklabels()]
            ax.set_xticks(ax.get_xticks(),labels=labels);ax.set_xlabel('')
        if name == 'PESQ_alignment' and ax==fig.axes[-1]:ax.set_xticks([1,2,3,4.5])
        if name == 'MAUVE_frontier' and 'KL' in ax.get_ylabel():
            ax.yaxis.set_label_position('right')
        if name == 'Observability_expected_range' and ax==fig.axes[0]:
            ax.set_xticks(ax.get_xticks(),labels=['M','Tu','W','Th','F','Sa','Su'])
        if name == 'wMAPE_compare_MAPE':
            ax.set_title(wrap(ax.get_title(),24),pad=12)
        for t in ax.texts:
            label=t.get_text()
            if name == 'Confusion_Matrix_cells' and ax==fig.axes[0] and label=='predicted':
                t.set_text('pred.')
            if name == 'DSG_question_graph':
                if label=='Is the bicycle\nred?':t.set_text('Bicycle\nred?')
                elif label=='Bicycle leaning\non the wall?':t.set_text('Leaning on\nthe wall?')
            if name == 'FID_mechanism' and label.startswith('two clouds,'):
                t.set_text(label.replace(', same', ',\nsame'))
            if name == 'MI_contingency_contributions' and 'count' in label and '$' in label:
                t.set_text('count\ncontribution\n(nats)');t.set_fontstyle('normal')
            if name == 'Homogeneity_cluster_composition' and label.startswith('H = '):
                t.set_text(label.replace('H = ', ''))
            if name == 'MOS_same_mean' and label.startswith('median'):
                t.set_text(label.replace(' ±', '\n±'));t.set_linespacing(1.2)
            if name == 'MOS_same_mean' and label.startswith('20 listeners'):
                t.set_position((.02,.48));t.set_text('20 people\n20 dots')
            if name == 'Predictive_Parity' and 'truly qualified' in label:
                counts,score = label.split('  →  ')
                a,b = re.findall(r'\d+', counts)
                t.set_text(score.replace(' = ', f' = {a}/{b} = '))
            if name == 'Perplexity_surprisal' and label.startswith('mean ') and 'PPL' in label:
                t.set_text(label.replace('  →  ', '\n').replace(' → ', '\n'))
            if name == 'P4_vs_F1' and label.startswith('gap ='):
                t.set_position((.4,.18));t.set_ha('left')
            if name == 'ROUGE_variants' and label=='concise':t.set_position((9.5,.9));t.set_ha('left')
            if name == 'ROUGE_variants' and label.startswith('padding the summary'):
                t.set_text(wrap(label,28));t.set_y(.92)
            if name == 'STOI_envelopes' and label.startswith('STOI ='):
                t.set_y(1.035)
            if name.startswith('Observability_') and hasattr(t,'xy'):
                if name=='Observability_average':
                    if label.startswith('refunds:'):t.set_position((-170,80));t.set_ha('right')
                    if label.startswith('half price'):t.set_position((-5,50));t.set_va('bottom');t.set_ha('right')
                elif name=='Observability_stddev':
                    if label.startswith('refunds:'):t.set_position((-20,42));t.set_ha('right')
                    if label.startswith('half price'):t.set_position((-5,-75));t.set_va('top');t.set_ha('right')
                elif name=='Observability_sum' and label.startswith('no load'):
                    t.anncoords='axes fraction';t.set_position((.49,.08));t.set_va('bottom');t.set_ha('right')
                elif name=='Observability_quartiles' and label.startswith('half price'):
                    t.set_text('Half price under 30:\nQ1 drops; median and Q3 do not')
                    t.set_position((25,8));t.set_va('bottom');t.set_ha('center')
    if name == 'Predictive_Parity':
        for ax in fig.axes:
            for t in ax.texts:
                if t.get_text() in ['Group A','Group B']:t.set_y(3.5)
