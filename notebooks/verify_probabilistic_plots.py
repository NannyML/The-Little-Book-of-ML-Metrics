"""Reproduce and independently check the five educational probabilistic figures.

Run: uv run python notebooks/verify_probabilistic_plots.py [--render] [--update-captions]
The default checks current source and records the existing artifact hashes.
Use --render to synchronize all five PNGs with the checked experiment; caption updates replace only the five caption bodies, never formulas.
This is an algorithm/illustration check, not a NannyML package parity test.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
import sklearn
import matplotlib
from matplotlib.text import Text
from PIL import Image

import probabilistic_plots as p

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'design/probabilistic-verification/results.json'
NAMES = dict(ece='ECE_reliability', cbpe='CBPE_estimation', pape='PAPE_reweighting',
             dle='DLE_nanny', rcd='RCD_decomposition')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rowkeys(a):
    a = np.ascontiguousarray(np.asarray(a).reshape(len(a), -1))
    return a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1]))).ravel()


def separate(*arrays):
    for i, a in enumerate(arrays):
        for b in arrays[i+1:]:
            assert not len(np.intersect1d(rowkeys(a),rowkeys(b))), 'Reused input rows'
    return [{'rows':len(a),'input_sha256':digest(a)} for a in arrays]


def expected_accuracy(c, labels):
    # Independently sum the expected confusion diagonal, rather than invoking p.
    return (sum(c[labels == 1])+sum((1-c)[labels == 0]))/len(labels)


def numeric_report(d):
    out={}
    e=d['ece'];bin_ece=[]
    for _,score,bins in e['panels']:
        independent=sum(share*abs(acc-conf) for _,_,share,acc,conf in bins if share)
        np.testing.assert_allclose(independent,score,atol=1e-15)
        assert all(e['axes'][0].get_ylim()[0]<=r[3]<=e['axes'][0].get_ylim()[1] for r in bins if r[2])
        bin_ece.append(score)
    splits=separate(e['x'][:20000],e['x'][20000:30000],e['x'][30000:])
    out['ece']={'seed':1,'roles':dict(zip(['child_training','temperature_fit','evaluation'],splits)),
                'ece':bin_ece,'temperature':e['T'],'auc':e['auc'],
                'first_bin':list(e['panels'][0][2][0]),'top_bin':list(e['panels'][0][2][-1]),
                'axis_y_limits':list(e['axes'][0].get_ylim())}
    c=d['cbpe'];roles=separate(c['x_train'],c['x_ref'],c['x_val'],c['x'])
    acc=expected_accuracy(c['c'],c['yhat']);np.testing.assert_allclose(acc,c['p_correct'].mean())
    # Independently replay the first illustration draw, with no outcome selection.
    xr,yr=p.sample_world(12,.5,.6,rng=np.random.default_rng(2004))
    np.testing.assert_array_equal(xr,c['x']);np.testing.assert_array_equal(yr,c['y'])
    rng=np.random.default_rng(2);p.sample_world(20000,0,1,rng=rng)
    realized=[];estimates=[]
    for centre,concept in zip(c['centres'],c['concepts']):
        x,y=p.sample_world(2000,centre,.55,concept,rng=rng)
        s=c['model'].predict_proba(x[:,None])[:,1];lab=s>=.5
        estimates.append(expected_accuracy(c['cal'].predict(s),lab))
        realized.append(sum(lab==y)/len(y))
    np.testing.assert_allclose(estimates,c['est']);np.testing.assert_allclose(realized,c['realized'])
    out['cbpe']={'seeds':{'child_training':2002,'reference_and_chunks':2,'validation':2003,'illustration_first_draw':2004},
                 'roles':dict(zip(['child_training','reference','validation','illustration'],roles)),
                 'illustration_draws':1,'illustration_expected':acc,
                 'illustration_correct_count':int(sum(c['yhat']==c['y'])),
                 'illustration_realized':float(np.mean(c['yhat']==c['y'])),
                 'reference_accuracy':c['ref_acc'],'illustrative_band_halfwidth':c['band'],
                 'production_realized':realized,'production_estimates':estimates,'validation':c['validation']}
    c=d['pape'];roles=separate(c['X_train'],c['X_ref'],c['X_val'])
    rng=np.random.default_rng(3);c['sample'](8000,0,rng=rng)
    for centre,realized,ec,ep,X,w in c['rows']:
        xr,yr=c['sample'](2500,centre,rng=rng);np.testing.assert_array_equal(xr,X)
        s=c['model'].predict_proba(X)[:,1];lab=s>=.5
        assert np.isclose(sum(lab==yr)/len(yr),realized)
        weighted=p.fit_calibrator(c['s_ref'],c['y_ref'],w)
        assert np.isclose(expected_accuracy(weighted.predict(s),lab),ep)
    domain_prob=np.clip(c['dre'].predict_proba(c['X_ref'])[:,1],.001,.999)
    independent_weights=len(c['X_ref'])/len(c['X_last'])*domain_prob/(1-domain_prob)
    np.testing.assert_allclose(independent_weights,c['w_last'])
    out['pape']={'seeds':{'child_training':3003,'reference_and_chunks':3,'validation':3004,'domain_classifier':0},
                 'roles':dict(zip(['child_training','reference','validation'],roles)),
                 'production_realized':c['r'].tolist(),'cbpe_estimates':c['ec'].tolist(),'pape_estimates':c['ep'].tolist(),
                 'mae_cbpe':float(np.mean(abs(c['ec']-c['r']))),'mae_pape':float(np.mean(abs(c['ep']-c['r']))),
                 'binned_weight_peak':float(np.nanmax(c['wbin'])),'maximum_raw_weight':float(c['w_last'].max()),
                 'effective_reference_sample_size':float(c['w_last'].sum()**2/sum(c['w_last']**2)),
                 'validation':c['validation']}
    c=d['dle'];roles=separate(c['x_train'],c['x_ref'],c['x_val'])
    rng=np.random.default_rng(4);rng.uniform(0,1,6000);rng.normal(0,1,6000)
    min_loss=float(c['hs'].min());r=[];est=[]
    for centre in c['centres']:
        x=np.clip(rng.normal(centre,.12,2000),0,1);y=2*x+rng.normal(0,1,len(x))*x
        f=c['child'].predict(x[:,None]);raw=c['nanny'].predict(np.column_stack([x,f]));loss=np.maximum(0,raw)
        min_loss=min(min_loss,float(loss.min()))
        r.append(sum(abs(y-f))/len(y));est.append(sum(loss)/len(loss))
    np.testing.assert_allclose(r,c['r']);np.testing.assert_allclose(est,c['e'])
    assert min_loss>=0
    np.testing.assert_allclose(c['hs'],np.maximum(0,c['nanny'].predict(np.column_stack([c['xs'],c['fs']]))))
    out['dle']={'seeds':{'child_training':4004,'reference_and_chunks':4,'validation':4005},
                'roles':dict(zip(['child_training','reference','validation'],roles)),
                'production_realized':r,'production_estimates':est,'maximum_absolute_gap':float(np.max(abs(c['r']-c['e']))),
                'minimum_predicted_loss':min_loss,'raw_prediction_at_zero':float(c['nanny'].predict([[0,c['child'].predict([[0]])[0]]])[0]),
                'validation':c['validation']}
    c=d['rcd'];roles=separate(c['X_train'],c['X_ref'],c['X_mon'],c['X_val'])
    v=expected_accuracy(c['p_new'],c['yhat_ref']);assert np.isclose(v,c['est_under_new'])
    assert np.isclose(c['acc_ref']+(c['acc_cov']-c['acc_ref'])+c['impact']+c['resid'],c['acc_mon'])
    cp=c['concept'];ref_old=expected_accuracy(cp(c['X_ref'],0),c['yhat_ref']);ref_new=expected_accuracy(cp(c['X_ref'],c['rot']),c['yhat_ref'])
    mon_old=expected_accuracy(cp(c['X_mon'],0),c['yhat_mon']);mon_new=expected_accuracy(cp(c['X_mon'],c['rot']),c['yhat_mon'])
    out['rcd']={'seeds':{'child_training':5005,'reference_and_monitoring':5,'validation':5006},
                'roles':dict(zip(['child_training','reference','monitored','validation'],roles)),
                'reference_accuracy':c['acc_ref'],'covariate_only_accuracy':c['acc_cov'],
                'covariate_effect':c['acc_cov']-c['acc_ref'],'concept_impact':c['impact'],
                'sum_before_residual':c['acc_cov']+c['impact'],'residual':c['resid'],
                'monitored_accuracy':c['acc_mon'],'magnitude':c['magnitude'],
                'oracle_concept_impact':ref_new-ref_old,'oracle_interaction':mon_new-mon_old-ref_new+ref_old,
                'concept_model_expected_accuracy_error':v-ref_new,'validation':c['validation']}
    return out


def captions(n):
    e,c,pap,d,r=(n[k] for k in ['ece','cbpe','pape','dle','rcd'])
    # Counts/decimals below are values from the same experiment used for the PNG.
    count=c['illustration_correct_count']
    count_text='all twelve' if count==12 else f'${count}$ of twelve'
    return {
      'ECE':rf'''Reliability diagrams for a simulated binary classifier before and after temperature scaling, which divides
its raw scores (logits) by a positive value $T$. Bar height shows observed accuracy, the numbers inside the bars show
the percentage of predictions in each bin, red segments show calibration gaps, and the dashed diagonal marks perfect calibration.
\textit{{Left:}} the top bin contains ${100*e['top_bin'][2]:.0f}\%$ of predictions, with mean confidence ${e['top_bin'][4]:.2f}$ and
accuracy ${e['top_bin'][3]:.2f}$; the weighted gaps sum to ECE $={e['ece'][0]:.3f}$.
\textit{{Right:}} with $T={e['temperature']:.1f}$, ECE falls to ${e['ece'][1]:.3f}$. ROC AUC stays at ${e['auc']:.3f}$ because
ranking is unchanged in this binary example.''',
      'CBPE':rf'''\textit{{Left:}} the first seeded draw of twelve production predictions, with no selection using their outcomes.
Each bar gives the chance of being right: $c$ for $\hat{{y}}=1$, or $1-c$ for $\hat{{y}}=0$. Their mean is ${c['illustration_expected']:.2f}$;
the labels that arrive later make {count_text} correct, an accuracy of ${c['illustration_realized']:.2f}$.
\textit{{Right:}} twelve simulated chunks of $2{{,}}000$ predictions. Accuracy rises as the input mix changes; from chunk $8$,
concept drift lowers accuracy to ${c['production_realized'][-1]:.2f}$ while the estimate stays near ${np.mean(c['production_estimates'][7:]):.2f}$.
The illustrative band is $\pm 3$ reference-chunk standard deviations around the estimate, not a validated uncertainty interval or fixed alert threshold.''',
      'PAPE':rf'''A simulated classifier whose calibration varies with input $x_2$.
\textit{{Left:}} reference and final-production densities are shown above the mean reference weight in each $x_2$ bin. The
weight peaks near ${pap['binned_weight_peak']:.0f}$ in the production region; bins with fewer than $30$ reference points are not drawn.
\textit{{Right:}} fixed reference calibration keeps CBPE near ${np.mean(pap['cbpe_estimates']):.2f}$ as realized accuracy reaches
${pap['production_realized'][-1]:.2f}$. PAPE follows it, with a mean absolute error of ${pap['mae_pape']:.3f}$ versus CBPE's ${pap['mae_cbpe']:.3f}$.''',
      'DLE':rf'''\textit{{Left:}} simulated reference data with noise that grows with the input, a linear model $f$, and a loss
predictor $h$ whose outputs are constrained to be nonnegative. The band $f(x)\pm h(x)$ illustrates estimated absolute-error size;
it has no stated coverage probability. Two absolute errors used to train $h$ are marked in red.
\textit{{Right:}} eight unseen production chunks move toward noisier inputs. Measured MAE rises from ${d['production_realized'][0]:.2f}$ to
${d['production_realized'][-1]:.2f}$; the maximum gap from DLE is ${d['maximum_absolute_gap']:.3f}$ in this example.''',
      'RCD':rf'''\textit{{Left:}} simulated reference inputs, the monitored model's boundary (black), and the new concept model's
boundary (red). \textit{{Right:}} accuracy falls from ${r['reference_accuracy']:.3f}$ to ${r['monitored_accuracy']:.3f}$.
Moving the inputs under the old concept gives ${round(r['reference_accuracy']+r['covariate_effect'],3)-round(r['reference_accuracy'],3):+.3f}$; applying the new concept to reference inputs gives
${round(r['sum_before_residual'],3)-round(r['reference_accuracy']+r['covariate_effect'],3):+.3f}$. Together these reach ${r['sum_before_residual']:.3f}$. The explicit residual step, ${round(r['monitored_accuracy'],3)-round(r['sum_before_residual'],3):+.3f}$,
accounts for the remaining difference. Interaction between shifts, concept-model error and sampling variation can all contribute;
these effects need not add independently.'''}


def update_caption_bodies(new):
    path=ROOT/'book/9-probabilistic.tex'
    # Read immediately before editing; other agents own the formula blocks.
    source=path.read_text()
    for name,text in new.items():
        token='\\metriccaption{Figure. '+name+'.}{';start=source.index(token)+len(token)
        pos=start;depth=1
        while depth:
            ch=source[pos]
            if ch in '{}' and (pos==0 or source[pos-1]!='\\'):depth += 1 if ch=='{' else -1
            pos+=1
        source=source[:start]+text+source[pos-1:]
    path.write_text(source)


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--render',action='store_true');cli.add_argument('--update-captions',action='store_true');args=cli.parse_args()
    typography={};original_save=p.save_figure
    def capture(fig,name,**kwargs):
        if args.render:original_save(fig,name,**kwargs)
        else:p.apply_plot_typography(fig)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        width_in=fig.get_tightbbox(renderer).width
        fonts=[t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        size=min(fonts)*p.PRINT_WIDTH_IN/width_in
        typography[name]={'source_minimum_pt':min(fonts),'tight_width_inches':width_in,'minimum_printed_pt':size}
        assert size>=6.5,(name,size)
    p.save_figure=capture
    out=io.StringIO()
    with contextlib.redirect_stdout(out):d={key:getattr(p,'fig_'+key)() for key in NAMES}
    numbers=numeric_report(d);new_captions=captions(numbers)
    if args.update_captions:update_caption_bodies(new_captions)
    artifacts={}
    for name in NAMES.values():
        path=ROOT/'book/figures'/f'{name}.png'
        artifacts[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pixels':list(Image.open(path).size),**typography[name]}
    report={'status':'passed','kind':'seeded educational reimplementation; not package parity',
            'original_proof_commit':'8e3df1f','scientific_audit_date':'2026-09-27',
            'rendered_in_this_run':args.render,'updated_captions_in_this_run':args.update_captions,
            'source_sha256':hashlib.sha256((ROOT/'notebooks/probabilistic_plots.py').read_bytes()).hexdigest(),
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__},
            'checks':['Disjoint child-training, reference and validation input rows','Separate ECE temperature-fit/evaluation rows','First unfiltered CBPE illustration draw reproduced','Independent expected-confusion accuracy','Independent seeded production replay','PAPE domain-prior correction and weighted calibration','Same nonnegative DLE predictions in envelope and chunks','RCD accounting reconciles exactly; oracle interaction measured','Minimum printed text >=6.5pt','Caption values generated from checked experiment values'],
            'intentional_simplifications':['Always-fit isotonic calibration rather than NannyML calibration-benefit selection','scikit-learn gradient boosting domain classifier for PAPE','Linear DLE loss predictions clipped to zero','Polynomial logistic RCD concept model, not NannyML LightGBM','Illustrative CBPE reference-variation band; no interval coverage or threshold claim'],
            'sources':['https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html','https://docs.nannyml.com/cloud/model-monitoring/how-it-works/probabilistic-adaptive-performance-estimation-pape','https://docs.nannyml.com/cloud/model-monitoring/how-it-works/reverse-concept-drift-rcd','https://proceedings.mlr.press/v70/guo17a.html'],
            'numbers':numbers,'artifacts':artifacts,'captions':new_captions,'render_log':out.getvalue()}
    REPORT.parent.mkdir(exist_ok=True,parents=True);REPORT.write_text(json.dumps(report,indent=2,default=float)+'\n')
    print(out.getvalue());print('Verified:',REPORT)


if __name__=='__main__':main()
