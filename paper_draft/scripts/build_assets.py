"""Build paper tables, bibliography, and vector figures from frozen local artifacts."""
from pathlib import Path
import csv, json, html, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
S = json.loads((ROOT/'results/summary.json').read_text())
A = json.loads((ROOT/'results/audited_summary.json').read_text())
M = list(csv.DictReader((ROOT/'results/detector_metrics.csv').open()))
F = list(csv.DictReader((ROOT/'results/subgroup_flags.csv').open()))

def table(name, columns, header, rows, caption, label, wide=False):
    body = '\n'.join(' & '.join(map(str,row)) + r' \\' for row in rows)
    tab = '\\begin{tabular}{@{}'+columns+'@{}}\n\\toprule\n'+header+'\n\\midrule\n'+body+'\n\\bottomrule\n\\end{tabular}'
    if wide: tab = '\\resizebox{\\textwidth}{!}{%\n'+tab+'%\n}'
    out='\\begin{table}[t]\n\\centering\n\\small\n'+tab+'\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{table}\n'
    (P/'tables'/f'{name}.tex').write_text(out)

def pctci(value, ci): return f'{100*value:.2f} [{100*ci[0]:.2f}, {100*ci[1]:.2f}]'
rows=[]
for r in S['arms']:
    rows.append([r['arm'].capitalize(), r['correct'],r['false'],r['abstain'],r['candidate_misreport'],r['verified_misreport'],pctci(r['false_rate'],r['false_ci'])])
table('behavior','lrrrrrl',r'Arm & Correct & False & Abstain & Cand. & Rec. & False rate (\%) [95\% CI] \\', rows,
      r'Strict behavioral outcomes, 600 responses per arm. Cand. and Rec. count incentive candidates and recovered proxies, respectively, so they exclude the instructed control. False-rate intervals use question bootstrap. The instructed arm has 39 false answers on known questions, 16 with correct follow-ups; these are separate controls.', 'tab:behavior', True)
names=['Recovered incentive misreport proxy','Uncertain neutral error proxy','Repeats stable wrong belief','Unresolved false output']
rows=[]
for name,s,a in zip(names,S['shares_equal_four_arm_mixture'],A['shares']):
    rows.append([name,s['count'],pctci(s['share'],s['ci']),a['count'],pctci(a['share'],a['ci'])])
table('composition','lr l r l',r'& \multicolumn{2}{c}{Strict ($n=939$)} & \multicolumn{2}{c}{Audited ($n=813$)} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}'+'\n'+r'Category & Count & Share (\%) [95\% CI] & Count & Share (\%) [95\% CI] \\',rows,
      r'Operational composition of all false outputs in the equal neutral/sham/reward/reputation mixture. The instructed arm is excluded. Intervals are question-bootstrap 95\% intervals, conditional on observed cases. These labels leave most false outputs unresolved and do not establish latent mechanisms.', 'tab:composition',True)
cs=['candidate_mismatch_vs_neutral_H_UNVERIFIED','instructed_false_vs_neutral_H_CONTROL','neutral_error_vs_correct']
methods=[('deception_response_L14','Instructed deception, response'),('deception_prompt_L14','Instructed deception, prompt only'),('error_response_L14','Neutral error probe'),('entropy_response_L14','Answer-entropy probe (approx.)'),('deception_response_L14_C0001','Deception, $C=0.001$'),('deception_text','Answer-text classifier'),('nll','Mean negative log probability'),('entropy','Independent answer entropy')]
lookup={(r['method'],r['contrast']):r for r in M}
rows=[]
for key,name in methods:
    cells=[]
    for c in cs:
        r=lookup[key,c]
        score=f"{float(r['auc']):.3f}"
        if float(r['auc'])==max(float(lookup[k,c]['auc']) for k,_ in methods): score=r'\textbf{'+score+'}'
        cells.append(score+f" [{float(r['auc_low']):.3f}, {float(r['auc_high']):.3f}]")
    rows.append([name]+cells)
table('detectors','llll',r'Detector / control & Incentive candidates & Instructed control & Neutral factuality \\'+ '\n' +r'Positive / negative counts & $4/19$ & $13/19$ & $38/59$ \\',rows,
      r'Held-out signed \auroc [95\% question-bootstrap interval]. The first two columns contrast false known-question answers with uncertain neutral errors; the third contrasts neutral errors with correct answers. Incentive candidates do not require recovery and therefore have unverified mechanism labels. Bold marks the numerical maximum in each column, not a significant ranking or demonstrated mechanism specificity. Chance \auroc is 0.5.', 'tab:detectors',True)
rows=[]
for arm in ['neutral','sham','reward','reputation','instructed']:
    r=next(x for x in F if x['method']=='deception_response_L14' and x['arm']==arm and x['state']=='known' and x['factual']=='correct')
    rows.append([arm.capitalize(),f"{r['flagged']}/{r['n']}",f"{100*float(r['rate']):.1f}",f"{100*float(r['ci_low']):.1f}--{100*float(r['ci_high']):.1f}"])
table('flags','lrrr',r'Known-correct condition & Flagged / total & Rate (\%) & Exact 95\% interval (\%) \\',rows,
      r'False alarms of the layer-14 response deception probe on known-correct test answers. The threshold uses 45 validation neutral-correct answers with a nominal 5\% target. These condition-specific rates diagnose calibration shift; they are not estimates of overall monitor precision.', 'tab:flags')
rows=[]
for r in S['paired_tests']:
    rows.append([r['arm'].capitalize()+' $-$ sham',f"{100*r['risk_difference']:.2f}",f"[{100*r['ci'][0]:.2f}, {100*r['ci'][1]:.2f}]",f"{r['increase_pairs']}/{r['decrease_pairs']}",f"{r['p_holm']:.4f}"])
table('paired','lrrrr',r'Comparison & Difference (pp) & 95\% CI (pp) & Increases/decreases & Holm $p$ \\',rows,r'Paired false-response comparisons on 229 strictly known questions. Increases and decreases are discordant pairs relative to sham. Intervals use question bootstrap; exact paired tests are Holm-corrected over the two comparisons.', 'tab:paired',True)
rows=[]
for arm in ['neutral','sham','reward','reputation','instructed']:
    r=A['arms'][arm]; rows.append([arm.capitalize(),r['correct'],r['false'],r['abstain'],600-r['correct']-r['false']-r['abstain'],r['candidate_misreport']])
table('audit','lrrrrr',r'Arm & Correct & False & Abstain & Invalid/unresolved & Candidates \\',rows,r'Reference-audited main responses. Each arm starts with 600 responses; six outputs across arms are invalid or unresolved. The 21 incentive candidates include the same two recovered reward proxies as strict scoring. Audit labels are a sensitivity analysis, not human ground truth.', 'tab:audit')
rows=[[f"{r['threshold']}/5 + verification",r['known'],r['candidate_misreports'],r['verified_misreports']] for r in S['sensitivity']]
table('knowledge','lrrr',r'Knowledge criterion & Known & Candidates & Recovered \\',rows,r'Strict-label sensitivity to the neutral-consistency threshold, always requiring correct independent verification. All thresholds are reported without selecting a preferred result from test performance.', 'tab:knowledge')
rows=[]
for key,name in [('deception_response_L14','Response, layer 14, $C=1$'),('deception_response_L28','Response, layer 28, $C=1$'),('deception_response_L14_C0001','Response, layer 14, $C=0.001$'),('deception_prompt_L14','Prompt, layer 14, $C=1$'),('deception_prompt_L28','Prompt, layer 28, $C=1$'),('length','Answer length')]:
    cells=[]
    for c in cs:
        r=lookup[key,c];cells.append(f"{float(r['auc']):.3f} [{float(r['auc_low']):.3f}, {float(r['auc_high']):.3f}]")
    rows.append([name]+cells)
table('sensitivity','llll',r'Fixed sensitivity & Incentive candidates & Instructed control & Neutral factuality \\',rows,r'Fixed layer and regularization sensitivities, plus answer length, on the contrasts in \tabref{tab:detectors}. Entries are \auroc [95\% question-bootstrap interval]. These comparisons were not used to select a test-set winner. Small positive classes limit interpretation.', 'tab:sensitivity',True)
rows=[]
for key,name in methods:
    rows.append([name]+[f"{float(lookup[key,c]['ap']):.3f}" for c in cs])
table('average_precision','lrrr',r'Detector / control & Incentive candidates & Instructed control & Neutral factuality \\',rows,
      r'Average precision for the same frozen contrasts as \tabref{tab:detectors}. Positive-prevalence reference levels are $4/23=0.174$, $13/32=0.406$, and $38/97=0.392$, respectively. Values are descriptive, without post-hoc significance claims.', 'tab:ap',True)

# Bibliographic metadata is already archived from source pages.
keys={'2503.03750':'ren2025mask','2511.16035':'kretschmar2025liars','2502.03407':'goldowskydill2025detecting','2602.01425':'natarajan2026one','2603.10003':'thormann2026probing','2609.00180':'luikham2026asymmetries','2607.20479':'moustafa2026beyond','2510.09033':'cheang2025know','2606.12618':'cooney2026did','2406.15927':'kossen2024semantic'}
bib=[]
for r in json.loads((ROOT/'notes/paper_metadata.json').read_text()):
    arxiv=r['id']; title=html.unescape(r['title'][0]).replace(chr(34)+'Did you lie?'+chr(34), "``Did you lie?''"); authors=' and '.join(r['authors'])
    year='20'+arxiv[:2]
    bib.append('@misc{'+keys[arxiv]+',\n  title={{'+title+'}},\n  author={'+authors+'},\n  year={'+year+'},\n  howpublished={arXiv preprint arXiv:'+arxiv+'},\n  url={https://arxiv.org/abs/'+arxiv+'},\n  doi={10.48550/arXiv.'+arxiv+'}\n}')
bib.append(r'''@misc{hopkins2026fine,
  author={Hopkins, Jack and Khullar, Dipika and Wang, Rowan and Roger, Fabien},
  title={{Fine-Tuned Lie Detectors Failed to Generalize}},
  year={2026},
  howpublished={Anthropic Alignment Science Blog},
  url={https://alignment.anthropic.com/2026/lie-detectors/},
  note={August 21, 2026}
}
@misc{triviaqaData,
  author={{TriviaQA}},
  title={{TriviaQA: rc.nocontext dataset}},
  howpublished={Hugging Face dataset repository},
  year={2026},
  url={https://huggingface.co/datasets/mandarjoshi/trivia_qa},
  note={Snapshot accessed October 4, 2026; pinned revision in Appendix A}
}
@misc{qwenModel,
  author={{Qwen Team}},
  title={{Qwen2.5-7B-Instruct}},
  year={2024},
  howpublished={Hugging Face model repository},
  url={https://huggingface.co/Qwen/Qwen2.5-7B-Instruct},
  note={Pinned checkpoint revision in Appendix A}
}''')
(P/'references.bib').write_text('\n\n'.join(bib)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.8,3.35));ax.set(xlim=(0,10),ylim=(0,3.35));ax.axis('off')
def box(x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.09',fc=color,ec='#728096',lw=.8))
    ax.text(x+w/2,y+h-.16,title,ha='center',va='top',fontsize=11,fontweight='bold')
    ax.text(x+w/2,y+h-.46,body,ha='center',va='top',fontsize=10,linespacing=1.4)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#526071',lw=1.1))
box(.12,1.02,2.05,1.65,'600 questions','Entity split\n355 / 124 / 121\ntrain / val. / test','#e8eef5')
box(2.62,1.86,3.02,1.12,'Neutral evidence','5 beliefs + verification\n3 entropy samples','#e8f1e9')
box(2.62,.24,3.02,1.18,'Five conditions','Neutral / Sham / Reward\nReputation / Instructed\nOne follow-up per answer','#f7efdf')
box(6.12,1.86,3.63,1.12,'Behavioral labels','Recovered / Uncertain\nStable wrong / Unresolved','#e8f1e9')
box(6.12,.24,3.63,1.18,'Frozen probes','Answer / prompt states\nFalse vs. false + correct\n1 recovered test positive','#f1e8ef')
arrow((2.26,2.08),(2.53,2.4));arrow((2.26,1.45),(2.53,.85));arrow((5.74,2.42),(6.01,2.42));arrow((5.74,.83),(6.01,.83));arrow((4.13,1.52),(6.01,2.04))
fig.savefig(P/'figures/protocol.pdf',bbox_inches='tight',pad_inches=.04);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(7.2,3.5),gridspec_kw={'width_ratios':[1.12,1]})
selected=methods[:4]+[methods[5],methods[6],methods[7]]
labels=['Deception: response','Deception: prompt only','Neutral error probe','Entropy probe (approx.)','Answer text','Mean neg. log prob.','Sampled entropy']
for j,(c,title) in enumerate(zip(cs[:2],['Incentive candidates\n4 positive / 19 negative','Instructed control\n13 positive / 19 negative'])):
    ax=axs[j]
    for i,(key,_) in enumerate(selected):
        r=lookup[key,c];auc=float(r['auc']);lo=float(r['auc_low']);hi=float(r['auc_high'])
        color='#bd6035' if 'prompt' in key else '#305e89'
        ax.errorbar(auc,i,xerr=[[auc-lo],[hi-auc]],fmt='o',color=color,capsize=3,ms=5,lw=1.3)
    ax.set(xlim=(-.03,1.035),ylim=(6.6,-.6),yticks=range(7),yticklabels=labels if j==0 else ['']*7,xlabel='AUROC',title=title)
    ax.axvline(.5,color='#999999',ls='--',lw=1);ax.grid(axis='x',alpha=.16)
fig.tight_layout(w_pad=1.8);fig.savefig(P/'figures/detector_controls.pdf',bbox_inches='tight');plt.close(fig)
# Preserve the original report's empirical figures without recomputing observations.
for name in ['response_categories','entropy_by_state']:
    shutil.copy2(ROOT/'figures'/f'{name}.png',P/'figures'/f'{name}.png')
print('Generated tables, bibliography, and figures from frozen artifacts.')
