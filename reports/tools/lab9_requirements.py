from pathlib import Path
from src.utils.io import write_json,file_hash

ROWS=[
('H01','§1/§13: reuse Lab0 EXP_ID/RUN_ID harness','Shared ExperimentRunner; additive Lab9 macro-F1/early-stop only','regression tests and manifest trace','PASS'),
('D01','§6: genuine RAVDESS1440 clips/24actors/8classes','Official24speech archives, officialMD5 and clip IDs','complete acquisition manifest','BLOCKED'),
('D02','§6: actor-disjoint14/3/2/5 inherited split','Deterministic backup1729; teacher split not found','actor lists/hash/counts','UNVERIFIED'),
('D03','§6: cal not read; val selection; test final','No cal media extraction; explicit role guards','filesystem and sample IDs','BLOCKED'),
('D04','§6: train-only normalization','Audio mel and cached clip features train only; visual upstream train normalization retained','recalculate train statistics','BLOCKED'),
('P01','§7: four SHA256 lab_pick fields','Private existing ID; preserve contradictory source/architecture fields','independent hash slice recomputation','PASS'),
('P02','§7: effective visual encoder identity','User-authorized visual-source precedence, CNN versus generic vit_small conflict retained','strict original checkpoint/hash/config','UNVERIFIED'),
('T11','§8.1.1:5FPS frozen preclassifier visual features','224face inputs; skip face_found=false; cache byclip','weights/hash/eval/padding checks','BLOCKED'),
('T12','§8.1.2/(9.3):logmel64,400,160,centerTrue','Explicit HTK power filters and periodic Hann; 56000->64x351','actual shape test','PASS'),
('T13','§8.1.2:AudioCNN train/val selection then freeze','Shared runner; val macro-F1 best; fixed budget/early stopping','checkpoint curves, no test access','BLOCKED'),
('T14','§8.1.3/(9.10): unified explicit mask','model(visual,audio,mask,align); zero token inside model','invariance/illegal mask tests','PASS'),
('T15','§8.1.4/(9.4): integer half-open synchronization','Integer samples; offsets/shifts/r; explicit valid coverage','boundary/no-gap/no-overlap tests','PASS'),
('T16','§8.1.5:passports include time and encoder hashes','Resolved config + feature metadata + cache identity','raw cache/checkpoint provenance','BLOCKED'),
('T21','§8.2.1:video/audio/early/gated/late registry','One hidden nonlinear head; scalar profile gate','normalization/limit tests','PASS'),
('T22','§8.2.2:separate fusion/head/frozen parameter counts','Explicit parameter breakdown and external frozen source','independent parameter sum','BLOCKED'),
('T23','§8.2.3:matched controls<=1% or nearest width','Exact integer width optimizer, record step and nearest error','enumerate adjacent widths','PASS'),
('T24','§8.2.4:late0.5 and val-selected grid0:.1:1','Use cached unimodal probabilities; val macro-F1 only','lambda provenance and limits','BLOCKED'),
('T25','§8.2.5:per-clip gates and conditional summaries','Mean, sample SD, emotion, correct/error','per-clip recomputation','BLOCKED'),
('T31','§8.3.1/(9.12):input noise, time mask, blur/dropframes','Waveform measured power; reuse Lab8 blur parameters; deterministic','SNR<=.1dB,level0 identity','PASS'),
('T32','§8.3.2:reencode corrupt raw signal/frames','Same frozen encoders; separate corrupt cache identities','input/cache hashes','BLOCKED'),
('T33','§8.3.3/(9.9):same-model contribution','Save full/withoutaudio/withoutvideo/shuffle probability vectors','independent macro-F1 and L1','BLOCKED'),
('T34','§8.3.4:distinguish separately-trained unimodal probs','Explicit source RUN_ID per predicted array','source registry audit','BLOCKED'),
('T35','§8.3.5:ProtocolB undefined metrics reject','ValueError accuracy/macroF1/ECE/Risk, no fabricated label','mandatory test6','PASS'),
('E11','§9.1:single-seed exploration all strategies','Controlled budgets, sources, classes, bestval checkpoints','val/test metrics and confusion','BLOCKED'),
('E12','§9.1:best strategy/controls chosen byval','Lock validation ranking before main test comparison','selection evidence','BLOCKED'),
('E13','§9.1/(9.8):G,Gint,Gctrl','Per-strategy test gain with matching-control scope','recompute from raw predictions','BLOCKED'),
('E14','§9.1:hypothesis.md realGit before main','F/U/fixedconditions/direction/falsifier/seeds1,2,3','Git blob and timestamp checks','BLOCKED'),
('E15','§9.1/(9.14):three actual paired seeds','Train F/U after prereg; shared frozen features','three differences, mean, ddof1SD','BLOCKED'),
('E16','§9.1:all same sign and absmean>=2SD','Preserve null/reverse result and prediction FAIL','independent teacher-rule calculation','BLOCKED'),
('E17','§9.1:parameter count and confusion/perclass figures','Automatically plot saved outputs','byte regeneration and visual QA','BLOCKED'),
('E21','§9.2:modality dropout profile probability','pdrop=.5; full=.5/single=.25each; neverbothmissing','sampling test and training manifest','BLOCKED'),
('E22','§9.2:all fusion Full/NoAudio/NoVideo','Same model zero null token; actual majority M0','R,Rtilde,delta and undefined reasons','BLOCKED'),
('E23','§9.2:audio mask versus no-fixed-point shuffle','Fixed derangement; video/labels unchanged','L1 and changed prediction counts','BLOCKED'),
('E24','§9.2:compare Rv-Ra versus MV-MA','Actual independent models versus same-model interventions','saved probability and source separation','BLOCKED'),
('E25','§9.2 boundary:SNR30,20,10,0,-5 both gated models','Same300testclips, waveform corruption then reencoding','perclip SNR/F1/gate and no-audio reference','BLOCKED'),
('E26','§9.2:highest gridSNR worse than noaudio or absence','Do not presuppose gate shift','grid scan and explanation','BLOCKED'),
('E31','§9.3:>=200 sameemotion and>=200 differentemotion pairs','300unique ordered pairs each, differentactors, fixedseed','pair identity/label tests','BLOCKED'),
('E32','§9.3:early/gated/late0.5 evaluate A/B','A F1/accuracy/conf; B rv,ra,r0,deltaij/conf and true gates only','raw pair prediction files','BLOCKED'),
('E33','§9.3:rv+ra+r0=1; B undefined','Raise on invalid metric requests; gates N/A when absent','independent counts','BLOCKED'),
('E34','§9.3:detect inconsistency AUROC1-conf anddeltaa','Anegative/Bpositive; deltaa differs fromdeltaij','independent AUROC and baseline.5','BLOCKED'),
('E35','§9.3:confidence/gate figures and actor confound','Original/A/B histograms; identity versus emotion limits','real histograms and discussion','BLOCKED'),
('Q1','§10.1:weak modality use not forced','Own Rtilde and mean delta evidence','RUN_ID-backed measured answer','BLOCKED'),
('Q2','§10.2:double control hidden width and retrain','New genuine training run and revisedGctrl','checkpoint/curve/probabilities','BLOCKED'),
('Q3','§10.3:audio shuffle versusmask','Use measured output changes andRa','RUN_ID-backed measured answer','BLOCKED'),
('Q4','§10.4:Lab8 level1blur real reencoding','MeasureM_AV,gate,same-modelvideo intervention','RUN_ID and corrupt input cache','BLOCKED'),
('Q5','§10.5:gate versusSNR not guaranteed','Use actual E2 curves; no invented trend','RUN_ID-backed measured answer','BLOCKED'),
('A01','§11:all six mandatory test groups','Unit tests + genuine data/experiment audit separated','JUnit and independent audit','PASS'),
('A02','§12:all results traceable toRUN_ID and automatic plots','Raw probability arrays, tables, deterministic generation','independent regeneration','BLOCKED'),
('A03','§13:all mandatory experiments and prereg complete','Final checklist distinguishes experiment/prediction/protocol','audit and clean Git commit','BLOCKED'),
('N01','§12:formal report','Explicitly deferred by user','No LaTeX/PDF changes','NOT_APPLICABLE'),
('N02','§14:crossattention/calibration/time-shift study/regression','Additional tasks not requested; synchronization shift still unit-tested','scope statement','NOT_APPLICABLE')]

def main():
    rows=[dict(zip(['ID','PDF_requirement','implementation','verification','status'],r)) for r in ROWS]
    write_json('experiments/lab9/requirement_status.json',{'rows':rows,'PDF_sha256':file_hash('drive-download-20261026T053112Z-1-001/lab9_multimodal_representation_learning_1010.pdf'),'state_note':'BLOCKED rows are not yet executed, awaiting complete authentic acquisition/implementation; no synthetic scientific results.'})
    lines=['# Lab9 需求检查表','', '教师原PDF14页已完整重读；公式9.1–9.14与附加9.15、六组测试、五题逐项核对。报告按用户明确指令暂不制作。','', '| ID | PDF原始要求 | 实施方法 | 验证方法 | 状态 |','|---|---|---|---|---|']
    for r in ROWS:lines.append('| '+' | '.join(r)+' |')
    Path('docs/requirements/requirements_lab9.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
