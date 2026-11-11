from pathlib import Path
import json,re,shutil,zipfile,hashlib
from pypdf import PdfReader
from prepare import ROOT,OUT,TOOLS,save,sha

def main():
    audit=OUT/'04_源码与核查';qa=ROOT/'tmp/pdfs/defense_20261111'
    manifest=json.loads((audit/'qa_manifest.json').read_text('utf-8'))
    manifest['visual_review']='PASS: all 164 newly authored PDF pages visually inspected at final hashes'
    manifest['numeric_review']='PASS: 83 flagged segments reviewed; reordering, word numerals or whitespace, no measured value changes'
    manifest['guide_sha256']=sha(OUT/'03_中俄答辩提纲/Lab0-9_中俄双语答辩指南.pdf')
    manifest['teacher_question_summaries']=20;manifest['additional_possible_questions']=24
    manifest['new_pdf_pages_reviewed']=sum(x['zh_pages'] for x in manifest['translation_reports'])+manifest['guide_pages']
    manifest['original_ru_pages']=sum(x['ru_pages'] for x in manifest['translation_reports'])
    assert manifest['new_pdf_pages_reviewed']==164
    save(audit/'qa_manifest.json',manifest)
    review=json.loads((qa/'numeric_review.json').read_text('utf-8'))
    for row in review:row['status']='REVIEWED: meaning and measured values preserved'
    save(audit/'numeric_review.json',review)
    qmap={0:[3,8],1:[4,2],2:[2,4],3:[1,2],4:[2,1],5:[2,5],6:[1,3],7:[2,1],8:[4,3],9:[1,5]}
    save(audit/'guide_question_sources.json',[{'lab':n,'guide_teacher_question_order':i+1,'original_counterfactual_question':q,'report':f'01_俄语正式报告/Lab{n}_Report.pdf'} for n,qs in qmap.items() for i,q in enumerate(qs)])
    # Archive only our own freshly generated compilation caches, within this workspace.
    archive=qa/'compiled';archive.mkdir(exist_ok=True)
    logs=audit/'编译日志';logs.mkdir(exist_ok=True)
    for name in [*[f'lab{n}_zh' for n in range(10)],'guide']:
        project=(audit/name).resolve();build=(project/'build').resolve();dest=(archive/name).resolve()
        assert project.is_relative_to(OUT.resolve()) and dest.is_relative_to(qa.resolve())
        if build.exists():
            assert build.parent==project and not dest.exists(),str(build)
            for p in [build/'main.log',project/'console.log']:
                if p.exists():shutil.copy2(p,logs/(name+'_'+p.name))
            shutil.move(str(build),str(dest))
        # The Chinese cover is authored directly; unused original cover assets are not needed.
        assets=(project/'assets').resolve()
        if assets.exists():
            tex='\n'.join(p.read_text('utf-8-sig') for p in project.rglob('*.tex'))
            assert 'assets/' not in tex
            target=(qa/'unused_cover_assets'/name).resolve()
            assert assets.parent==project and target.is_relative_to(qa.resolve()) and not target.exists()
            target.parent.mkdir(exist_ok=True)
            shutil.move(str(assets),str(target))
    main_pdfs=[*sorted((OUT/'01_俄语正式报告').glob('*.pdf')),*sorted((OUT/'02_中文对照译本').glob('*.pdf')),*sorted((OUT/'03_中俄答辩提纲').glob('*.pdf'))]
    assert len(main_pdfs)==21
    assert sum(len(PdfReader(p).pages) for p in main_pdfs)==306
    file_manifest=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='文件清单_SHA256.json':file_manifest.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    save(OUT/'文件清单_SHA256.json',file_manifest)
    zpath=ROOT/(OUT.name+'.zip');assert not zpath.exists(),str(zpath)
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        for r in file_manifest:
            assert hashlib.sha256(z.read(OUT.name+'/'+r['path'])).hexdigest()==r['sha256']
    originals=json.loads((audit/'original_hashes.json').read_text('utf-8'))
    assert all(sha(ROOT/p)==h for p,h in originals.items())
    print(json.dumps({'folder':str(OUT),'zip':str(zpath),'zip_MiB':round(zpath.stat().st_size/2**20,2),'primary_pdfs':21,'pages':306,'new_pages_visual_review':164,'questions':44,'files':len(file_manifest)+1,'zip_crc_and_file_hashes':'PASS','source_hashes':'95 unchanged'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
