"""Authorized official RAVDESS speech acquisition, resumable and hash verified."""
import concurrent.futures,hashlib,json,time,urllib.request,zipfile
from pathlib import Path
from src.utils.io import write_json,file_hash

R=Path('experiments/lab9');ROOT=Path('data/lab9');RAW=ROOT/'official_archives';MEDIA=ROOT/'media'

def acquire(row):
    RAW.mkdir(parents=True,exist_ok=True);MEDIA.mkdir(parents=True,exist_ok=True)
    name=row['key'];dest=RAW/name;part=RAW/(name+'.partial');expected=row['checksum'].split(':',1)[1];events=[]
    def md5(p):
        h=hashlib.md5()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    if part.exists() and part.stat().st_size==row['size'] and md5(part)==expected:part.replace(dest)
    if not(dest.exists() and dest.stat().st_size==row['size'] and md5(dest)==expected):
        stalled=0;requests=0
        while not part.exists() or part.stat().st_size<row['size']:
            requests+=1
            if requests>4096:raise RuntimeError('Bounded acquisition request budget exceeded')
            begin=part.stat().st_size if part.exists() else 0;end=min(row['size']-1,begin+4*1024*1024-1)
            try:
                req=urllib.request.Request(row['links']['self'],headers={'User-Agent':'TPU-course-Lab9/1.0','Range':f'bytes={begin}-{end}'})
                with urllib.request.urlopen(req,timeout=60) as response:
                    if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {begin}-'):
                        raise RuntimeError('Server did not honor exact range; refuse unsafe append')
                    request_started=time.monotonic()
                    with part.open('ab') as f:
                        while True:
                            chunk=response.read1(64*1024)
                            if not chunk:break
                            f.write(chunk);f.flush()
                            if time.monotonic()-request_started>120:raise TimeoutError('Range wall-time cap; retain exact received prefix')
                if part.stat().st_size!=end+1:raise IOError('Transport closed before requested range completed')
                stalled=0
                write_json(R/'acquisition'/('download_state_'+name+'.json'),{'received_bytes':part.stat().st_size,'expected_bytes':row['size'],'last_range':[begin,end],'requests_this_process':requests})
            except Exception as exc:
                current=part.stat().st_size if part.exists() else 0;stalled=stalled+1 if current<=begin else 0
                events.append({'request':requests,'error':repr(exc),'partial_bytes':current,'range':[begin,end]})
                write_json(R/'acquisition'/('errors_'+name+'.json'),events)
                if stalled>=3:raise
                time.sleep(5)
        if part.stat().st_size!=row['size'] or md5(part)!=expected:raise ValueError('Complete archive failed official MD5; retain evidence without claiming success')
        part.replace(dest)
    actor=int(name.rsplit('_',1)[1].split('.')[0]);split=json.loads((R/'planned_actor_split.json').read_text());role=next(k for k,v in split['roles'].items() if actor in v)
    records=[]
    with zipfile.ZipFile(dest) as archive:
        entries=[i for i in archive.infolist() if Path(i.filename).name.startswith('01-01-') and i.filename.lower().endswith('.mp4')]
        assert len(entries)==60
        for info in entries:
            filename=Path(info.filename).name;parts=filename[:-4].split('-');assert len(parts)==7 and int(parts[-1])==actor
            target=MEDIA/f'Actor_{actor:02d}'/filename
            if role!='cal' and not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(info) as src,target.open('wb') as output:
                    for chunk in iter(lambda:src.read(1024*1024),b''):output.write(chunk)
            records.append({'clip_id':'-'.join(parts[1:]),'actor_id':actor,'emotion':int(parts[2])-1,'role':role,'official_archive':name,'archive_entry':info.filename,
                            'media_path':str(target) if role!='cal' else None,'cal_media_extracted':False,'video_audio_same_file':True,'uncompressed_bytes':info.file_size,'zip_crc':info.CRC})
    result={'actor_id':actor,'role':role,'archive':str(dest),'bytes':dest.stat().st_size,'official_md5':expected,'verified_md5':md5(dest),'sha256':file_hash(dest),'clips':records,'errors':events}
    write_json(R/'acquisition'/f'actor_{actor:02d}.json',result);print('VERIFIED',name,'role',role,'clips',len(records),flush=True);return result

def main():
    metadata=json.loads((R/'official_source_metadata.json').read_text());rows=sorted([x for x in metadata['files'] if x['key'].startswith('Video_Speech')],key=lambda x:x['key']);assert len(rows)==24
    start=time.time();results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures={pool.submit(acquire,row):row['key'] for row in rows}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result());write_json(R/'acquisition_progress.json',{'verified_actors':sorted(x['actor_id'] for x in results),'verified_bytes':sum(x['bytes'] for x in results),'elapsed_seconds':time.time()-start})
    write_json(R/'acquisition_complete.json',{'status':'PASS','actors':24,'clips':sum(len(r['clips']) for r in results),'bytes':sum(r['bytes'] for r in results),'wall_seconds':time.time()-start,'source':'https://zenodo.org/records/1188976','cal_decoded':False})

if __name__=='__main__':main()
