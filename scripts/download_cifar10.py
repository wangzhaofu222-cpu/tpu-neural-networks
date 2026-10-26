"""Download only the dataset explicitly allowed by the course, verify MD5 and record SHA256."""
import argparse
import hashlib
import json
import sys
import urllib.request
import tarfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.io import file_hash,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('data'))
    args=p.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    archive=args.root/'cifar-10-python.tar.gz'
    url='https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
    if not archive.exists():
        temporary=archive.with_suffix('.partial')
        total=170498071
        prefix=temporary.stat().st_size if temporary.exists() else 0
        if prefix>total:
            raise RuntimeError('Partial file exceeds official archive length')
        parts=args.root/'cifar-parts'
        parts.mkdir(exist_ok=True)
        ranges=[]
        for i in range(4):
            start=prefix+(total-prefix)*i//4
            end=prefix+(total-prefix)*(i+1)//4-1
            if start<=end:
                ranges.append((start,end))
        def fetch(pair):
            start,end=pair
            out=parts/f'{start}-{end}.bin'
            if out.exists() and out.stat().st_size==end-start+1:
                return out
            request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise RuntimeError('Server did not honor the exact byte range')
                with out.with_suffix('.partial').open('wb') as target:
                    shutil.copyfileobj(response,target,length=1024*1024)
            if out.with_suffix('.partial').stat().st_size!=end-start+1:
                raise RuntimeError('Incomplete range')
            out.with_suffix('.partial').replace(out)
            print(f'Verified range length: {start}-{end}',flush=True)
            return out
        print(f'Resuming {prefix}/{total} bytes with four official-source ranges',flush=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            files=list(executor.map(fetch,ranges))
        assembled=archive.with_suffix('.assembled')
        with assembled.open('wb') as output:
            if prefix:
                with temporary.open('rb') as stream:
                    shutil.copyfileobj(stream,output)
            for part in files:
                with part.open('rb') as stream:
                    shutil.copyfileobj(stream,output)
        if hashlib.md5(assembled.read_bytes()).hexdigest()!='c58f30108f718f92721af3b95e74349a':
            raise RuntimeError('Assembled archive MD5 mismatch; partial files retained for diagnosis')
        assembled.replace(archive)
    md5=hashlib.md5(archive.read_bytes()).hexdigest()
    if md5!='c58f30108f718f92721af3b95e74349a':
        raise RuntimeError('Official archive MD5 mismatch')
    target=args.root/'cifar-10-batches-py'
    if not target.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(args.root,filter='data')
    write_json(ROOT/'docs/environment/cifar10_download.json',{
        'source':url,
        'archive_md5':md5,'archive_sha256':file_hash(archive),'bytes':archive.stat().st_size,
        'licence':'Official download page does not specify a standalone licence; see docs/data_sources.md',
        'course_protocol_file_present':False})
    print('Official archive verified; no training started.')


if __name__=='__main__':
    main()
