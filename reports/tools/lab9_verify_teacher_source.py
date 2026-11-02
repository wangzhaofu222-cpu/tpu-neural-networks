import json,urllib.request,hashlib
from pathlib import Path
from src.utils.io import write_json,file_hash

def main():
    url=json.loads(Path('tmp/lab9_teacher_download_private.json').read_text())['url']
    with urllib.request.urlopen(url,timeout=60) as response:raw=response.read()
    local=Path('drive-download-20261026T053112Z-1-001/lab9_multimodal_representation_learning_1010.pdf')
    assert raw==local.read_bytes(),'Teacher file differs: reread remote source before experimentation'
    Path('tmp/pdfs/lab9/teacher_download.pdf').write_bytes(raw)
    write_json('experiments/lab9/teacher_source_identity.json',{'status':'PASS','source':'https://drive.google.com/file/d/1EQAZATnNTYls29_T8uZ4FBVLzQQa_4ql/view','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'local_sha256':file_hash(local),'comparison':'byte-for-byte identical','access':'authenticated connected Google Drive raw download; temporary URL excluded'})
    print('Teacher PDF byte identity PASS',len(raw))

if __name__=='__main__':main()
