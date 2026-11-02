"""Compare two official URLs using small ranges, no archive mutation."""
import urllib.request,time,concurrent.futures,json
from src.utils.io import write_json
def probe(url):
    start=time.perf_counter()
    try:
        request=urllib.request.Request(url,headers={'Range':'bytes=490000000-491048575','User-Agent':'TPU-course-Lab9/1.0'})
        with urllib.request.urlopen(request,timeout=45) as r:
            data=r.read(1048576);return {'url':url,'status':r.status,'range':r.headers.get('Content-Range'),'bytes':len(data),'seconds':time.perf_counter()-start}
    except Exception as exc:return {'url':url,'error':type(exc).__name__,'seconds':time.perf_counter()-start}
def main():
    urls=['https://zenodo.org/api/records/1188976/files/Video_Speech_Actor_04.zip/content','https://zenodo.org/records/1188976/files/Video_Speech_Actor_04.zip?download=1']
    with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=list(pool.map(probe,urls))
    write_json('experiments/lab9/transport_probe.json',rows);print(json.dumps(rows))
if __name__=='__main__':main()
