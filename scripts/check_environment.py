"""Read-only environment and small CPU/CUDA arithmetic verification."""
import json
import platform
import shutil
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import psutil
import torch
from src.utils.io import write_json
from src.utils.provenance import environment


def main():
    info=environment(torch.device('cpu'))
    info.update(python_requirement_met=sys.version_info>=(3,11),torch_requirement_met=torch.__version__.startswith('2.'),
                ram=psutil.virtual_memory()._asdict(),disk=shutil.disk_usage(root)._asdict())
    info['cuda_available']=torch.cuda.is_available()
    info['cuda_arithmetic_check']=None
    if torch.cuda.is_available():
        x=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
        info['cuda_arithmetic_check']=bool(torch.equal((x@torch.eye(4,device='cuda')).cpu(),x.cpu()))
        info['gpu']=torch.cuda.get_device_properties(0).name
        info['gpu_total_memory']=torch.cuda.get_device_properties(0).total_memory
    out=root/'docs/environment/current.json'
    write_json(out,info)
    print(json.dumps({k:v for k,v in info.items() if k!='packages'},indent=2))


if __name__=='__main__':
    main()
