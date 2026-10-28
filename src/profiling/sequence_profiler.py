import gc
import time
import tracemalloc
import torch
from src.metrics.objective import objective

def measure(model,length,batch=4,warmup=10,repeats=30):
    if warmup<10:raise ValueError('At least10 warmup calls required')
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    x=torch.randint(0,64,(batch,length),device=device);y=torch.randint(0,64,(batch,length),device=device)
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.,eps=1e-4)
    def forward():
        with torch.no_grad():return model(x)
    def train():
        optimizer.zero_grad(set_to_none=True);loss=objective(model(x).float(),y);loss.backward();optimizer.step()
    timings={};peaks={};raw={}
    for name,call in [('forward',forward),('training_step',train)]:
        model.train(name=='training_step')
        for _ in range(warmup):call()
        optimizer.zero_grad(set_to_none=True);sync()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        else:tracemalloc.start();tracemalloc.reset_peak()
        samples=[]
        for _ in range(repeats):
            sync();start=time.perf_counter();call();sync();samples.append(time.perf_counter()-start)
        peaks[name]=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else tracemalloc.get_traced_memory()[1]
        if device.type!='cuda':tracemalloc.stop()
        timings[name]=sum(samples)/len(samples);raw[name]=samples
    width=model.config['width'];heads=model.config['heads'];element=torch.empty((),dtype=dtype).element_size()
    return {'status':'COMPLETED_PROFILE','length':length,'batch':batch,'width':width,'depth':model.config['depth'],
        'backend':model.config['backend'],'dtype':str(dtype),'device':str(device),'warmup':warmup,'repeats':repeats,
        'forward_seconds':timings['forward'],'training_step_seconds':timings['training_step'],
        'peak_memory_bytes':max(peaks.values()),'peak_by_operation':peaks,'raw_seconds':raw,
        'tokens_per_second':batch*length/timings['forward'],
        'profiling_optimizer':'AdamW lr0 eps1e-4; gradients/state/step measured, initialized weights unchanged',
        'memory_measurement':'cuda.max_memory_allocated' if device.type=='cuda' else 'tracemalloc Python allocations; native tensors excluded',
        'theoretical_attention_matrix_bytes':batch*heads*length*length*element,
        'largest_explicit_intermediate_tensor_bytes':max(batch*heads*length*length*element if model.config['backend']=='explicit' else 0,batch*length*3*width*element),
        'largest_tensor_scope':'analytic shape audit of activations, excludes allocator peak and optimizer state'}

def safe_measure(model,*args,**kwargs):
    try:return measure(model,*args,**kwargs)
    except torch.OutOfMemoryError as exc:
        result={'status':'OOM','length':args[0] if args else kwargs['length'],'backend':model.config['backend'],'exception':str(exc)}
        model.zero_grad(set_to_none=True);gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        return result
