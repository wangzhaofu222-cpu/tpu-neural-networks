import torch
from src.models.small_cnn import SmallCNN
from src.profile import profile_model


def test_profiler_actual_measurements_and_no_mutation():
    torch.set_num_threads(1)
    model=SmallCNN(4,1)
    state={k:v.clone() for k,v in model.state_dict().items()}
    result=profile_model(model,[1,2],torch.device('cpu'))
    assert model.training
    assert all(torch.equal(state[k],v) for k,v in model.state_dict().items())
    for row in result['measurements']:
        assert len(row['samples_seconds'])==30
        assert row['mean_seconds']>0
        assert row['throughput_samples_per_second']==row['batch_size']/row['mean_seconds']
        assert row['peak_gpu_memory_bytes'] is None
