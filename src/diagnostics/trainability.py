import copy
import math
import statistics as st
import torch
from torch.nn import functional as F
from src.diagnostics.activations import ActivationMonitor, statistics
from src.diagnostics.gradients import gradient_statistics


def imbalance(values):
    values = [v for v in values if v is not None]
    return math.log10((max(values)+1e-12)/(min(values)+1e-12)) if values else None


def summarize(activations, gradient_samples, loss=None):
    failure = any(r.get('numerical_failure', False) or r.get('delta', {}).get('numerical_failure', False)
                  for r in activations.values())
    by_layer = {}
    for sample in gradient_samples:
        for name, row in sample.items():
            failure |= row.get('numerical_failure', False)
            if row.get('g') is not None:
                by_layer.setdefault(name, []).append(row['g'])
    medians = {k: st.median(v) for k,v in by_layer.items()}
    variances = [r['variance'] for r in activations.values() if 'variance' in r]
    if loss is not None and not math.isfinite(loss):
        failure = True
    return {'rho_T': loss/math.log(10) if loss is not None and math.isfinite(loss) else None,
            'GI': imbalance(medians.values()), 'median_g_by_layer': medians,
            'AVD': max(abs(math.log10(v+1e-12)-math.log10(variances[0]+1e-12)) for v in variances) if variances else None,
            'max_zero_fraction': max((r.get('zero_fraction') or 0 for r in activations.values()), default=0),
            'max_saturated_fraction': max((r.get('saturated_fraction') or 0 for r in activations.values()), default=0),
            'numerical_failure': bool(failure)}


def chi_estimates(rows):
    eligible = [(name, r) for name,r in rows.items() if r.get('q', 0) >= 1e-30]
    if len(eligible) < 2:
        return {'fwd': None, 'bwd': None, 'effective_layers': len(eligible)}
    first, last = eligible[0][1], eligible[-1][1]
    steps = int(eligible[-1][0][1:])-int(eligible[0][0][1:])
    def ratio(a,b):
        return math.exp((math.log(a)-math.log(b))/steps) if a > 0 and b > 0 else None
    return {'fwd': ratio(last['q'], first['q']),
            'bwd': ratio(first.get('delta', {}).get('q', 0), last.get('delta', {}).get('q', 0)),
            'effective_layers': len(eligible), 'last_layer': eligible[-1][0]}


def probe(model, batch, seed, activation):
    # Copy preserves the training model, BN buffers, mode, gradients and RNG exactly.
    probe_model = copy.deepcopy(model).double().eval()
    device = next(model.parameters()).device
    x = batch.to(device=device, dtype=torch.float64)
    monitor = ActivationMonitor(probe_model.diagnostic_layers(), activation)
    activation_monitor = ActivationMonitor({name: module for name,module in probe_model.named_modules()
                       if isinstance(module, (torch.nn.ReLU, torch.nn.Tanh))}, activation)
    extra = {}
    branch_handles = []
    if hasattr(probe_model, 'blocks'):
        extra['branch_moments'] = {}
        for i,block in enumerate(probe_model.blocks):
            def branch(module, inputs, output, i=i, alpha=block.alpha):
                outq,inq = statistics(output).get('q'),statistics(inputs[0]).get('q')
                c = outq/inq if outq is not None and inq is not None and inq>0 else None
                cross = (inputs[0].detach().double()*output.detach().double()).mean().item()/inq if c is not None else None
                row = {'c':c,'input_q':inq,'branch_q':outq,'normalized_cross':cross,
                       'predicted_local_ratio_with_cross':1+alpha*alpha*c+2*alpha*cross if cross is not None else None}
                extra['branch_moments'][str(i+1)] = row
                if i==0:
                    extra['c'] = c
            branch_handles.append(block.branch.register_forward_hook(branch))
    try:
        h = probe_model.forward_features(x)
        g = torch.Generator().manual_seed(seed)
        u = torch.randn(h.shape[-1], generator=g, dtype=torch.float64)
        u = (u/u.norm()).to(device)
        (h@u).mean().backward()
        gradients = gradient_statistics(probe_model)
        if hasattr(probe_model, 'blocks'):
            endq,startq = monitor.rows[f'h{len(probe_model.blocks)}'].get('q'),monitor.rows['h0'].get('q')
            extra['q_ratio'] = endq/startq if endq is not None and startq is not None and startq>0 else None
            extra['growth_per_block'] = extra['q_ratio']**(1/len(probe_model.blocks)) if extra['q_ratio'] is not None else None
        result = {'activations': monitor.rows, 'activation_outputs': activation_monitor.rows,
                'gradients': gradients, 'chi': chi_estimates(monitor.rows),
                'trainability': summarize(monitor.rows, [gradients]), **extra,
                'mode': 'eval', 'model_dtype': 'float64 copy of float32 weights', 'statistics_dtype': 'float64',
                'batch_size': len(x), 'u_seed': seed, 'u_norm': u.norm().item()}
        extra_summary = summarize(activation_monitor.rows, [])
        for key in ['max_zero_fraction','max_saturated_fraction']:
            result['trainability'][key] = extra_summary[key]
        result['trainability']['numerical_failure'] |= extra_summary['numerical_failure']
        return result
    finally:
        monitor.close()
        activation_monitor.close()
        for handle in branch_handles:
            handle.remove()


class TrainingDiagnostics:
    def __init__(self, model, dataset, cfg, path):
        from src.utils.io import digest, write_json
        self.model, self.cfg, self.path = model, cfg, path
        n = cfg['diagnostics']['loss_subset_size']
        self.indices = list(range(min(n, len(dataset))))
        self.loss_x = torch.stack([dataset[i][0] for i in self.indices])
        self.loss_y = torch.tensor([dataset[i][1] for i in self.indices])
        self.batch = self.loss_x[:cfg['diagnostics']['probe_batch_size']]
        self.samples, self.monitor = [], None
        write_json(path/'diagnostic_protocol.json', {'loss_indices_in_train_subset': self.indices,
                   'probe_indices_in_train_subset': self.indices[:len(self.batch)],
                   'input_checksum': digest(self.batch.tolist()), 'probe_mode': 'eval', 'probe_dtype': 'float64',
                   'training_dtype': 'float32', 'gradient_steps': 'first, middle, last batch each epoch',
                   'GI_scope': 'all directly parameterized modules, including input and head; median per layer before GI',
                   'AVD_scope': 'hidden outputs (residual includes h0), fixed eval probe', 'u_seed': cfg['profile']['base_seed']})
        self.initial = probe(model, self.batch, cfg['profile']['base_seed'], cfg['model']['activation'])
        write_json(path/'probe.initial.json', self.initial)

    def begin(self, index, batches):
        if index in {0, batches//2, batches-1}:
            self.monitor = ActivationMonitor(self.model.diagnostic_layers(), self.cfg['model']['activation'])

    def end(self, epoch, index):
        if self.monitor is not None:
            from src.engine.runner import ExperimentRunner
            gradients = gradient_statistics(self.model)
            self.samples.append(gradients)
            ExperimentRunner._append(self.path/'diagnostics.jsonl', {'epoch': epoch, 'batch': index,
                                     'activations': self.monitor.rows, 'gradients': gradients})
            self.monitor.close()
            self.monitor = None

    def finish(self, failed=False):
        from src.utils.io import write_json
        if self.monitor is not None:
            if failed:
                write_json(self.path/'diagnostic.failure.json', {'activations': self.monitor.rows,
                           'gradients': gradient_statistics(self.model)})
            self.monitor.close()
            self.monitor = None
        result = probe(self.model, self.batch, self.cfg['profile']['base_seed'], self.cfg['model']['activation']) if not failed else self.initial
        device = next(self.model.parameters()).device
        loss = None
        if not failed:
            mode = self.model.training
            self.model.eval()
            with torch.no_grad():
                loss = F.cross_entropy(self.model(self.loss_x.to(device)), self.loss_y.to(device)).item()
            self.model.train(mode)
        summary = summarize(result['activations'], self.samples, loss)
        for key in ['max_zero_fraction','max_saturated_fraction']:
            summary[key] = result['trainability'][key]
        summary['numerical_failure'] |= failed
        summary['diagnostic_state'] = 'initial; training failed' if failed else 'final'
        if failed:
            summary['rho_T'] = summary['AVD'] = None
        else:
            write_json(self.path/'probe.final.json', result)
        return summary
