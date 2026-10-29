import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv

class GraphNetwork(nn.Module):
    """Depth counts convolutions including unactivated output; skips on hidden layers."""
    def __init__(self,cfg):
        super().__init__();self.kind=cfg['kind'];self.dropout=cfg['dropout'];self.residual=cfg.get('residual',False)
        self.layers=nn.ModuleList();self.skips=nn.ModuleList()
        dims=[cfg['input_dim']]+[cfg['width']]*(cfg['depth']-1)+[cfg['classes']]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.get('initialization_seed',1))
            for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
                last=i==len(dims)-2
                if self.kind=='mlp':layer=nn.Linear(a,b)
                elif self.kind=='gcn':layer=GCNConv(a,b,cached=False,add_self_loops=True,normalize=True)
                elif self.kind=='gat':layer=GATConv(a,b if last else b//8,heads=1 if last else 8,concat=not last,dropout=self.dropout,add_self_loops=True)
                elif self.kind=='sage':layer=SAGEConv(a,b)
                else:raise ValueError(self.kind)
                self.layers.append(layer)
                self.skips.append(nn.Linear(a,b,bias=False) if self.residual and not last and a!=b else nn.Identity())
            if self.kind in {'mlp','gcn'}:
                # Matching Xavier weights independent of constructor RNG consumption.
                gen=torch.Generator().manual_seed(cfg.get('initialization_seed',1))
                for layer in self.layers:
                    weight=layer.weight if self.kind=='mlp' else layer.lin.weight
                    nn.init.xavier_uniform_(weight,generator=gen)
                    nn.init.zeros_(layer.bias)
        edges=torch.empty((2,0),dtype=torch.long)
        if cfg.get('graph_path'):edges=torch.load(cfg['graph_path'],weights_only=False)['edge_index']
        self.register_buffer('_edge_index',edges,persistent=False)

    def forward_features(self,x,edge_index=None):
        edges=self._edge_index if edge_index is None else edge_index;out=[]
        for i,layer in enumerate(self.layers):
            prev=x;z=F.dropout(x,p=self.dropout,training=self.training)
            x=layer(z) if self.kind=='mlp' else layer(z,edges)
            if i<len(self.layers)-1:
                x=F.relu(x)
                if self.residual:x=self.skips[i](prev)+x
            out.append(x)
        return out

    def forward(self,x,edge_index=None):
        return self.forward_features(x,edge_index)[-1]
