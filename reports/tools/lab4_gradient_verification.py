"""Independent full NT-Xent gradient check, including symmetric anchors and L2 chain rule."""
import torch
from src.ssl.losses import nt_xent
from src.utils.io import write_json

def main():
    torch.manual_seed(214);n,d,tau=7,11,.1
    a=torch.randn(n,d,dtype=torch.float64,requires_grad=True);b=torch.randn(n,d,dtype=torch.float64,requires_grad=True)
    loss,_=nt_xent(a,b,tau);ga,gb=torch.autograd.grad(loss,(a,b))
    x=torch.cat((a.detach(),b.detach()));norm=x.norm(dim=1,keepdim=True);z=x/norm;m=2*n
    scores=(z@z.T/tau).masked_fill(torch.eye(m,dtype=torch.bool),float('-inf'))
    p=scores.softmax(1);targets=torch.zeros_like(p);targets[torch.arange(m),(torch.arange(m)+n)%m]=1
    gz=((p-targets)+(p-targets).T)@z/(m*tau)
    expected=(gz-z*(gz*z).sum(1,keepdim=True))/norm
    error=(torch.cat((ga,gb))-expected).abs().max().item()
    assert error<=1e-5
    write_json('experiments/lab4/full_loss_gradient_verification.json',{'status':'PASS','max_absolute_error':error,
        'tolerance':1e-5,'N':n,'d':d,'tau':tau,'derivation':'Eq4.6 peranchor + candidate-column contributions, averaged over2N, followed by exact L2 normalization chain rule',
        'implementation_under_test':'src.ssl.losses.nt_xent; autograd on raw z1/z2','dtype':'float64','not_a_training_result':True})
    print('Full NT-Xent gradient error',error)
if __name__=='__main__':main()
