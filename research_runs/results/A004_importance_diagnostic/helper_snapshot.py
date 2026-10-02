"""Exact per-window DLinear MSE gradient norms; no validation information."""
import torch


@torch.no_grad()
def gradient_norm(model, x, y):
    channels=x.transpose(1,2)
    trend=torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(channels,(model.kernel//2,)*2,mode='replicate'),model.kernel,stride=1)
    seasonal=channels-trend
    error=(model(x)-y).double()
    feature_energy=seasonal.double().square().sum(-1)+trend.double().square().sum(-1)+2
    # Separate channel/horizon weights and two biases; no parameter sharing.
    squared=(error.square().sum(1)*feature_energy).sum(1)*4/(y.shape[1]*y.shape[2])**2
    return squared.sqrt()


def distribution(norm):
    if not torch.isfinite(norm).all() or torch.any(norm<0): raise ValueError('Invalid gradient norm')
    if norm.sum()==0: return torch.full_like(norm,1/len(norm))
    return .5/len(norm)+.5*norm/norm.sum()


def verify_norm(model, x, y):
    measured=gradient_norm(model,x,y)
    reference=[]
    for index in range(len(x)):
        loss=(model(x[index:index+1])-y[index:index+1]).square().mean()
        gradients=torch.autograd.grad(loss,tuple(model.parameters()))
        reference.append(sum(g.double().square().sum() for g in gradients).sqrt())
    actual=torch.stack(reference)
    relative=((actual-measured).abs()/actual.clamp_min(1e-12)).max().item()
    if relative>1e-5: raise AssertionError(f'Gradient norm mismatch: {relative}')
    return relative
