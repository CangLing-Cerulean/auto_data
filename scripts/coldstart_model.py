"""Batched form of the fixed, channel-independent DLinear backbone."""
import time
import torch
from torch import nn
from torch.nn import functional as F


class DLinear(nn.Module):
    def __init__(self, channels, length=32, horizon=8, kernel=25):
        super().__init__()
        self.length, self.horizon, self.kernel = length, horizon, kernel
        # Preserve the original per-branch Linear initialization order.
        seasonal = [nn.Linear(length, horizon) for _ in range(channels)]
        trend = [nn.Linear(length, horizon) for _ in range(channels)]
        self.seasonal_weight = nn.Parameter(torch.stack([m.weight.T.detach() for m in seasonal]))
        self.seasonal_bias = nn.Parameter(torch.stack([m.bias.detach() for m in seasonal]))
        self.trend_weight = nn.Parameter(torch.stack([m.weight.T.detach() for m in trend]))
        self.trend_bias = nn.Parameter(torch.stack([m.bias.detach() for m in trend]))

    def forward(self, values):
        x = values.transpose(1, 2)
        trend = F.avg_pool1d(F.pad(x, (self.kernel//2,)*2, mode='replicate'), self.kernel, stride=1)
        seasonal = x-trend
        ys = torch.bmm(seasonal.transpose(0, 1), self.seasonal_weight) + self.seasonal_bias[:, None]
        yt = torch.bmm(trend.transpose(0, 1), self.trend_weight) + self.trend_bias[:, None]
        return (ys+yt).permute(1, 2, 0)


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8,
                            weight_decay=0., foreach=False, fused=False)


def benchmark():
    """Engineering checks on generated tensors only, never dataset values."""
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(18421)
    model = DLinear(3).double()
    x = torch.randn(7, 32, 3, dtype=torch.float64)
    y = torch.randn(7, 8, 3, dtype=torch.float64)
    copies = [p.detach().clone().requires_grad_() for p in model.parameters()]
    sw, sb, tw, tb = copies
    z = x.transpose(1,2)
    trend = F.avg_pool1d(F.pad(z,(12,12),mode='replicate'),25,stride=1)
    reference = torch.stack([F.linear(z[:,c]-trend[:,c],sw[c].T,sb[c]) + F.linear(trend[:,c],tw[c].T,tb[c]) for c in range(3)],dim=2)
    actual = model(x)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    weights = torch.tensor([.1,.2,.5,1.,1.5,2.,3.],dtype=torch.float64)
    loss = (((actual-y)**2).mean((1,2))*weights).mean()
    reference_loss = (((reference-y)**2).mean((1,2))*weights).mean()
    loss.backward(); reference_loss.backward()
    gaps = []
    for p,q in zip(model.parameters(),copies):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-11,atol=1e-12)
        gaps.append(float((p.grad-q.grad).abs().max()))
    opt = optimizer(model)
    refopt = torch.optim.Adam(copies,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    opt.step();refopt.step()
    for p,q in zip(model.parameters(),copies):torch.testing.assert_close(p,q,rtol=1e-11,atol=1e-12)
    assert torch.cuda.is_available()
    torch.manual_seed(18422)
    full = DLinear(370).cuda(); opt = optimizer(full)
    bx = torch.randn(256,32,370,device='cuda'); by = torch.randn(256,8,370,device='cuda')
    torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        opt.zero_grad(set_to_none=True);((full(bx)-by)**2).mean().backward();opt.step()
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(100):
        opt.zero_grad(set_to_none=True);((full(bx)-by)**2).mean().backward();opt.step()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    return dict(status='PASS',channels=370,length=32,horizon=8,batch=256,
        mathematical_reference_max_gradient_gap=max(gaps),parameters=sum(p.numel() for p in full.parameters()),
        synthetic_optimizer_updates=112,measured_updates=100,seconds=elapsed,
        training_update_only_3000_step_extrapolation_seconds=elapsed*30,
        end_to_end_training_time_known=False,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        torch=torch.__version__,device=torch.cuda.get_device_name(),real_data_access=False,full_trainings=0)
