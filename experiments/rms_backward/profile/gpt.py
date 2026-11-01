import copy, json, time
import torch
from torch import nn
from torch.nn import functional as F
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.rms_norm import RMSNorm
import diff_gpt.domain.model.mha as mha

torch.set_num_threads(1)
torch.manual_seed(155)
base=GPT(97,64,2,4,4,pad_vocab_size_to=1,use_checkpoint=False).cuda()
ids=torch.randint(97,(256,2),device='cuda')
rows=[]
for name in ('native','accurate'):
    model=copy.deepcopy(base)
    if name=='native':
        for parent in model.modules():
            for key, child in list(parent.named_children()):
                if isinstance(child,RMSNorm):
                    setattr(parent,key,nn.RMSNorm(child.normalized_shape,elementwise_affine=False).cuda())
        mha.rms_norm=lambda value: F.rms_norm(value,(value.shape[-1],))
    else:
        from diff_gpt.domain.model.rms_norm import rms_norm
        mha.rms_norm=rms_norm
    compiled=torch.compile(model,mode='max-autotune',fullgraph=True,dynamic=False)
    def step():
        model.zero_grad(set_to_none=True)
        compiled(ids,ids)[1].backward()
    start=time.perf_counter()
    for _ in range(5):step()
    torch.cuda.synchronize()
    preparation=time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(5):
        start=time.perf_counter()
        for _ in range(100):step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)/100)
    rows.append({'variant':name,'preparation_seconds':preparation,'step_seconds':samples,'peak_bytes':torch.cuda.max_memory_allocated(),'finite_gradients':all(torch.isfinite(p.grad).all().item() for p in model.parameters())})
    print(json.dumps(rows[-1]),flush=True)
    del compiled,model
    torch._dynamo.reset()
print(json.dumps({'torch':str(torch.__version__),'device':torch.cuda.get_device_name(),'cases':rows},indent=2))
