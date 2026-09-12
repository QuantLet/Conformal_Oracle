#!/usr/bin/env python3
"""Check that sampling optimisations retain the upstream one-step laws."""
import os
os.environ['HF_HUB_OFFLINE']='1'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from moirai_sampling import marginal, forward

ROOT=Path(__file__).resolve().parents[3]/'artifacts/extension_20260831'
torch.set_num_threads(2)
targets=[]
for asset in ['SP500','BTC']:
    r=pd.read_csv(ROOT/'data/returns'/f'{asset}.csv')
    targets.append(r.log_return.to_numpy(dtype=np.float32)[:512])
x=torch.from_numpy(np.stack(targets))[:,:,None]
obs=torch.ones_like(x,dtype=torch.bool);pad=torch.zeros((2,512))
module=MoiraiModule.from_pretrained(str(ROOT/'models/moirai'),local_files_only=True)
f=MoiraiForecast(module=module,prediction_length=1,context_length=512,patch_size='auto',num_samples=4,
                target_dim=1,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0).eval()
report={}
with torch.no_grad():
    for p in f.module.patch_sizes:
        d=f._get_distr(p,x,obs,pad)
        token=f.context_token_length(p)
        scalar=marginal(d,token)
        diffs=[]
        for value in [-.10,-.03,0.,.02,.10]:
            full=d.log_prob(torch.full(d.batch_shape,value))[:,token,0]
            small=scalar.log_prob(torch.full(scalar.batch_shape,value))
            diffs.append(float((full-small).abs().max()))
        assert max(diffs)<2e-5,(p,diffs)
        report[f'moirai_patch_{p}_log_density_max_difference']=max(diffs)
    # Compare the upstream and optimised auto-patch decisions on identical
    # padded histories. Sampling discarded coordinates must not affect them.
    tx=torch.nn.functional.pad(x,(0,0,1,0));ob=torch.nn.functional.pad(obs,(0,0,1,0));pa=torch.nn.functional.pad(pad,(1,0),value=1.)
    losses=[]; original_val=f._val_loss
    def record(*a,**kw):
        result=original_val(*a,**kw);losses.append(result.clone());return result
    f._val_loss=record
    torch.manual_seed(42);f(past_target=tx,past_observed_target=ob,past_is_pad=pa)
    original=torch.stack(losses);losses.clear()
    torch.manual_seed(42);out=forward(f,tx,ob,pa)
    changed=torch.stack(losses)
    assert torch.equal(original.argmin(0),changed.argmin(0))
    assert torch.allclose(original,changed,atol=1e-6,rtol=1e-6)
    assert out.shape==(2,4,1) and torch.isfinite(out).all()
    report['moirai_patch_loss_max_difference']=float((original-changed).abs().max())
    report['moirai_selected_patches']=f.last_selected_patches.tolist()

sys.path.insert(0,str(ROOT/'models/lag-llama-source'))
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.common import ListDataset
est=LagLlamaEstimator(prediction_length=1,context_length=512,input_size=1,n_layer=8,n_embd_per_head=36,n_head=4,
                     num_parallel_samples=2,batch_size=2,device=torch.device('cpu'),
                     rope_scaling={'type':'linear','factor':16.},ckpt_path=str(ROOT/'models/lagllama/lag-llama.ckpt'),time_feat=True)
lm=est.create_lightning_module();pred=est.create_predictor(est.create_transformation(),lm)
ds=ListDataset([{'target':c,'start':pd.Period('2000-01-04',freq='D')} for c in targets],freq='D')
captured=[]
def hook(mod,inputs,output):
    params,loc,scale=output
    captured.append([p[:,-1:].detach().clone() for p in params]+[loc.clone(),scale.clone()])
handle=lm.model.register_forward_hook(hook)
samples=[]
for single in [False,True]:
    lm.use_single_pass_sampling=single
    torch.manual_seed(42)
    with torch.no_grad():samples.append(np.stack([f.samples for f in pred.predict(ds,num_samples=2)]))
handle.remove()
assert len(captured)==2
diff=max(float((old-new.repeat_interleave(2,0)).abs().max()) for old,new in zip(*captured))
assert diff<2e-5,diff
report['lagllama_distribution_parameters_max_difference']=diff
report['lagllama_same_seed_samples_max_difference']=float(np.max(np.abs(samples[0]-samples[1])))
report['passed']=True
report['sampler_sha256']=hashlib.sha256(Path(__file__).with_name('moirai_sampling.py').read_bytes()).hexdigest()
(ROOT/'quality/sampling_optimisation_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
