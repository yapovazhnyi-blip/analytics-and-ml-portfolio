Set-Content /tmp/check.py @'
import torch
from pathlib import Path

MODEL_DIR = Path('/app/data/models')
ckpts = sorted(MODEL_DIR.glob('ddpm_mnist_*.pt'))
print('Checkpoints:', [p.name for p in ckpts])

if ckpts:
    ckpt = torch.load(ckpts[-1], map_location='cpu')
    sd = ckpt.get('model_state_dict', ckpt)
    cfg = ckpt.get('config') if isinstance(ckpt, dict) else None
    print('Config in checkpoint:', cfg)
    print('input_conv:', sd['input_conv.weight'].shape)
    print('time_emb.1.bias:', sd['time_emb.1.bias'].shape)
    n_ds = sum(1 for k in sd if k.startswith('downsamples.'))
    n_levels = n_ds + 1
    n_dec = sum(1 for k in sd if k.startswith('decoder_blocks.') and k.endswith('.conv1.weight'))
    print('n_ds:', n_ds, 'n_levels:', n_levels, 'n_dec:', n_dec, 'n_res:', n_dec // n_levels)
    for k in sorted(sd):
        if k.startswith('decoder_blocks.') and k.endswith('.conv1.weight'):
            print(k, sd[k].shape)
'@