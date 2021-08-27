#%%

import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

resnet18 = models.resnet18(False)
writer = SummaryWriter('logs/exp-1')
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

for n_iter in range(100):
    dummy_s1 = torch.rand(1)
    dummy_s2 = torch.rand(1)
    writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

    writer.add_scalars(
        'data/scalar_group',{
            'xsinx': n_iter * np.sin(n_iter), 
            'xcosx': n_iter * np.cos(n_iter),
            'arctanx': np.arctan(n_iter)
            }, n_iter)

# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()