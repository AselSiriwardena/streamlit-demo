import torch
import torch.optim
import model
import numpy as np
import gc
import os


def lowlight(image):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_lowlight = (np.asarray(image) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
    _, enhanced_image, _ = DCE_net(data_lowlight)

    gc.collect()
    torch.cuda.empty_cache()

    return enhanced_image