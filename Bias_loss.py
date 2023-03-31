import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def kernel_gauss(sigma):
    size = 4 * sigma + 1
    kernel = np.zeros((size,size))
    center = size // 2
    sigma2 = sigma * sigma * 2
    sum = 0

    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            kernel[i,j] = np.exp(-(x*x+y*y)/sigma2)
            sum = sum + kernel[i,j]

    kernel = kernel / sum
    return kernel

def classCenter(I,b,b2,up,C):

    center = torch.FloatTensor(C).cuda().zero_()
    for i in range(C):
        bd = I * b * up[:,i,:,:]
        db = b2 * up[:,i,:,:]
        bd_sum = torch.sum(bd)
        db_sum = torch.sum(db)
        center[i] = bd_sum / (db_sum + 1e-9)

    return center

class BiasLoss(nn.Module):
    def __init__(self):
        super(BiasLoss,self).__init__()
        kernel = kernel_gauss(sigma=5)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.cuda()
        self.weight = nn.Parameter(data=kernel,requires_grad=False)

    def forward(self, I, u, b, p, sigma=5):

        loss_sum = torch.FloatTensor(1).cuda().zero_()
        C = u.shape[1]
        bd = torch.zeros_like(b)
        db = torch.zeros_like(b)
        b_detach = b.detach()
        up = torch.pow(u,p) #u^p, 1*1*H*W
        b2 = torch.pow(b_detach,2) #b^2, 1*1*H*W

        v = classCenter(I,b,b2,up,C)
        for i in range(C):
            bd = bd + v[i] * up[:,i,:,:]
            db = db + v[i] * v[i] * up[:,i,:,:]

        bd = bd * I
        bd_smooth = F.conv2d(bd,self.weight,padding=2*sigma)
        db_smooth = F.conv2d(db,self.weight,padding=2*sigma)

        b_smooth = bd_smooth / db_smooth
        # b_max = torch.max(b_smooth)
        # b_smooth = b_smooth / b_max

        loss_sum = loss_sum + torch.mean((b - b_smooth) * (b - b_smooth))

        return loss_sum
