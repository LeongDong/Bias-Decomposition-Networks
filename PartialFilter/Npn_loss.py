import torch
import torch.nn as nn
from PartialConv import maskCreate, partialFilter, convParFilt, mediFilter

def classCenter(I, b, b2, up, e, C, kernel):
    center = torch.FloatTensor(C).cuda().zero_()
    b_K = convParFilt(b, kernel)
    b2_K = convParFilt(b2, kernel)
    for i in range(C - 1):
        bd = (I - e) * up[:, i + 1, :, :] * b_K
        db = up[:, i + 1, :, :] * b2_K
        bd_sum = torch.sum(bd)
        db_sum = torch.sum(db)
        center[i + 1] = bd_sum / (db_sum + 1e-9)

    bd0 = I * up[:, 0, :, :] * torch.ones_like(b)
    db0 = up[:, 0, :, :] * torch.ones_like(b)
    center[0] = torch.sum(bd0) / (torch.sum(db0) + 1e-9)
    return center

def sign(x):
    mask = torch.zeros_like(x)

    mask[x > 0] = 1
    mask[x < 0] = -1
    return mask

def softThres(x, y):

    sign_x = sign(x)
    x_y = torch.abs(x) - y
    x_y[x_y < 0] = 0
    return x_y * sign_x


class NoisePredictLoss(nn.Module):
    def __init__(self):
        super(NoisePredictLoss, self).__init__()

    def forward(self, I, u, b, e, p, size = 21, lamda = 21): #I：B*1*H*W; u:B*C*H*W; b:B*1*H*W

        C = u.shape[1]
        bd = torch.zeros_like(b) #B*1*H*W
        db = torch.zeros_like(b) #B*1*H*W
        b_detach = b.detach()
        u_detach = u.detach()
        e_detach = e.detach()
        up = torch.pow(u_detach, p) #B*C*H*W
        b2 = torch.pow(b_detach, 2) #B*1*H*W
        mask = maskCreate(I) #B*1*H*W
        kernel = partialFilter(mask, size) #B*1*H*W*size*size

        v = classCenter(I, b, b2, up, e_detach, C, kernel) #C
        for i in range(C):
            bd_sub = b_detach #B*1*H*W
            bd = bd + (I - v[i] * convParFilt(bd_sub, kernel)) * up[:, i, :, :]
            db_sub = up[:, i, :, :] #B*H*W
            db_sub = db_sub.unsqueeze(dim=1) #B*1*H*W
            db = db + db_sub

        bd = softThres(bd, lamda / 2)
        e_new = bd / (db + 1e-9)
        loss = torch.mean((e - e_new) * (e - e_new))

        return loss

