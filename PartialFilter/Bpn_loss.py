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

class BiasPredictLoss(nn.Module):
    def __init__(self):
        super(BiasPredictLoss, self).__init__()

    def forward(self, I, u, b, e, p, size = 21): #I：B*1*H*W; u:B*C*H*W; b:B*1*H*W

        C = u.shape[1]
        # u = mediFilter(u, size=3)
        bd = torch.zeros_like(b) #B*1*H*W
        db = torch.zeros_like(b) #B*1*H*W
        b_detach = b.detach()
        e_detach = e.detach()
        up = torch.pow(u, p) #B*C*H*W
        b2 = torch.pow(b_detach, 2) #B*1*H*W
        mask = maskCreate(I) #B*1*H*W
        kernel = partialFilter(mask, size) #B*1*H*W*size*size

        v = classCenter(I, b, b2, up, e_detach, C, kernel) #C
        for i in range(C):
            bd_sub = (I - e_detach) * v[i] * up[:, i, :, :] #B*1*H*W
            bd = bd + convParFilt(bd_sub, kernel)
            db_sub = v[i] * v[i] * up[:, i, :, :] #B*H*W
            db_sub = db_sub.unsqueeze(dim=1) #B*1*H*W
            db = db + convParFilt(db_sub, kernel)

        bd = bd * mask + (1 - mask)
        db = db * mask + (1 - mask)
        b_new = bd / (db + 1e-9)
        loss = torch.mean((b - b_new) * (b - b_new))

        return loss

