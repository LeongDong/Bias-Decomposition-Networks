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

class ProbPredictLoss(nn.Module):
    def __init__(self):
        super(ProbPredictLoss, self).__init__()

    def forward(self, I, u, b, e, p, size = 21):

        C = u.shape[1]
        D = torch.zeros_like(u) #B*C*H*W
        new_u = torch.zeros_like(u) #B*C*H*W
        u_detach = u.detach()
        e_detach = e.detach()
        up = torch.pow(u_detach, p)
        b2 = torch.pow(b, 2)
        q = 1 / (p - 1)
        mask = maskCreate(I) #B*1*H*W
        kernel = partialFilter(mask, size) #B*C*H*W*size*size

        v = classCenter(I, b, b2, up, e_detach, C, kernel)
        b_cov = convParFilt(b, kernel)
        b_cov = b_cov * mask + (1 - mask) #B*C*H*W
        for i in range(C):
            d = (I - e_detach - v[i] * b_cov) * (I - e_detach - v[i] * b_cov)
            D[:, i, :, :] = torch.pow(d, q) + 1e-9

        f = 1 / D
        f_sum = torch.sum(f, dim = 1, keepdim = True)
        for i in range(C):
            new_u[:, i, :, :] = 1 / (D[:, i, :, :] * f_sum + 1e-9)

        # new_u = mediFilter(new_u, size=3)
        loss = torch.mean((u - new_u) * (u - new_u))
        return loss

