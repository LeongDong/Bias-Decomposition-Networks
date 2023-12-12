import torch
import torch.nn as nn
from classCenter import classCenter
from PartialConv import maskCreate, partialFilter, convParFilt

class ProbPredictLoss(nn.Module):
    def __init__(self):
        super(ProbPredictLoss, self).__init__()

    def forward(self, I, u, b, e, p, size = 3):

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
        I_e = I - e_detach
        I_e2 = I_e * I_e
        KI_e2 = convParFilt(I_e2, kernel)
        KI_e = convParFilt(I_e, kernel)
        for i in range(C):
            d = KI_e2 - 2 * v[i] * b * KI_e + b * b * v[i] * v[i]
            D[:, i, :, :] = torch.pow(d, q) + 1e-9

        f = 1 / D
        f_sum = torch.sum(f, dim = 1, keepdim = True)
        for i in range(C):
            new_u[:, i, :, :] = 1 / (D[:, i, :, :] * f_sum + 1e-9)

        loss = torch.mean((u - new_u) * (u - new_u))
        return loss

