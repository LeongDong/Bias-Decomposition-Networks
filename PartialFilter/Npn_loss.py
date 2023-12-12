import torch
import torch.nn as nn
from classCenter import classCenter
from PartialConv import maskCreate, partialFilter, convParFilt

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

    def forward(self, I, u, b, e, p, size = 3, lamda = 21): #I：B*1*H*W; u:B*C*H*W; b:B*1*H*W

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
            up_i = up[:, i, :, :]
            up_i = up_i.unsqueeze(dim=1)
            db_sub = convParFilt(up_i, kernel)
            bd_sub =  I * db_sub - convParFilt(b * v[i] * up[:, i, :, :], kernel)#B*1*H*W
            bd = bd + bd_sub
            db = db + db_sub

        bd = softThres(bd, lamda / 2)
        e_new = bd / (db + 1e-9)
        loss = torch.mean((e - e_new) * (e - e_new))

        return loss

