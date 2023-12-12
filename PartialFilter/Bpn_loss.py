import torch
import torch.nn as nn
from classCenter import classCenter
from PartialConv import maskCreate, partialFilter, convParFilt

class BiasPredictLoss(nn.Module):
    def __init__(self):
        super(BiasPredictLoss, self).__init__()

    def forward(self, I, u, b, e, p, size = 3): #I：B*1*H*W; u:B*C*H*W; b:B*1*H*W

        C = u.shape[1]
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
            bd_sub = I - e_detach #B*1*H*W
            bd = bd + convParFilt(bd_sub, kernel) * v[i] * up[:, i, :, :]
            db_sub = v[i] * v[i] * up[:, i, :, :] #B*H*W
            db_sub = db_sub.unsqueeze(dim=1) #B*1*H*W
            db = db + db_sub

        bd = bd * mask + (1 - mask)
        db = db * mask + (1 - mask)
        b_new = bd / (db + 1e-9)
        loss = torch.mean((b - b_new) * (b - b_new))

        return loss

