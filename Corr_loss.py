import torch
import torch.nn as nn

def classCenter(I,b,b2,up,C):

    center = torch.FloatTensor(C).cuda().zero_()
    for i in range(C):
        bd = I * b * up[:,i,:,:]
        db = b2 * up[:,i,:,:]
        bd_sum = torch.sum(bd)
        db_sum = torch.sum(db)
        center[i] = bd_sum / (db_sum + 1e-9)

    return center

class CorrLoss(nn.Module):
    def __init__(self):
        super(CorrLoss,self).__init__()

    def forward(self, I, u, b, p):

        loss_sum = torch.FloatTensor(1).cuda().zero_()
        C = u.shape[1]
        D = torch.zeros_like(u)
        new_u = torch.zeros_like(u)
        u_detach = u.detach()
        up = torch.pow(u_detach,p)
        b2 = torch.pow(b,2)

        v = classCenter(I,b,b2,up,C)

        for i in range(C):
            D[:,i,:,:] = (I - v[i] * b) * (I - v[i] * b) + 1e-9

        q = 1 / (p - 1)
        f = 1 / torch.pow(D,q)
        f_sum = torch.sum(f,dim=1,keepdim=True)#1,1,H,W
        for i in range(C):
            new_u[:,i,:,:] = 1 / (torch.pow(D[:,i,:,:],q) * f_sum)

        loss_sum = loss_sum + torch.mean((u - new_u) * (u - new_u))
        return loss_sum


