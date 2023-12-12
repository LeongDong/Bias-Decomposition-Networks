import torch
from PartialConv import convParFilt

def classCenter(I, b, b2, up, e, C, kernel):
    center = torch.FloatTensor(C).cuda().zero_()
    I_e = I - e
    KI_e = convParFilt(I_e, kernel)
    for i in range(C - 1):
        bd = up[: , i + 1, :, :] * b * KI_e
        db = b2 * up[:, i + 1, :, :]
        bd_sum = torch.sum(bd)
        db_sum = torch.sum(db)
        center[i + 1] = bd_sum / (db_sum + 1e-9)

    bd0 = I * up[:, 0, :, :]
    db0 = up[:, 0, :, :]
    center[0] = torch.sum(bd0) / torch.sum(db0 + 1e-9)

    return center
