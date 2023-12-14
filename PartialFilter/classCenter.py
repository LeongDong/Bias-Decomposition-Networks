import torch
from PartialConv import convParFilt

def classCenter(I, b, b2, up, e, C, kernel):
    center = torch.FloatTensor(C).cuda().zero_()
    I_eb = (I - e) * b
    KI_eb = convParFilt(I_eb, kernel)
    K_b2 = convParFilt(b2, kernel)
    for i in range(C):
        bd = up[: , i, :, :] * KI_eb
        db = K_b2 * up[:, i, :, :]
        bd_sum = torch.sum(bd)
        db_sum = torch.sum(db)
        center[i] = bd_sum / (db_sum + 1e-9)

    # bd0 = I * up[:, 0, :, :]
    # db0 = up[:, 0, :, :]
    # center[0] = torch.sum(bd0) / torch.sum(db0 + 1e-9)

    return center
