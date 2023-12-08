import torch
import numpy as np
import torch.nn.functional as F

def maskCreate(image): #B*1*H*W

    image = image * 255
    image[image <= 20] = 0
    image[image > 20] = 1
    return image #B*1*H*W

def GaussianKernel(size):

    sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    center = size // 2
    ele = (np.arange(size, dtype=np.float64) - center)

    kernel1d = np.exp(- (ele ** 2) / (2 * sigma ** 2)) #size * 1
    kernel = kernel1d[..., None] @ kernel1d[None, ...] #size * size
    kernel = torch.from_numpy(kernel)

    return kernel.unsqueeze(0).unsqueeze(0) #1*1*size*size, Unnormalized gaussian kernel

def partialFilter(mask, size): #mask: B*C*H*W

    kernel = GaussianKernel(size)
    kernel = kernel.cuda()
    pad = (size - 1) // 2
    maskPad = F.pad(mask, pad=[pad, pad, pad, pad], mode='reflect') # B*C*(H+2pad)*(W+2pad)
    patches = maskPad.unfold(2, size, 1).unfold(3, size, 1) #B*C*H*W*size*size

    parKernels = patches * kernel.unsqueeze(0).unsqueeze(0) #(B*C*H*W*size*size)*(1*1*1*1*size*size)->B*C*H*W*size*size
    kernelNorm = parKernels / (parKernels.sum(dim=(-1,-2), keepdim=True) + 1e-9) #B*C*H*W*size*size
    kernelNorm = kernelNorm * mask.unsqueeze(-1).unsqueeze(-1) #(B*C*H*W*size*size) * (B*C*H*W*1*1)->B*C*H*W*size*size

    return kernelNorm

def convParFilt(image, kernel): #B*C*H*W

    size = kernel.shape[-1]
    pad = (size - 1) // 2
    imgPad = F.pad(image, pad=[pad, pad, pad, pad], mode='reflect') #B*C*(H+2pad)*(W+2pad)
    imgPatches = imgPad.unfold(2, size, 1).unfold(3, size, 1) #B*C*H*W*size*size
    imgConv = kernel * imgPatches #B*C*H*W*size*size
    return imgConv.sum(dim=(-1, -2), keepdim=False) #B*C*H*W

def mediFilter(image, size):

    N, B, H, W = image.shape
    pad = (size - 1) // 2
    imgPad = F.pad(image, pad = [pad, pad, pad, pad], mode = 'reflect')
    patches = imgPad.unfold(2, size, 1).unfold(3, size, 1) #N*B*H*W*size*size

    patchFlat = patches.reshape(N,B,H,W,size * size) #N*B*H*W*(size*size)
    patchMedia = torch.median(patchFlat, dim=-1, keepdim=False) #N*B*H*W

    return patchMedia[0]

def LSQSurfFit(oribias, img):

    N, B, He, Wi = oribias.shape
    mask = maskCreate(img)
    t = 0
    sx = []
    sy = []
    sz = []
    for i in torch.arange(He):
        for j in torch.arange(Wi):
            if(oribias[0, 0, i, j] > 0): #img: 1*1*H*W
                sy.append(i / (He - 1) * 2 - 1)
                sx.append(j / (Wi - 1) * 2 - 1)
                sz.append(oribias[0, 0, i, j])
                t = t + 1

    sx = torch.tensor(sx)
    sy = torch.tensor(sy)
    sz = torch.tensor(sz)
    W = torch.zeros([t, 13], dtype=torch.float32)
    W[:, 0] = 1
    W[:, 1] = sx
    W[:, 2] = sy
    W[:, 3] = sx * sy
    W[:, 4] = sx * sx
    W[:, 5] = sy * sy
    W[:, 6] = sx * sx * sy
    W[:, 7] = sx * sy * sy
    W[:, 8] = sx * sx * sx
    W[:, 9] = sy * sy * sy
    W[:, 10] = torch.sin(sx)
    W[:, 11] = torch.sin(sy)
    W[:, 12] = torch.sin(sx * sy)

    a1 = torch.inverse(torch.matmul(W.t(), W))
    a2 = torch.matmul(a1, W.t())
    a = torch.matmul(a2, sz.t())

    bias = torch.zeros([N, B, He, Wi])
    bias = bias.cuda()
    for i in torch.arange(-1.0, 1.0, 2 / (He-1)):
        for j in torch.arange(-1.0, 1.0, 2 / (Wi-1)):
            ti = (i + 1) / 2 * (He - 1)
            tj = (j + 1) / 2 * (Wi - 1)
            ti = torch.floor(ti + 0.5)
            tj = torch.floor(tj + 0.5)

            bias[0, 0, ti.long(), tj.long()] = a[0] + a[1] * j + a[2] * i + a[3] * j * i + a[4] * j * j + a[5] * i * i + a[
                6] * j * j * i + a[7] * i * i * j + a[8] * j * j * j + a[9] * i * i * i + a[10] * torch.sin(j) + a[
                                     11] * torch.sin(i) + a[12] * torch.sin(i * j)

    bias = bias / torch.max(bias * mask) * mask
    return bias