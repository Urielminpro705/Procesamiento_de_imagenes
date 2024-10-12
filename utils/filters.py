import numpy as np
import math

def conv2d(img, k):
    k_rotado = np.rot90(k, k=2)
    hk,wk = k_rotado.shape
    h,w = img.shape
    padding = math.floor(hk/2)
    if not hk % 2:
        print("El tamaño del kernel tiene que se un numero impar")
        return img
    else:
        padding_img = np.zeros((h+(2*padding),w+(2*padding)))
        img_out = np.zeros((h,w))
        for f in range(padding, h+padding):
            for c in range(padding, w+padding):
                padding_img[f,c] = img[f-padding,c-padding]
        
        for f in range(h):
            for c in range(w):
                total_celda = 0
                for fk in range(hk):
                    for ck in range(wk):
                        total_celda = total_celda + (k_rotado[fk,ck] * padding_img[fk+f,ck+c])
                img_out[f,c] = total_celda
        return img_out
    
def filtro_promedio(n):
    if n % 2:
        k = np.ones((n,n))/(n**2)
        return k
    else:
        print("Kernel size should be an odd number")

def gaussian_kernel(sigma=1, n=3):
    if n % 2 == 0:
        raise ValueError("Kernel size should be an odd number")
    
    # Define grid of (x, y) coordinates
    a = -(n // 2)
    b = n // 2
    x, y = np.meshgrid(np.arange(a, b + 1), np.arange(a, b + 1))

    # Compute Gaussian function
    g = 1 / (2 * np.pi * sigma**2) * np.exp(-((x**2 + y**2) / (2 * sigma**2)))

    # Normalize so that the sum of the kernel is 1
    g /= g.sum()

    return g

# img = np.array([
#     [15,13,13],
#     [23,27,23],
#     [23,27,23],
# ])

# kernel = [
#     [-1,-1,-1],
#     [0,5,0],
#     [-1,0,0]
# ]

# img_out = conv2d(img, kernel)
# print(img_out)