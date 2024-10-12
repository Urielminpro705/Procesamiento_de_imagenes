from scipy.signal import convolve2d, correlate2d
from utils import io_image
from utils import operaciones_img
import numpy as np
from utils.filters import filtro_promedio

img = io_image.read_image("data/images/pim.jpg")
kernel = np.array([
    [-1,-2,1],
    [0,0,0],
    [1,2,1]])



# def gaussian_kernel(sigma=1, n=3):
    

k_pro = filtro_promedio(9)
img = operaciones_img.agrega_ruido_gaussiano(img, 10)
img_corre = correlate2d(img, k_pro, mode='full', boundary='fill', fillvalue=0)
img_conv = convolve2d(img, k_pro, mode='full', boundary='fill', fillvalue=0)
io_image.planes_print([img,img_corre,img_conv], ["Normal","Correlacion","Convolucion"], 1, 3)