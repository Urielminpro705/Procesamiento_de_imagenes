from utils import io_image as io
img = io.read_image('data/images/pepper.jpg')

lista = [0,1,2,3,4,5,6,7]
# img_lol = io.image_bit_combination(img,lista)
# io.print_img(img_lol)

io.planes_print(io.extract_bit_planes(img),2,4)