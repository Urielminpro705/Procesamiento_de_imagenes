from utils import io_image as io
img = io.read_image('data/images/pim.jpg')

io.print_img(io.image_bit_combination(img,[0,1,2,3,4,5,6,7]))

# io.planes_print(io.extract_bit_planes(img),2,4)