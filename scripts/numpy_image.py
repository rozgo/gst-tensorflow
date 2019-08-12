from PIL import Image
import numpy as np

INPUT_SIZE = 512

image = Image.open('image.png')

width, height = image.size
resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
target_size = (int(resize_ratio * width), int(resize_ratio * height))

print(width, height, resize_ratio, target_size)

resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

print(resized_image.size)

tensor = [np.asarray(resized_image)]

print(tensor)

# im2arr = np.array(resized_image) # im2arr.shape: height x width x channel
# arr2im = Image.fromarray(im2arr)
# print(im2arr)

