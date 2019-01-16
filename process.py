from PIL import Image
import os

data_path = "./Coins/OttomanCropped/"
dirs = os.listdir(data_path)


def resize_imgs(final_size):
    for item in dirs:
         im = Image.open(data_path+item)
         f, e = os.path.splitext(data_path+"Resized_"+ item)
         size = im.size
         ratio = float(final_size) / max(size)
         new_image_size = tuple([int(x*ratio) for x in size])
         im = im.resize(new_image_size, Image.ANTIALIAS)
         new_im = Image.new("RGB", (final_size, final_size))
         new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
         new_im.save(f + 'resized_.jpg', 'JPEG', quality=100)


resize_imgs(256)
