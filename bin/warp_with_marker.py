from libs.image_warper import ImageWarper
from skimage import io

if __name__ == '__main__':
    import sys
    image_name = sys.argv[-1]
    save_name = '.'.join(image_name.split('.')[:-1]) + '_warped.png'
    iw = ImageWarper(image_name)
    print("Rectifying {}".format(image_name))
    result_image = iw.warp_with_marker()
    if result_image is not None:
        io.imsave(save_name, result_image)