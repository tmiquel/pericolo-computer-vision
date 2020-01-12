import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import distance as dist

def image_to_tile(im, tile_size=8):
    """Convert an image to sub-tiled images
    
    Args:
        im (np.array): Image
        tile_size (int, optional): Tile Size in pixel. Defaults to 8.
    
    Returns:
        list: List of generated tiles
    """
    M = im.shape[0]//tile_size
    N = im.shape[1]//tile_size
    tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
    return tiles

def tiles_profile(tiles, tile_size=8):
    """Creates an image profile by checking the strongest color on each sub-tiles
    
    Args:
        tiles (list): List of generated tiles
        tile_size (int, optional): Tile Size in pixel. Defaults to 8.
    
    Returns:
        np.array: Profile of a marker
    """
    medians = list(map(lambda x: np.median(x),tiles))
    medians = np.array(medians) // 255
    medians = medians.reshape((tile_size, tile_size))
    return medians

def polygon_to_mask(width, height, polygon):
    """From a list a points and a given size generate a binary mask
    
    Args:
    -----
        width (int): Image width 
        height (int): Image height
        polygon (list): list of points (contour)
    
    Returns:
    --------
        np.array: Binary mask
    """
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    return np.array(img)

def order_points(pts):
    """Order a list of 4 points to `top_left`, `top_right`, `bottom_right`, `bottom_left`
    
    Args:
    -----
        pts (list): List of points to order
    
    Returns:
    --------
        list: Ordered list of points
    """
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")