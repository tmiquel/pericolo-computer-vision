import numpy as np
import matplotlib.pyplot as plt

def imshow(img, figsize=(6, 6), cmap=None):
    """Notebook based function to plot an image
    
    Args:
    -----
        img (np.array): Image array
        figsize (tuple, optional): Plot size. Defaults to (6, 6).
        cmap (str, optional): Color map name. Defaults to None.
    """
    _ = plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.show()

def imshow_masked(img, mask, figsize=(6,6), factor=0.3, cmap=None):
    """[summary]
    
    Args:
    -----
        img (np.array): Image array
        mask (np.array): Mask binary array
        figsize (tuple, optional): Plot size. Defaults to (6, 6).
        factor (float, optional): Shadowing factor for the mask background. Defaults to 0.3.
        cmap (str, optional): Color map name. Defaults to None.
    """
    _ = plt.figure(figsize=figsize)
    aux_img = img.astype(float)
    aux_mask = np.expand_dims(mask.astype(float)*(1-factor) + factor, axis=-1)
    drk_img = (aux_img * aux_mask).astype(np.uint8)
    plt.imshow(drk_img)
    plt.show()