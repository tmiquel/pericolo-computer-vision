def IoU(mask_a, mask_b):
    """Compute the Intersection Over Union
    
    Args:
    -----
        mask_a (np.array): Mask binary array
        mask_b (np.array): Mask binary array
    
    Returns:
    --------
        float: Intersection over Union score
    """
    I = mask_a * mask_b
    U = (mask_a + mask_b).clip(max=1)
    IOU = I.sum() / U.sum()
    return IOU

def inclusion_ratio(small_mask, large_mask):
    """Compute the Inclusion Ratio score
    
    Args:
    -----
        small_mask (np.array): Mask binary array
        large_mask (np.array): Mask binary array
    
    Returns:
    --------
        score: Inclusion Ratio score
    """
    I = small_mask * large_mask
    ratio = I.sum() / small_mask.sum()
    return ratio