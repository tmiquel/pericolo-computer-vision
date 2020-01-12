import logging
import numpy as np
import cv2
from config import MIN_MATCH_COUNT
from utils.plot import imshow

def train_sift(train_img, draw=False, figsize=(8, 8)):
    """Train a SIFT detector
    
    Args:
    -----
        train_img (np.array): `GRAYSCALE` Image of the element to detect by the SIFT detector
        draw (bool, optional): Whether or not to plot the image. Defaults to False.
        figsize (tuple, optional): Plot size. Defaults to (8,8).
    
    Returns:
    --------
        [type]: SIFT Detector
        [type]: Keypoints
        [type]: Descriptors
        [type]: FLANN Matcher
    """
    FLANN_INDEX_KDITREE = 0
    flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
    flann = cv2.FlannBasedMatcher(flannParam, {})

    # Training
    sift_detector = cv2.xfeatures2d.SIFT_create()
    sift_trainImg = train_img.copy()
    sift_trainKP, sift_trainDesc = sift_detector.detectAndCompute(sift_trainImg, None)
    if draw:
        sift_trainImg1 = cv2.drawKeypoints(
            sift_trainImg, sift_trainKP, None, (255, 0, 0), 4
        )
        print("Detector Result on Train Image")
        imshow(sift_trainImg1, figsize=figsize)

    return sift_detector, sift_trainKP, sift_trainDesc, flann


def query_sift(query_img, sift_detector, sift_trainDesc, flann):
    """Queries a SIFT detector to find matches in an image
    
    Args:
    -----
        query_img (np.array): Image to check for element presence
        sift_detector ([type]): SIFT Detector
        sift_trainDesc ([type]): SIFT Descriptors
        flann ([type]): FLANN Matcher
    
    Returns:
    --------
        [type]: [description]
    """
    # Querying
    sift_QueryImgBGR = query_img.copy()
    sift_QueryImg = cv2.cvtColor(sift_QueryImgBGR, cv2.COLOR_BGR2GRAY)
    sift_queryKP, sift_queryDesc = sift_detector.detectAndCompute(sift_QueryImg, None)
    sift_matches = flann.knnMatch(sift_queryDesc, sift_trainDesc, k=2)
    sift_goodMatch = []

    for m, n in sift_matches:
        if m.distance < 0.75 * n.distance:
            sift_goodMatch.append(m)
    logging.debug(f"Found matches: {len(sift_goodMatch)}")
    return sift_goodMatch, sift_queryKP, sift_queryDesc


def compute_detector_corners(trainImg, goodMatch, trainKP, queryKP, force_true=False):
    """Compute the position of the element to detect by SIFT Detector in the query image
    
    Args:
    -----
        trainImg (np.array): `GRAYSCALE` Image of the element to detect by the SIFT detector
        goodMatch (list): List of points matching from `trainImg` to query image
        trainKP ([type]): SIFT Keypoints from training image
        queryKP ([type]): SIFT Keypoints from query image
        force_true (bool, optional): If True, bypass the check of `MIN_MATCH_COUNT`. Defaults to False.
    
    Returns:
    --------
        list: list of contour points in the train image (corresponding to the image corners/size)
        list: list of contour points in the query image
    """
    if (len(goodMatch) >= MIN_MATCH_COUNT) or force_true:
        tp = []
        qp = []
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp, qp = np.float32((tp, qp))
        H, _ = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h, w = trainImg.shape
        trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        return trainBorder, queryBorder
    else:
        logging.warning("Not Enough match found")
        return [], []


def draw_detector_borders(image, queryBorder, color=(0, 255, 0), thickness=5, draw=True, figsize=(8, 8)):
    """Generate an image with the SIFT detected element borders
    
    Args:
    -----
        image (np.array): [description]
        queryBorder ([type]): Borders of the element as a list of points
        color (tuple, optional): Integer tuple for BGR color. Defaults to (0,255,0).
        thickness (int, optional): Line thickness. Defaults to 5.
        draw (bool, optional): Whether or not to plot the image. Defaults to True.
        figsize (tuple, optional): Plot size. Defaults to (8,8).
    
    Returns:
    --------
        np.array: Image with drawn borders
    """
    QueryImgBGR = image.copy()
    cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, color, thickness)
    if draw:
        imshow(QueryImgBGR, figsize=figsize)
    return QueryImgBGR


def detect_object_via_sift(query_image, train_image, draw=False, figsize=(8, 8)):
    """Full SIFT detection process
    
    Args:
    -----
        query_image ([type]): BGR Image representing the scene to investigate
        train_image ([type]): GRAYSCALE Image representing the object to detect
        draw (bool, optional): Whether or not to plot the image. Defaults to False.
        figsize (tuple, optional): Plot size. Defaults to (8, 8).
    
    Returns:
    --------
        [type]: [description]
        [type]: [description]
    """
    sift_detector, sift_trainKP, sift_trainDesc, flann = train_sift(
        train_image, draw=draw
    )
    sift_goodMatch, sift_queryKP, _ = query_sift(
        query_image.copy(), sift_detector, sift_trainDesc, flann
    )
    _, sift_queryBorder_board = compute_detector_corners(
        train_image, sift_goodMatch, sift_trainKP, sift_queryKP, force_true=False
    )
    return sift_queryBorder_board, len(sift_goodMatch)
