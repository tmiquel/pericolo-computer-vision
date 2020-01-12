from cv2 import aruco

ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)

def detect_markers(gray_img, aruco_dict=ARUCO_DICT):
    """Detect the position and id of Aruco markers in an image
    
    Args:
    -----
        gray_img (np.array): GRAYSCALE Image
        aruco_dict ([type]): Aruco dictionnary describing the markers
    
    Returns:
    --------
        list: List of detected markers' corners
        list: List of ids of the detected markers
    """
    parameters =  aruco.DetectorParameters_create()
    # We use the detectMarkers from aruco package in OpenCv to generate markers position and ids
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)
    _ = rejectedImgPoints
    return corners, ids

def draw_markers(image, corners, ids):
    """Draw the detected Aruco markers' border on image
    
    Args:
        image (np.array): Image
        corners (list): List of detected markers' corners
        ids (list): List of ids of the detected markers
    
    Returns:
        np.array: Image with detected markers' border
    """
    # The following lines of codes are for display purpose only
    frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
    return frame_markers