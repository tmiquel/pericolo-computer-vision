import numpy as np
from cv2 import aruco

# Generate a dictionnary of 1000 6X6 Aruco markers
SAMPLE_ARUCO_DICT = aruco.DICT_6X6_1000

def generate_enclosed_marker(aruco_dict, marker_id, marker_size, square_factor=4):
    aruco_marker = aruco.drawMarker(aruco_dict,marker_id, marker_size)
    offset = marker_size // square_factor
    h_stack = 255*np.ones((marker_size, offset))
    v_stack = 255*np.ones(( offset, marker_size+2*offset))
    aruco_marker = np.hstack([h_stack, aruco_marker, h_stack])
    aruco_marker = np.vstack([v_stack, aruco_marker, v_stack])
    aruco_marker[0:offset, 0:offset] = 0
    aruco_marker[marker_size+offset:marker_size+2*offset, 0:offset] = 0
    aruco_marker[0:offset, marker_size+offset:marker_size+2*offset] = 0
    aruco_marker[marker_size+offset:marker_size+2*offset, marker_size+offset:marker_size+2*offset] = 0
    aruco_marker[marker_size+offset:marker_size+2*offset, 0:offset] = 0
    return aruco_marker