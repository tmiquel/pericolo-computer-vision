import numpy as np
from skimage import io, feature, color, transform

def generate_edges(image, sigma=3):
    """Generate the edges via a Canny Edge detection and compute potential lines in the image
    via a Probabilistic Hough Transform
    
    Args:
    -----
        image (np.array): Image as `RGB` numpy array
        sigma (int, optional): Canny Edge detection Gaussian width parameter. Defaults to 3.
    
    Returns:
    --------
        np.array: Image as `GRAYSCALE` numpy array
        np.array: Canny Edge Image Result as numpy array
        list: List of lines, each line is represented as a 2 item list corresponding to the extremum coordinates
    
    """
    # Convert RGB to GrayScale
    gray_img = color.rgb2gray(image)
    # Edges Coarse detection
    edges = feature.canny(gray_img, sigma=sigma) 
    # Generate potential lines from edges
    lines = transform.probabilistic_hough_line(edges, line_length=3, line_gap=2) 
    return gray_img, edges, lines

def vector_representation(lines):
    """Represents lines resulting from Hough Transform to edgelets
    
    Args:
    -----
        lines (list): List of lines, each line is represented as a 2 item list corresponding to the extremum coordinates
    
    Returns:
    --------
        tuple: tuple representation of edgelets

    Notes:
    ------
    Edgelets, as described in "Auto-Rectification of User Photo" by Krishnendu Chaudhury, Stephen DiVerdi, Sergey Ioffe, 
    are (small) lines representations defined as follow
        locations: Middle points of the edgelets
        directions: Normalized vectors describing orientation of edgelets
        strengths: Scalar norm (L2 norm) of edgelets

    Bibliography:
    -------------
    Publication: https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/42532.pdf
    """
    locations = []
    directions = []
    strengths = []
    
    # For each detected line of 2 points
    for a,b in lines:
        a,b = np.array(a), np.array(b)
        # generate the location as the middle point
        locations.append((a + b) / 2) 
        # compute the direction vector
        directions.append(b - a) 
        # compute the strength as the norm
        strengths.append(np.linalg.norm(b - a)) 
    
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)
    
    # normalize directions
    directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis] 
    
    return (locations, directions, strengths)

def to_homogenous_geometry(locations, directions, strengths):
    """Convert edgelets in standard image plane geometry to Homogenous geometry by generating
    third dimension
    
    Args:
    -----
        locations (np.array): Middle points of the edgelets
        directions (np.array): Normalized vectors describing orientation of edgelets
        strengths (np.array): Scalar norm (L2 norm) of edgelets
    
    Returns:
    --------
        np.array: Representation of edgelets in homogenous geometry
    
    """
    
    normals = np.zeros_like(directions)
    # Compute the edglets normal vector (a, b) -> (b, -a)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    # Create the 3rd dimension representing the same point
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1) 
    return lines