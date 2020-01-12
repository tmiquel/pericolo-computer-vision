import numpy as np
import logging
from libs.edgelets_processing import to_homogenous_geometry

def compute_score(vanishing_point, locations, directions, strengths, threshold_inlier=5):
    """Compute score for a model definied by a `vanishing_point` description and a `threshold_inlier`
    
    Args:
    -----
        vanishing_point (np.array): Vanishing Point description
        locations (np.array): Middle points of the edgelets
        directions (np.array): Normalized vectors describing orientation of edgelets
        strengths (np.array): Scalar norm (L2 norm) of edgelets
        threshold_inlier (int, optional): Angle (in degrees) to consider edgelets for scoring. Defaults to 5.
    
    Returns:
    --------
        np.array: Scores for each edgelets considering the vanishing point and the `threshold_inlier`
    
    Notes:
    ------
    If the absolute angle between the edglet direction and the `edglet_middle_point -> vanishing_point` line direction
    lies under the `threshold_inlier` then the edgelet score is equal to its strength else it is zero
    """
    # Represent Vanishing Point (Intersection Point) in (X, Y, 1) Initial Plane
    vp_img_plane = vanishing_point[:2] / vanishing_point[2]
    
    # Generate vector from edglets middle points to vanishing point
    # For each edgelet compute the angle between the direction vector
    # and the vector defined as edgelet location minus vanishing point
    estimated_directions = locations - vp_img_plane

    # Compute the Angle between edglets directions (directions) and
    # previously computed estimated direction of the edglet_middle_point -> vanishing_point line
    dot_prod = np.sum(estimated_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * np.linalg.norm(estimated_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5
    threshold_inlier = 5
    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))
    
    # Compute scores
    # Consider for scoring only the edglets within a range of +/- threshold_inlier
    theta_thresh = threshold_inlier * np.pi / 180
    scores = (theta < theta_thresh) * strengths
    return scores


def ransac(locations, directions, strengths, num_ransac_iter=2000, threshold_inlier=5):
    """RANSAC algorithm to sample best vanishing point possible
    
    Args:
    -----
        locations (np.array): Middle points of the edgelets
        directions (np.array): Normalized vectors describing orientation of edgelets
        strengths (np.array): Scalar norm (L2 norm) of edgelets
        num_ransac_iter (int, optional): Number of iteration to perform. Defaults to 2000.
        threshold_inlier (int, optional): Angle (in degrees) to consider edgelets for scoring. Defaults to 5.
    
    Returns:
    --------
        np.array: Vanishing Point description
    
    """
     # Convert to homogenous geometry
    lines = to_homogenous_geometry(locations, directions, strengths)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    # Select the top 20 percentile
    first_index_space = arg_sort[:num_pts // 5] 
    # Select the top 50 percentile
    second_index_space = arg_sort[:num_pts // 2] 

    best_model = None
    best_scores = np.zeros(num_pts)

    # Sampling process
    for ransac_iter in range(num_ransac_iter):
        # Sample a line index from the top 20 percentile
        ind1 = np.random.choice(first_index_space) 
        # Sample a line index from the top 50 percentile
        ind2 = np.random.choice(second_index_space) 

        l1 = lines[ind1]
        l2 = lines[ind2]

        # In Homogenous geometry the cross-product (vectot product)
        # represents the intersection points between two vectors
        current_model = np.cross(l1, l2)
        
        # In case of degeneracy
        # e.g colinearity between l1 and l2
        # e.g sampling where l1 and l2 are the same line
        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # Force resampling
            continue

        current_scores = compute_score(
            vanishing_point=current_model,
            locations=locations,
            directions=directions,
            strengths=strengths,
            threshold_inlier=threshold_inlier
        )
        
        # In case the new model is better, replace the old one
        # The score is equal to the sum of edgelet's strength falling in the threshold range
        if current_scores.sum() > best_scores.sum():
            best_model = current_model
            best_scores = current_scores
            logging.info(f"Current best model has {current_scores.sum()} votes at iteration {ransac_iter}")
    return best_model


def remove_compliant_edgelets(vanishing_point, locations, directions, strengths, threshold_inlier=10):
    """Remove edgelets that falls under a vanishing point range of explainability. 
    Filter only the edgelets that do not fit the vanishing point by a threshold (in degree) angle
    
    Args:
    -----
        vanishing_point (np.array): Vanishing Point description
        locations (np.array): Middle points of the edgelets
        directions (np.array): Normalized vectors describing orientation of edgelets
        strengths (np.array): Scalar norm (L2 norm) of edgelets
        threshold_inlier (int, optional): Angle (in degrees) to consider edgelets for scoring. Defaults to 10.
    
    Returns:
    --------
        tuple: tuple representation of edgelets that do not lie under the vanishing point range
    """
    # Compute scores for each edgelets
    scores = compute_score(
        vanishing_point=vanishing_point,
        locations=locations,
        directions=directions,
        strengths=strengths,
        threshold_inlier=threshold_inlier
    )
    
    # Define each edgelet falling under the threshold range as 1 
    compliant_indices = scores > 0
    
    # Remove each edgelet falling under the threshold range
    locations = locations[~compliant_indices]
    directions = directions[~compliant_indices]
    strengths = strengths[~compliant_indices]
    return (locations, directions, strengths)