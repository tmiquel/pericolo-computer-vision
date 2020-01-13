import os
import logging
import numpy as np
import cv2
from skimage import io, color

from libs.aruco_detection import detect_markers
from libs.edgelets_processing import (
    generate_edges,
    vector_representation,
    to_homogenous_geometry,
)
from libs.sift_detection import detect_object_via_sift
from libs.vanishing_point_sampling import (
    ransac,
    remove_compliant_edgelets,
    compute_score,
)
from libs.warping import warp_image, infer_warp_shape, marker_position_in_projection

from utils.folders import MARKER_FOLDER
from utils.metrics import inclusion_ratio, IoU
from utils.scalebar import draw_scalebar
from utils.utility_functions import order_points, polygon_to_mask

from config import (
    MIN_MATCH_COUNT,
    COMPLIANCE_THRESHOLD,
    TRAINING_THRESHOLD,
    NUM_ITERATION,
    CLIP,
    CLIP_FACTOR,
    ALLOW_WARPING_INTERUPT,
    MAX_SIZE_ALLOWED,
    FIRST_RANSAC_SCORE,
    SECOND_RANSAC_SCORE,
    MIN_RANSAC_EDGELETS,
    ENCLOSED_MARKER_FILE,
    UNENCLOSED_MARKER_FILE,
    MIN_MATCH_COUNT,
    MODERATE_MATCH_COUNT,
    STRONG_MATCH_COUNT,
    CONFIDENCE_LEVEL,
    IOU_THRESHOLD,
    IR_THRESHOLD,
    ENFORCE_SIFT_DETECTION,
    RANSAC_CONDITIONS_TO_VALID,
    MARKER_SIZE_IN_MM,
)


class ImageWarper:
    """ImageWarper class is used to warp images both by using Aruco Marker detection or Vanishing Point reprojection
    
    """

    def __init__(
        self,
        image,
        num_ransac_iter=NUM_ITERATION,
        training_threshold=TRAINING_THRESHOLD,
        compliance_threshold=COMPLIANCE_THRESHOLD,
        clip=CLIP,
        clip_factor=CLIP_FACTOR,
        allow_warping_interupt=ALLOW_WARPING_INTERUPT,
        max_size_allowed=MAX_SIZE_ALLOWED,
        first_ransac_score=FIRST_RANSAC_SCORE,
        second_ransac_score=SECOND_RANSAC_SCORE,
        min_ransac_edgelets=MIN_RANSAC_EDGELETS,
        min_match_count=MIN_MATCH_COUNT,
        moderate_match_count=MODERATE_MATCH_COUNT,
        strong_match_count=STRONG_MATCH_COUNT,
        enclosed_marker_file=ENCLOSED_MARKER_FILE,
        unenclosed_marker_file=UNENCLOSED_MARKER_FILE,
        iou_threshold=IOU_THRESHOLD,
        ir_threshold=IR_THRESHOLD,
        enforce_sift_detection=ENFORCE_SIFT_DETECTION,
        confidence_level=CONFIDENCE_LEVEL,
        ransac_conditions_to_valid=RANSAC_CONDITIONS_TO_VALID,
        marker_size_in_mm=MARKER_SIZE_IN_MM,
    ):
        # Image related variables
        self.image_path = image
        self.image = io.imread(image)
        self.mode = "rgb"

        # Warp logs
        self.warp_log_dict = {}

        # Config variables
        self._num_ransac_iter = num_ransac_iter
        self._training_threshold = training_threshold
        self._compliance_threshold = compliance_threshold
        self._clip = clip
        self._clip_factor = clip_factor
        self._allow_warping_interupt = allow_warping_interupt
        self._max_size_allowed = max_size_allowed
        self._first_ransac_score = first_ransac_score
        self._second_ransac_score = second_ransac_score
        self._min_ransac_edgelets = min_ransac_edgelets
        self._enclosed_marker_file = enclosed_marker_file
        self._unenclosed_marker_file = unenclosed_marker_file
        self._min_match_count = min_match_count
        self._moderate_match_count = moderate_match_count
        self._strong_match_count = strong_match_count
        self._iou_threshold = iou_threshold
        self._ir_threshold = ir_threshold
        self._enforce_sift_detection = enforce_sift_detection
        self._confidence_level = confidence_level
        self._ransac_conditions_to_valid = ransac_conditions_to_valid
        self._marker_size_in_mm = marker_size_in_mm

    # Color Conversion
    def to_rgb(self):
        """Converts initial image to RGB mode
        
        Returns:
            np.array: RGB Initial Image
        """
        if self.mode == "bgr":
            return cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2BGR)
        elif self.mode == "rgb":
            return image

    def to_bgr(self):
        """Converts initial image to BGR mode
        
        Returns:
            np.array: BGR Initial Image
        """
        if self.mode == "rgb":
            return cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
        elif self.mode == "bgr":
            return image

    def to_gray(self):
        """Converts initial image to GRAYSCALE mode
        
        Returns:
            np.array: GRAYSCALE Initial Image
        """
        if self.mode == "rgb":
            return cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2GRAY)
        elif self.mode == "bgr":
            return cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)

    # Warp Logs
    @property
    def logs(self):
        """Property accessor for warping logs

        Note:
        -----
        Logging is set at INFO level except when the algorithm fails, in that case the level is set to WARNING
        
        Returns:
            dict: Warping logs
        """
        return self.warp_log_dict

    def reset_logs(self):
        """Reset the logs dictionnary
        
        Returns:
            ImageWaeper: returns itself
        """
        self.warp_log_dict = {}
        return self

    def set_logs(self, key, value):
        """Set a new entry in log dictionnary
        
        Args:
            key (str): condition name
            value (bool): Whether a condition is validated or not
        """
        self.warp_log_dict[key] = value

    # Condition Checking
    def _check_edgelets_number_condition(self, locations):
        """Checks RANSAC conditions over minimum edgelets number
        
        Args:
            locations (list): list of edgelets locations 
        
        Returns:
            bool: Whether or not the condition is validated
        """
        if len(locations) >= self._min_ransac_edgelets:
            logging.info(f"CHECK - RANSAC - {self.image_path}")
            logging.info(f"CHECK - RANSAC - EDGELETS NUMBER")
            logging.info(f"CHECK - RANSAC - EDGELETS NUMBER - SUCCESS")
            logging.info(
                f"CHECK - RANSAC - EDGELETS NUMBER - {len(locations)} edgelets found"
            )
            self.set_logs(f"RANSAC - EDGELETS NUMBER", True)
            return True
        else:
            logging.warning(f"CHECK - RANSAC - {self.image_path}")
            logging.warning(f"CHECK - RANSAC - EDGELETS NUMBER")
            logging.warning(f"CHECK - RANSAC - EDGELETS NUMBER - FAILURE")
            logging.warning(
                f"CHECK - RANSAC - EDGELETS NUMBER - {len(locations)} edgelets found"
            )
            self.set_logs(f"RANSAC - EDGELETS NUMBER", False)
            return False

    def _check_ransac_score(
        self, vanishing_point, locations, directions, strengths, target_score
    ):
        """Checks RANSAC conditions over minimum score
        
        Args:
            vanishing_point (np.array): Vanishing point to evaluate score for
            locations (np.array): list of edgelets locations 
            directions (np.array): list of edgelets directions 
            strengths (np.array): list of edgelets strengths 
            target_score (float): Score threshold
        
        Returns:
            bool: Whether or not the condition is validated
        """
        score = compute_score(
            vanishing_point,
            locations,
            directions,
            strengths,
            threshold_inlier=self._training_threshold,
        )
        if score.sum() >= target_score:
            logging.info(f"CHECK - RANSAC - {self.image_path}")
            logging.info(f"CHECK - RANSAC - MINIMUM SCORE")
            logging.info(f"CHECK - RANSAC - MINIMUM SCORE - SUCCESS")
            logging.info(
                f"CHECK - RANSAC - MINIMUM SCORE - {score.sum()}/{target_score}"
            )
            self.set_logs(f"RANSAC - MINIMUM SCORE", True)
            return True
        else:
            logging.warning(f"CHECK - RANSAC - {self.image_path}")
            logging.warning(f"CHECK - RANSAC - MINIMUM SCORE")
            logging.warning(f"CHECK - RANSAC - MINIMUM SCORE - FAILURE")
            logging.warning(
                f"CHECK - RANSAC - MINIMUM SCORE - {score.sum()}/{target_score}"
            )
            self.set_logs(f"RANSAC - MINIMUM SCORE", False)
            return False

    def _check_match(self, n_matches, name, threshold):
        """Checks Aruco detection SIFT matches conditions
        
        Args:
            n_matches (int): number of matches found
            name (str): Marker type evaluated
            threshold (int): matches threshold
        
        Returns:
            bool: Whether or not the condition is validated or not
        """
        if n_matches >= threshold:
            logging.info(f"CHECK - MARKER - {self.image_path}")
            logging.info(f"CHECK - MARKER - {name} SIFT DETECTION")
            logging.info(f"CHECK - MARKER - {name} SIFT DETECTION - SUCCESS")
            self.set_logs(f"{name} SIFT DETECTION", True)
            return True
        else:
            logging.warning(f"CHECK - MARKER - {self.image_path}")
            logging.warning(f"CHECK - MARKER - {name} SIFT DETECTION")
            logging.warning(f"CHECK - MARKER - {name} SIFT DETECTION - FAILURE")
            self.set_logs(f"{name} SIFT DETECTION", False)
            return False

    def _check_matches(self, n_matches, name):
        """Checks Aruco detection SIFT matches conditions for each confidence level
        
        Args:
            n_matches (int): number of matches found
            name (str): Marker type evaluated
        
        Returns:
            bool: Whether or not the condition is validated or not
        """
        min_check = self._check_match(
            n_matches, name + " MINIMUM", self._min_match_count
        )
        moderate_check = self._check_match(
            n_matches, name + " MODERATE", self._moderate_match_count
        )
        strong_check = self._check_match(
            n_matches, name + " STRONG", self._strong_match_count
        )
        confidence_check = [min_check, moderate_check, strong_check]
        if self._confidence_level > -1:
            return confidence_check[self._confidence_level]
        else:
            return True

    def _check_metric(self, value, name, threshold):
        """Check Aruco detection metrics conditions
        
        Args:
            value (float): Current value of the metric
            name (str): Metric name to evaluate
            threshold (float): Threshold for the metric
        
        Returns:
            bool: Whether or not the condition is validated
        """
        if value >= threshold:
            logging.info(f"CHECK - MARKER - {self.image_path}")
            logging.info(f"CHECK - MARKER - SIFT {name}")
            logging.info(f"CHECK - MARKER - SIFT {name} - SUCCESS")
            logging.info(
                f"CHECK - MARKER - SIFT {name} - FOUND {value} /  THRESH {threshold}"
            )
            self.set_logs(f"MARKER - SIFT {name}", True)
            return True
        else:
            logging.warning(f"CHECK - MARKER - {self.image_path}")
            logging.warning(f"CHECK - MARKER - SIFT {name}")
            logging.warning(
                f"CHECK - MARKER - SIFT {name} - FOUND {value} /  THRESH {threshold}"
            )
            self.set_logs(f"MARKER - SIFT {name}", False)
            return False

    def _check_aruco(self, aruco_corners_ordered):
        """Checks if Aruco detection algorithm has succeeded to find at least one Aruco marker
        
        Args:
            aruco_corners_ordered (list): list of resulting Aruco detection
        
        Returns:
            bool: Whether or not the condition is validated
        """
        if len(aruco_corners_ordered) == 0:
            logging.warning(f"CHECK - MARKER - {self.image_path}")
            logging.warning(f"CHECK - MARKER - ARUCO")
            logging.warning(f"CHECK - MARKER - ARUCO - FAILURE")
            self.set_logs(f"MARKER - ARUCO", False)
            return False
        logging.info(f"CHECK - MARKER - {self.image_path}")
        logging.info(f"CHECK - MARKER - ARUCO")
        logging.info(f"CHECK - MARKER - ARUCO - SUCCESS")
        self.set_logs(f"MARKER - ARUCO", True)
        return True

    # Hidden Warping Method
    def _vanishing_point_from_aruco(self, corners_ordered):
        """Hidden method for Aruco marker warping
        
        Args:
            corners_ordered (list): list of ordered Aruco corners coordinates
        
        Returns:
            np.array: Horizontal vanishing point
            np.array: Vertical vanishing point
        """
        horizontal_lines = [
            [corners_ordered[0][0], corners_ordered[0][1]],
            [corners_ordered[0][3], corners_ordered[0][2]],
        ]
        vertical_lines = [
            [corners_ordered[0][3], corners_ordered[0][0]],
            [corners_ordered[0][2], corners_ordered[0][1]],
        ]
        vertical_edgelets = vector_representation(vertical_lines)
        horizontal_edgelets = vector_representation(horizontal_lines)
        vertical_hlines = to_homogenous_geometry(*vertical_edgelets)
        horizontal_hlines = to_homogenous_geometry(*horizontal_edgelets)
        vp_vert = np.cross(vertical_hlines[0], vertical_hlines[1])
        vp_hor = np.cross(horizontal_hlines[0], horizontal_hlines[1])
        return vp_hor, vp_vert

    def _ransac_confirmation(self, conditions):
        """Checks every condition to meet to consider RANSAC successful
        
        Args:
            conditions (list): list of RANSAC condition results
        
        Returns:
            bool: Whether or not the RANSAC should be validated
        """
        cond_ratio = np.array(conditions).astype(int).sum() / len(conditions)
        if cond_ratio >= self._ransac_conditions_to_valid:
            logging.info(f"CHECK - MARKER - {self.image_path}")
            logging.info(f"CHECK - MARKER - RANSAC VALIDATION")
            logging.info(f"CHECK - MARKER - RANSAC VALIDATION - SUCCESS")
            logging.info(
                f"CHECK - MARKER - RANSAC VALIDATION - {cond_ratio} on {self._ransac_conditions_to_valid} "
            )
            self.set_logs(f"RANSAC VALIDATION", True)
            return True
        else:
            logging.warning(f"CHECK - MARKER - {self.image_path}")
            logging.warning(f"CHECK - MARKER - RANSAC VALIDATION")
            logging.warning(f"CHECK - MARKER - RANSAC VALIDATION - FAILURE")
            logging.warning(
                f"CHECK - MARKER - RANSAC VALIDATION - {cond_ratio} on {self._ransac_conditions_to_valid} "
            )
            self.set_logs(f"RANSAC VALIDATION", False)
            return False

    def _sift_confirmation(self, aruco_corners_ordered):
        """Checks out every condition to meet to consider Aruco Detection successful
        
        Args:
            aruco_corners_ordered (list): list of ordered Aruco corners coordinates
        
        Returns:
            bool: Whether or not Aruco detection should be used
        """
        # Open Enclosed/Unenclosed markers images
        train_img_enclosed = cv2.imread(
            os.path.join(MARKER_FOLDER, self._enclosed_marker_file), 0
        )
        train_img_unenclosed = cv2.imread(
            os.path.join(MARKER_FOLDER, self._unenclosed_marker_file), 0
        )
        # SIFT Detection
        enclosed_borders, enclosed_n_matches = detect_object_via_sift(
            self.to_bgr(), train_img_enclosed
        )
        unenclosed_borders, unenclosed_n_matches = detect_object_via_sift(
            self.to_bgr(), train_img_unenclosed
        )

        if self._enforce_sift_detection and not (
            self._check_matches(enclosed_n_matches, "ENCLOSED")
            & self._check_matches(unenclosed_n_matches, "UNENCLOSED")
        ):
            return False

        # Generate Masks
        enclosed_mask = polygon_to_mask(
            self.image.shape[1], self.image.shape[0], enclosed_borders
        )
        unenclosed_mask = polygon_to_mask(
            self.image.shape[1], self.image.shape[0], unenclosed_borders
        )
        aruco_mask = polygon_to_mask(
            self.image.shape[1],
            self.image.shape[0],
            (
                aruco_corners_ordered[0]
                if len(aruco_corners_ordered)
                else aruco_corners_ordered
            ),
        )
        if self._confidence_level > -1:
            iou_condition = self._check_metric(
                value=IoU(aruco_mask, unenclosed_mask),
                name="IOU",
                threshold=self._iou_threshold,
            )
            ir_condition = self._check_metric(
                value=inclusion_ratio(aruco_mask, enclosed_mask),
                name="IR",
                threshold=self._ir_threshold,
            )

            return iou_condition & ir_condition
        else:
            return True

    def _warp_without_marker(self):
        """Algorithm to find vanishing points in a scene without marker
        
        Returns:
            np.array: First vanishing point
            np.array: Second vanishing point
        """
        # First generate the lines
        _, _, lines = generate_edges(self.image)
        # Convert the lines to edgelets representation
        locations, directions, strengths = vector_representation(lines)
        # First condition to check - Number of edgelets to perform first RANSAC
        first_edgelets_condition = self._check_edgelets_number_condition(locations)
        # Perform RANSAC to find the first vanishing point
        best_vp = ransac(
            locations=locations,
            directions=directions,
            strengths=strengths,
            num_ransac_iter=self._num_ransac_iter,
            threshold_inlier=self._training_threshold,
        )
        # Second condition to check - Score of first RANSAC
        first_score_condition = self._check_ransac_score(
            best_vp, locations, directions, strengths, self._first_ransac_score
        )
        # Filter to remove compliant edgelets
        (
            remaining_location,
            remaining_directions,
            remaining_strengths,
        ) = remove_compliant_edgelets(
            vanishing_point=best_vp,
            locations=locations,
            directions=directions,
            strengths=strengths,
            threshold_inlier=self._compliance_threshold,
        )
        # Third condition to check - Number of edgelets to perform second RANSAC
        second_edgelets_condition = self._check_edgelets_number_condition(
            remaining_location
        )
        # Perform RANSAC to find the first vanishing point
        second_best_vp = ransac(
            locations=remaining_location,
            directions=remaining_directions,
            strengths=remaining_strengths,
            num_ransac_iter=self._num_ransac_iter,
            threshold_inlier=self._training_threshold,
        )
        # Fourth condition to check - Score of second RANSAC
        second_score_condition = self._check_ransac_score(
            second_best_vp,
            remaining_location,
            remaining_directions,
            remaining_strengths,
            self._second_ransac_score,
        )
        # Check whether or not to perform reprojection
        conditions = [
            first_edgelets_condition,
            first_score_condition,
            second_edgelets_condition,
            second_score_condition,
        ]

        if not (self._ransac_confirmation(conditions)):
            return False

        return best_vp, second_best_vp

    def _warp_with_marker(self):
        """Algorithm to find vanishing points in a scene with an Aruco marker
        
        Returns:
            np.array: Horizontal vanishing point
            np.array: Vertical vanishing point
        """
        # Compute Corner detection & order points
        corners, ids = detect_markers(self.to_gray())
        aruco_corners_ordered = [order_points(marker[0]) for marker in corners]

        if not (self._check_aruco(aruco_corners_ordered)):
            return False

        if self._enforce_sift_detection and not (
            self._sift_confirmation(aruco_corners_ordered)
        ):
            return False

        self.marker_corners = aruco_corners_ordered[0]

        vp_hor, vp_vert = self._vanishing_point_from_aruco(aruco_corners_ordered)

        return vp_hor, vp_vert

    def _check_size(self, vp1, vp2):
        """Checks if the output size conditions are met
        
        Args:
            vp1 (np.array): First vanishing point
            vp2 (np.array): Second vanishing point
        
        Returns:
            bool: Whether or not the condition is validated
        """
        clipped_shape = infer_warp_shape(
            self.image, vp1, vp2, clip=self._clip, clip_factor=self._clip_factor,
        )
        unclipped_shape = infer_warp_shape(
            self.image, vp1, vp2, clip=False, clip_factor=self._clip_factor,
        )
        if (
            (unclipped_shape[0] > self._max_size_allowed[0])
            and (unclipped_shape[1] > self._max_size_allowed[1])
        ) and (
            (clipped_shape[0] > self._max_size_allowed[0])
            and (clipped_shape[1] > self._max_size_allowed[1])
        ):
            logging.warning(f"CHECK - WARPING - {self.image_path}")
            logging.warning(f"CHECK - WARPING - FAILURE")
            logging.warning(
                f"CHECK - WARPING - Huge Image Size, warping might take a very long time"
            )
            if self._allow_warping_interupt:
                logging.warning(f"CHECK - WARPING - INTERUPTION")
                self.set_logs(f"WARPING - INTERUPTION", True)
                return False
        self.set_logs(f"WARPING - INTERUPTION", False)
        return None

    def _warp_image(self, vp1, vp2):
        """Given a set of 2 vanishing point, warp the initial image
        
        Args:
            vp1 (np.array): First vanishing point
            vp2 (np.array): Second vanishing point
        
        Returns:
            np.array: Warped image
        """
        warped_img = warp_image(
            self.image, vp1, vp2, clip=self._clip, clip_factor=self._clip_factor
        )
        warped_img = (warped_img * 255).astype(np.uint8)
        return warped_img

    # Warp functions
    def warp_with_marker(self):
        """Proceed to the warping with Aruco Marker
        
        Returns:
            np.array: Warped image with Aruco Marker
        """
        result = self._warp_with_marker()
        if result is False:
            return None
        vp1, vp2 = result
        image_warped = self._warp_image(vp1, vp2)
        self.marker_corners_reproj = marker_position_in_projection(
            self.image,
            vp1,
            vp2,
            aruco_borders=self.marker_corners,
            clip=self._clip,
            clip_factor=self._clip_factor,
        )
        image_warped = draw_scalebar(
            image=image_warped, marker=self.marker_corners_reproj
        )
        return image_warped

    def warp_without_marker(self):
        """Proceed to the warping with vanishing point methodology
        
        Returns:
            np.array: Warped image
        """
        result = self._warp_without_marker()
        if result is False:
            return None
        vp1, vp2 = result
        return self._warp_image(vp1, vp2)

    def warp(self):
        """Proceed to the most suitable warping algorithm
        
        Returns:
            np.array: Warped image
        """
        warp_result = self.warp_with_marker()
        if warp_result is None:
            warp_result = self.warp_without_marker()
        return warp_result
