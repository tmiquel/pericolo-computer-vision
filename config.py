# RANSAC
NUM_ITERATION = 2000                                            # Number of RANSAC iterations to perform
TRAINING_THRESHOLD = 5                                          # Angle Threshold for RANSAC in degrees
COMPLIANCE_THRESHOLD = 10                                       # Angle Threshold for edglets removal in degrees

# CONFIDENCE INDEX
MIN_RANSAC_EDGELETS = 2000                                      # Minimum number of edgelets to consider RANSAC viable
FIRST_RANSAC_SCORE = 15000                                      # Minimum score to consider for first RANSAC
SECOND_RANSAC_SCORE = 10000                                     # Minimum score to consider for second RANSAC
RANSAC_CONDITIONS_TO_VALID = 1                                  # Float in [0-1] describing the ratio of condition to meet to perform reprojection
IOU_THRESHOLD = 0.85                                            # IoU Threshold between Aruco marker detection and unenclosed SIFT detection
IR_THRESHOLD = 0.99                                             # Inclusion ratio threshold between Aruco marker and enclosed SIFT detection

# ARUCO 
MARKER_ID = 0                                                   # Aruco ID of the marker (Not Used but useful to have in the future)
ENCLOSED_MARKER = True                                          # Whether or not the marker is enclosed (Not Used but useful to have in the future)
MARKER_SIZE_IN_MM = 120                                         # Marker size (on mililmeters) when printed

# HOMOGRAPHY
CLIP = True                                                     # Whether or not to clip the image if the warping is too big
CLIP_FACTOR = 4                                                 # By which factor of the initial size should the clip be done
ALLOW_WARPING_INTERUPT = True                                   # In case the resulting image (clip or not) is too big, whether or not to interupt the process
MAX_SIZE_ALLOWED = (15000, 15000)                               # Maximum sized allowed to not interupt the process

# SIFT DETECTION
ENCLOSED_MARKER_FILE = "single_aruco_marker_enclosed_id0.png"   # FILE TO PUT IN markers FOLDER
UNENCLOSED_MARKER_FILE = "single_aruco_marker_id0.png"          # FILE TO PUT IN markers FOLDER
MIN_MATCH_COUNT = 15                                            # Minimum matches to find to perform SIFT detection
ENFORCE_SIFT_DETECTION = True                                   # Force the algorithm to find the marker via SIFT to confirm its position
MODERATE_MATCH_COUNT = 30                                       # Matches to find for MODERATE confidence
STRONG_MATCH_COUNT = 50                                         # Matches to find for STRONG confidence
CONFIDENCE_LEVEL = 0                                            # -1 No confidence check, 0 Minimum confidence check, 1 Moderate confidence check, 2 Strong confidence check

# SCALEBAR
PADDING_H_RATIO = 0.05                                          # Percentage of image height to define padding height
PADDING_W_RATIO = 0.05                                          # Percentage of image width to define padding width
BOX_POSITION = "upper_right"                                    # Scale box position
BOX_SIZE_H_RATIO = 0.1                                          # Percentage of image height to define box height
BOX_SIZE_W_RATIO = 0.25                                         # Percentage of image width to define box height
BOX_COLOR = (255, 255, 255)                                     # Background box color
BOX_THICKNESS = -1                                              # Background box thickness (set it to negative to fill the box)
BOX_LINETYPE = 8                                                # 
INBOX_TITLE_TEXT = "Echelle"                                    # Box title word
INBOX_COLOR = (0, 0, 0)                                         # Box text color
INBOX_PADDING_RATIO = 0.05                                      # Percentage of box height to define inner padding height (1 = 2*INBOX_PADDING_RATIO + INBOX_TITLE_RATIO + INBOX_LINE_RATIO + INBOX_VALUE_RATIO)
INBOX_TITLE_RATIO = 0.3                                         # Percentage of box height to define inner title zone height (1 = 2*INBOX_PADDING_RATIO + INBOX_TITLE_RATIO + INBOX_LINE_RATIO + INBOX_VALUE_RATIO)
INBOX_LINE_RATIO = 0.3                                          # Percentage of box height to define inner scalebar zone height (1 = 2*INBOX_PADDING_RATIO + INBOX_TITLE_RATIO + INBOX_LINE_RATIO + INBOX_VALUE_RATIO)
INBOX_VALUE_RATIO = 0.3                                         # Percentage of box height to define inner scale text zone height (1 = 2*INBOX_PADDING_RATIO + INBOX_TITLE_RATIO + INBOX_LINE_RATIO + INBOX_VALUE_RATIO)
INBOX_W_PAD_RATIO = 0.05                                        # Percentage of box width to define inner padding width



