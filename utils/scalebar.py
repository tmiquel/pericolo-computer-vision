import cv2
import numpy as np
from config import (
    MARKER_SIZE_IN_MM,
    BOX_COLOR,
    BOX_LINETYPE,
    BOX_POSITION,
    BOX_SIZE_H_RATIO,
    BOX_SIZE_W_RATIO,
    BOX_THICKNESS,
    PADDING_H_RATIO,
    PADDING_W_RATIO,
    INBOX_COLOR,
    INBOX_LINE_RATIO,
    INBOX_PADDING_RATIO,
    INBOX_TITLE_TEXT,
    INBOX_TITLE_RATIO,
    INBOX_VALUE_RATIO,
    INBOX_W_PAD_RATIO,
)

METER_SUBDIVISION_DICT = {
    "e+00": " mm",
    "e+01": " cm",
    "e+02": " dm",
    "e+03": " m",
    "e+04": " dam",
    "e+05": " hm",
    "e+06": " km",
}


def replace_values_in_string(text, args_dict):
    """Replace text substring from a mapping dictionnary
    
    Args:
        text (str): string containing substring to replace
        args_dict (dict): dictionnary mapping containing replacements
    
    Returns:
        str: replaced string
    """
    for key in args_dict.keys():
        text = text.replace(key, str(args_dict[key]))
    return text


def get_text_size(value_str, inbox_value_px, line_br_pt, line_tl_pt):
    """Get the size, in pixels, and font size to use to draw text fitting the inside box size
    
    Args:
        value_str (str): Text to draw
        inbox_value_px (int): Maximum size allowed in pixel
        line_br_pt (np.array): Bottom-right corner of the scalebar
        line_tl_pt (np.array): Top-left corner of the scalebar
    
    Returns:
        int: font size value
        tuple: size, in pixels, of the text to be drawn
    """
    for i in range(16):
        resulting_size = cv2.getTextSize(
            value_str, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=i + 1, thickness=10
        )
        if (
            resulting_size[0][0] > abs(line_br_pt[1] - line_tl_pt[1])
            or resulting_size[0][1] > inbox_value_px
        ):
            break
    value_font_size = i
    resulting_size = cv2.getTextSize(
        value_str,
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=value_font_size,
        thickness=10,
    )
    return value_font_size, resulting_size


def compute_inbox_details(
    size_h_px,
    size_w_px,
    inbox_w_pad_ratio=INBOX_W_PAD_RATIO,
    inbox_padding_ratio=INBOX_PADDING_RATIO,
    inbox_title_ratio=INBOX_TITLE_RATIO,
    inbox_line_ratio=INBOX_LINE_RATIO,
    inbox_value_ratio=INBOX_VALUE_RATIO,
):
    """Compute inbox details
    
    Args:
        size_h_px (int): box height in pixel
        size_w_px (int): box width in pixel
        inbox_w_pad_ratio (float, optional): Inbox padding width ratio. Defaults to INBOX_W_PAD_RATIO.
        inbox_padding_ratio (float, optional): Inbox padding height ratio. Defaults to INBOX_PADDING_RATIO.
        inbox_title_ratio (float, optional): Inbox Title height ratio. Defaults to INBOX_TITLE_RATIO.
        inbox_line_ratio (float, optional): Inbox scalebar height ratio. Defaults to INBOX_LINE_RATIO.
        inbox_value_ratio (float, optional): Inbox scale text height ratio. Defaults to INBOX_VALUE_RATIO.
    
    Returns:
        [type]: [description]
    """
    inbox_w_pad_px = size_w_px * inbox_w_pad_ratio
    inbox_pad_px = size_h_px * inbox_padding_ratio
    inbox_title_px = size_h_px * inbox_title_ratio
    inbox_line_px = size_h_px * inbox_line_ratio
    inbox_line_thickness_px = inbox_line_px / 3
    inbox_value_px = size_h_px * inbox_value_ratio
    return (
        inbox_w_pad_px,
        inbox_pad_px,
        inbox_title_px,
        inbox_line_px,
        inbox_line_thickness_px,
        inbox_value_px,
    )


def compute_inbox_scalebar_corners(
    tl_pt,
    br_pt,
    inbox_pad_px,
    inbox_title_px,
    inbox_value_px,
    inbox_line_thickness_px,
    inbox_w_pad_px,
):
    """Generate the inbox scalebar coordinates
    
    Args:
        tl_pt (tuple): Top-left box coordinates
        br_pt (tuple): Bottom-right box coordinates
        inbox_pad_px (int): Inbox padding height  in pixel
        inbox_title_px (int): Inbox titlebox height in pixel
        inbox_value_px (int): Inbox scale text height in pixel
        inbox_line_thickness_px (int): Inbox scalebar thickness height in pixel
        inbox_w_pad_px (int): Inbox padding width in pixel
    
    Returns:
        np.array: Scalebar top-left coordinates
        np.array: Scalebar bottom-right coordinates
    """
    line_tl_pt = np.array(tl_pt) + np.array(
        (inbox_pad_px + inbox_title_px + inbox_line_thickness_px, inbox_w_pad_px)
    )
    line_br_pt = np.array(br_pt) - np.array(
        (inbox_pad_px + inbox_value_px + inbox_line_thickness_px, inbox_w_pad_px)
    )
    line_tl_pt = tuple(line_tl_pt.astype(int))
    line_br_pt = tuple(line_br_pt.astype(int))
    return line_tl_pt, line_br_pt


def compute_background_box_corners(
    pad_px, size_h_px, size_w_px, height, width, box_position
):
    """Compute the background box coordinates
    
    Args:
        pad_px (int): padding size
        size_h_px (int): box height in pixel
        size_w_px (int): box width in pixel
        height (int): image height
        width (int): image width
        box_position (str): box corner position
    
    Returns:
        np.array: Background box top-left coordinates
        np.array: Background box bottom-right coordinates
    """
    # Compute background box coordinates
    if box_position == "upper_right":
        tl_pt = (pad_px, width - pad_px - size_w_px)
        br_pt = (pad_px + size_h_px, width - pad_px)
    elif box_position == "upper_left":
        tl_pt = (pad_px, pad_px)
        br_pt = (pad_px + size_h_px, pad_px + size_w_px)
    elif box_position == "bottom_left":
        tl_pt = (height - pad_px - size_h_px, pad_px)
        br_pt = (HEIGHT - pad_px, pad_px + size_w_px)
    elif box_position == "bottom_right":
        tl_pt = (height - pad_px - size_h_px, width - pad_px - size_w_px)
        br_pt = (height - pad_px, width - pad_px)

    tl_pt, br_pt = (
        tuple(np.array(tl_pt).astype(int)),
        tuple(np.array(br_pt).astype(int)),
    )

    return tl_pt, br_pt


def draw_scalebar(
    image,
    marker,
    rescale_marker=True,
    box_position=BOX_POSITION,
    marker_size_in_mm=MARKER_SIZE_IN_MM,
    padding_h_ratio=PADDING_H_RATIO,
    padding_w_ratio=PADDING_W_RATIO,
    box_size_h_ratio=BOX_SIZE_H_RATIO,
    box_size_w_ratio=BOX_SIZE_W_RATIO,
    box_color=BOX_COLOR,
    box_thickness=BOX_THICKNESS,
    box_linetype=BOX_LINETYPE,
    inbox_color=INBOX_COLOR,
    inbox_line_ratio=INBOX_LINE_RATIO,
    inbox_padding_ratio=INBOX_PADDING_RATIO,
    inbox_title_text=INBOX_TITLE_TEXT,
    inbox_title_ratio=INBOX_TITLE_RATIO,
    inbox_value_ratio=INBOX_VALUE_RATIO,
    inbox_w_pad_ratio=INBOX_W_PAD_RATIO,
):
    # Create copy of image to avoid to compromise input array
    img = image.copy()
    # Compute marker size on image
    marker_h_size = abs(marker[0][0] - marker[1][0])
    marker_w_size = abs(marker[0][1] - marker[-1][1])

    # Marker Ratio correction
    if rescale_marker:
        if marker_h_size > marker_w_size:
            r = marker_h_size / marker_w_size
            new_shape = np.array(img.shape[:2]) * (r, 1)
            marker = marker * (1, r)
        else:
            r = marker_w_size / marker_h_size
            new_shape = np.array(img.shape[:2]) * (1, r)
            marker = marker * (r, 1)
        marker_h_size = abs(marker[0][0] - marker[1][0])
        marker_w_size = abs(marker[0][1] - marker[-1][1])
        img = cv2.resize(img, tuple(new_shape.astype(int))[::-1])

    # Compute scale of image
    height, width, _ = img.shape
    full_plan_height_mm = height * marker_size_in_mm / marker_h_size
    full_plan_width_mm = width * marker_size_in_mm / marker_w_size

    # Compute padding
    pad_h_px, pad_w_px = padding_h_ratio * height, padding_w_ratio * width
    pad_px = min(pad_h_px, pad_w_px)
    size_h_px, size_w_px = box_size_h_ratio * height, box_size_w_ratio * width

    # Compute Background box corners
    tl_pt, br_pt = compute_background_box_corners(
        pad_px=pad_px,
        size_h_px=size_h_px,
        size_w_px=size_w_px,
        height=height,
        width=width,
        box_position=box_position,
    )

    # Draw background box
    cv2.rectangle(
        img=img,
        pt1=tl_pt[::-1],
        pt2=br_pt[::-1],
        color=box_color,
        thickness=box_thickness,
        lineType=box_linetype,
        shift=0,
    )

    # Compute inbox details
    (
        inbox_w_pad_px,
        inbox_pad_px,
        inbox_title_px,
        inbox_line_px,
        inbox_line_thickness_px,
        inbox_value_px,
    ) = compute_inbox_details(size_h_px=size_h_px, size_w_px=size_w_px)

    # Compute inbox scalebar corners
    line_tl_pt, line_br_pt = compute_inbox_scalebar_corners(
        tl_pt,
        br_pt,
        inbox_pad_px,
        inbox_title_px,
        inbox_value_px,
        inbox_line_thickness_px,
        inbox_w_pad_px,
    )

    # Draw Scalebar
    cv2.rectangle(
        img=img,
        pt1=line_tl_pt[::-1],
        pt2=line_br_pt[::-1],
        color=inbox_color,
        thickness=box_thickness,
        lineType=box_linetype,
        shift=0,
    )

    # Compute black bar size
    black_bar_scale_mm = (line_br_pt[1] - line_tl_pt[1]) / height * full_plan_height_mm
    value_str = replace_values_in_string(
        np.format_float_scientific(black_bar_scale_mm, precision=1),
        METER_SUBDIVISION_DICT,
    )

    # Compute scale font size and bounding box
    value_font_size, resulting_size = get_text_size(
        value_str=value_str,
        inbox_value_px=inbox_value_px,
        line_br_pt=line_br_pt,
        line_tl_pt=line_tl_pt,
    )

    # Compute scale text origin point
    value_text_origin_pt = line_br_pt - np.array(
        inbox_line_thickness_px + inbox_title_px,
        (abs(line_br_pt[1] - line_tl_pt[1]) - resulting_size[0][1]) / 2,
    )
    value_text_origin_pt = tuple(np.array(value_text_origin_pt).astype(int))
    value_text_origin_pt = np.array(line_tl_pt) + (
        inbox_line_thickness_px * 2 + inbox_title_px,
        abs(line_br_pt[1] - line_tl_pt[1]) / 2 - resulting_size[0][0] / 2,
    )
    value_text_origin_pt = tuple(np.array(value_text_origin_pt).astype(int))
    
    # Draw scale text
    cv2.putText(
        img=img,
        text=value_str,
        org=value_text_origin_pt[::-1],
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=value_font_size,
        color=inbox_color,
        thickness=10,
    )

    # Compute title font size and bounding box
    title_font_size, resulting_title_size = get_text_size(
        value_str=inbox_title_text,
        inbox_value_px=inbox_value_px,
        line_br_pt=line_br_pt,
        line_tl_pt=line_tl_pt,
    )

    # Compute title text origin point
    title_text_origin_pt = np.array(line_tl_pt) - (
        inbox_line_thickness_px,
        -abs(line_br_pt[1] - line_tl_pt[1]) / 2 + resulting_title_size[0][0] / 2,
    )
    title_text_origin_pt = tuple(np.array(title_text_origin_pt).astype(int))

    # Draw title text
    _ = cv2.putText(
        img=img,
        text=inbox_title_text,
        org=title_text_origin_pt[::-1],
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=title_font_size,
        color=inbox_color,
        thickness=10,
    )

    return img