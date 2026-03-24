import cv2
import numpy as np
import os
from scipy.interpolate import splprep, splev

"""
This script creates an ROI mask from an image by letting you interactively add and adjust (drag) anchors.
You may use the anchors to define a shape as either a polygon (straight lines) or a smooth B-spline curve.
In B-spline mode you can adjust the smoothing parameter with '+' (more smoothing) and '-' (less smoothing).

Controls:
 - Left-click: if near an existing anchor, begin dragging it; otherwise add a new anchor.
 - Drag with the mouse (with left button held) to reposition an anchor.
 - Press 'b': toggle B-spline (curved) mode (current: Polygon / B-spline).
 - In B-spline mode, '+' increases smoothing and '-' decreases smoothing.
 - Press 'c': close the shape (connect last point to first).
 - Press 'r': reset the shape.
 - Press 's': save the shape points.
 - Press 'Esc': exit without saving.
"""

def b_spline_curve(P, num_points=100, smoothing=0.0, closed=False):
    """
    Computes a B-spline curve for the given control points using SciPy's splprep/splev.
    
    Parameters:
      P          : List of (x,y) control points.
      num_points : Number of points to sample along the spline.
      smoothing  : Smoothing factor (s=0 forces interpolation; higher s gives more smoothing).
      closed     : If True, the spline is computed as periodic (closed).
    
    Returns:
      An array of shape (num_points, 2) containing the B-spline curve points (as int32).
    """
    if len(P) < 2:
        return np.array(P, dtype=np.int32)
    P_arr = np.array(P)
    x = P_arr[:, 0]
    y = P_arr[:, 1]
    try:
        tck, _ = splprep([x, y], s=smoothing, per=closed)
        unew = np.linspace(0, 1, num_points)
        out = splev(unew, tck)
        curve = np.vstack(out).T
        return curve.astype(np.int32)
    except Exception as e:
        print("Error in B-spline computation:", e)
        # Fallback: simply connect the points.
        return np.array(P, dtype=np.int32)

def create_mask(img_path):
    # Load the image
    img_original = cv2.imread(img_path)
    if img_original is None:
        print('Error: Image not found.')
        exit()

    # Resize the image for display purposes
    max_display_width = 1000    
    max_display_height = 1000
    h_original, w_original = img_original.shape[:2]
    aspect_ratio = w_original / h_original
    if w_original > max_display_width or h_original > max_display_height:
        if aspect_ratio > 1:
            display_width = max_display_width
            display_height = int(max_display_width / aspect_ratio)
        else:
            display_height = max_display_height
            display_width = int(max_display_height * aspect_ratio)
        img_display = cv2.resize(img_original, (display_width, display_height), interpolation=cv2.INTER_AREA)
    else:
        img_display = img_original.copy()
        display_height, display_width = h_original, w_original

    # Data dictionary for storing interactive parameters and control points.
    data = {
        'points': [],
        'curved': False,       # Default: polygon (straight-line) mode.
        'smoothing': 0.0,      # B-spline smoothing parameter (0 = exact interpolation).
        'closed': False,       # Whether the shape is closed.
        'drag_index': None,    # Index of the anchor currently being dragged.
        'dragging': False      # Whether we are in dragging mode.
    }
    # Distance threshold (in pixels) for detecting an anchor near the click.
    selection_threshold = 10

    def mouse_callback(event, x, y, flags, param):
        # When the left button is pressed, either start dragging an anchor if near one,
        # or add a new anchor.
        if event == cv2.EVENT_LBUTTONDOWN:
            found = False
            for idx, pt in enumerate(param["points"]):
                # Compute Euclidean distance.
                if np.hypot(x - pt[0], y - pt[1]) < selection_threshold:
                    param["drag_index"] = idx
                    param["dragging"] = True
                    found = True
                    print(f"Dragging anchor {idx}")
                    break
            if not found:
                param["points"].append((x, y))
                print(f"Added anchor: ({x}, {y})")
        elif event == cv2.EVENT_MOUSEMOVE:
            # If dragging an existing anchor, update its coordinates.
            if param.get("dragging", False) and param.get("drag_index") is not None:
                param["points"][param["drag_index"]] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging.
            param["drag_index"] = None
            param["dragging"] = False

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback, data)

    print("Instructions:")
    print(" - Left-click near an anchor to drag it; otherwise, left-click adds a new anchor.")
    print(" - Press 'b': toggle B-spline (curved) mode (current: Polygon).")
    print(" - In B-spline mode, '+' increases smoothing and '-' decreases smoothing.")
    print(" - Press 'c': close the shape (connect last point to first).")
    print(" - Press 'r': reset the shape.")
    print(" - Press 's': save the shape points.")
    print(" - Press 'Esc': exit without saving.")

    while True:
        temp_img = img_display.copy()
        # Draw anchors.
        for pt in data['points']:
            cv2.circle(temp_img, pt, 5, (0, 255, 0), -1)
        # Draw connecting lines or B-spline curve if there are at least two points.
        if len(data['points']) >= 2:
            if data['curved']:
                curve_pts = b_spline_curve(data['points'], num_points=200, smoothing=data['smoothing'], closed=data['closed'])
                pts_curve = curve_pts.reshape((-1, 1, 2))
                cv2.polylines(temp_img, [pts_curve], isClosed=data['closed'], color=(255, 0, 0), thickness=2)
            else:
                for i in range(1, len(data['points'])):
                    cv2.line(temp_img, data['points'][i-1], data['points'][i], (255, 0, 0), 2)
                if data['closed'] and len(data['points']) > 2:
                    cv2.line(temp_img, data['points'][-1], data['points'][0], (0, 0, 255), 2)
        # Show current mode and smoothing (if applicable).
        mode_text = "B-spline" if data['curved'] else "Polygon"
        cv2.putText(temp_img, f"Mode: {mode_text}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if data['curved']:
            cv2.putText(temp_img, f"Smoothing: {data['smoothing']:.2f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Image', temp_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('b'):
            data['curved'] = not data['curved']
            mode_str = "B-spline" if data['curved'] else "Polygon"
            print(f"Switched to {mode_str} mode.")
        elif key in (ord('+'), ord('=')):
            if data['curved'] and data['smoothing'] < 10.0:
                data['smoothing'] = min(10.0, data['smoothing'] + 0.5)
                print(f"Smoothing increased to {data['smoothing']:.2f}")
        elif key in (ord('-'), ord('_')):
            if data['curved'] and data['smoothing'] > 0.0:
                data['smoothing'] = max(0.0, data['smoothing'] - 0.5)
                print(f"Smoothing decreased to {data['smoothing']:.2f}")
        elif key == ord('c'):
            if len(data['points']) > 2:
                data['closed'] = True
                if not data['curved']:
                    # In polygon mode, add the first point to the end if needed.
                    if data['points'][0] != data['points'][-1]:
                        data['points'].append(data['points'][0])
                print("Shape closed.")
            else:
                print("Need at least 3 points to close the shape.")
        elif key == ord('r'):
            data['points'] = []
            data['closed'] = False
            print("Shape reset.")
        elif key == ord('s'):
            print("Shape saved.")
            break
        elif key == 27:
            print("Exited without saving.")
            data['points'] = []
            break

    cv2.destroyAllWindows()

    if len(data['points']) > 0:
        # In polygon mode, remove the duplicate closing point before normalization.
        if not data['curved'] and len(data['points']) > 1 and data['points'][0] == data['points'][-1]:
            data['points'] = data['points'][:-1]
        normalized_points = [(x / display_width, y / display_height) for x, y in data['points']]
        mode = "curved" if data['curved'] else "polygon"
        print(f"\nNormalized {mode} control points (relative to display dimensions):")
        print("normalized_points = [")
        for x_norm, y_norm in normalized_points:
            print(f"    [{x_norm:.6f}, {y_norm:.6f}],")
        print("]")
        return {
            "points": normalized_points,
            "mode": mode,
            "smoothing": data['smoothing'],
            "closed": data['closed']
        }
    else:
        print("No shape was drawn.")
        return None

def denormalize_points(normalized_points, image_width, image_height):
    denorm = [(int(x_norm * image_width), int(y_norm * image_height)) for x_norm, y_norm in normalized_points]
    print("\nDenormalized points (relative to image dimensions):")
    print("points = [")
    for x, y in denorm:
        print(f"    [{x:.6f}, {y:.6f}],")
    print("]")
    return denorm

def add_mask_to_img(img_path: str, mask_data):
    if mask_data is None or not mask_data.get("points"):
        print('No points provided for masking.')
        return

    image = cv2.imread(img_path)
    if image is None:
        print('Error: Image not found.')
        exit()
    
    h, w = image.shape[:2]
    denorm_points = denormalize_points(
        normalized_points=mask_data["points"],
        image_width=w,
        image_height=h
    )
    
    if mask_data.get("mode", "polygon") == "curved":
        curve_pts = b_spline_curve(
            denorm_points,
            num_points=200,
            smoothing=mask_data.get("smoothing", 0.0),
            closed=mask_data.get("closed", False)
        )
        polygon_points = curve_pts.reshape((-1, 1, 2))
    else:
        polygon_points = np.array(denorm_points, np.int32)

    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_img, [polygon_points], 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask_img)
    cv2.imshow('Masked Image', masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the image used for drawing the mask
    img_path = os.path.join('..', 'MooTrack360_code' 'datasets', 'raw-images',
                            'version_2025-05-01', '2024_08_22_2024_09_12_0002D1A4B8B7_video03_0004.jpg')
    # Path to the image on which to apply the mask
    img_for_masking_path = os.path.join('..', 'MooTrack360_code', 'datasets', 'raw-images',
                                        'version_2025-05-01', '2024_08_22_2024_09_12_0002D1A4B8B7_video03_0004.jpg')
    
    mask_data = create_mask(img_path=img_path)
    add_mask_to_img(img_path=img_for_masking_path, mask_data=mask_data)
