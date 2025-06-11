from matplotlib.pyplot import figure
import matplotlib.image as image
from matplotlib import pyplot as plt


import numpy as np
from PIL import Image

import cv2


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def detect_corners(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    cv2.imwrite('debug_original.jpg', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('debug_gray.jpg', gray)

    # Apply Bilateral Filter for noise reduction while preserving edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75) # Retaining original bilateral filter for edge preservation
    cv2.imwrite('debug_blurred.jpg', blurred)

    # Canny Edge Detection with adjusted thresholds
    edges = cv2.Canny(blurred, 30, 90) # Lowering thresholds to capture fainter edges
    cv2.imwrite('debug_edges.jpg', edges)

    # Morphological Closing to connect broken edge lines with a larger, rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) # Using a rectangular kernel
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('debug_closed_edges.jpg', closed_edges)

    # Find contours
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    chessboard_outer_corners = None
    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If the approximated contour has 4 vertices (a quadrilateral) and is reasonably large
        if len(approx) == 4 and cv2.contourArea(approx) > 5000: # Reverted to lower minimum area
            # Check aspect ratio for square-like contours
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.8 <= aspect_ratio <= 1.2: # Allow for some perspective distortion
                chessboard_outer_corners = order_points(approx.reshape(4, 2))
                
                # Draw the detected contour and corners for debugging
                img_contour = img.copy()
                cv2.drawContours(img_contour, [approx], -1, (0, 255, 0), 10) # Green contour, thick line
                for corner_pt in chessboard_outer_corners:
                    cv2.circle(img_contour, (int(corner_pt[0]), int(corner_pt[1])), 20, (0, 0, 255), -1) # Red circles for corners
                cv2.imwrite('debug_outer_contour.jpg', img_contour)

                print("Chessboard outer contour successfully detected.")
                break # Found the largest likely chessboard

    if chessboard_outer_corners is None:
        raise ValueError("Could not find a clear quadrilateral contour representing the chessboard.")
    
    return chessboard_outer_corners

def four_point_transform(image, pts):
    img = Image.open(image)
    image = np.array(img)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    img = Image.fromarray(warped, "RGB")
    return img

def plot_grid_on_transformed_image(image):
    corners = np.array([[0,0], 
                    [image.size[0], 0], 
                    [0, image.size[1]], 
                    [image.size[0], image.size[1]]])
    
    corners = order_points(corners)

    figure(figsize=(10, 10), dpi=80)
    implot = plt.imshow(image)
    
    TL = corners[0]
    BL = corners[3]
    TR = corners[1]
    BR = corners[2]

    def interpolate(xy0, xy1):
        x0,y0 = xy0
        x1,y1 = xy1
        dx = (x1-x0) / 8
        dy = (y1-y0) / 8
        pts = [(x0+i*dx,y0+i*dy) for i in range(9)]
        return pts

    ptsT = interpolate( TL, TR )
    ptsL = interpolate( TL, BL )
    ptsR = interpolate( TR, BR )
    ptsB = interpolate( BL, BR )
        
    for a,b in zip(ptsL, ptsR):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
    for a,b in zip(ptsT, ptsB):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
        
    plt.axis('off')
    plt.savefig('chessboard_transformed_with_grid.jpg')
    return ptsT, ptsL

def main(image_path):
    try:
        corners = detect_corners(image_path)
        print("Detected corners:")
        print(corners)
        
        # Transform the image to get a bird's-eye view of the chessboard
        transformed_board = four_point_transform(image_path, corners)
        # Convert PIL Image to NumPy array for saving with OpenCV
        transformed_board_np = np.array(transformed_board)
        cv2.imwrite('debug_cropped_board.jpg', transformed_board_np)
        print("Cropped chessboard image saved as debug_cropped_board.jpg")

        # You can add code here to visualize the corners on the original image
        img_original = cv2.imread(image_path)
        if img_original is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(img_original, (int(x), int(y)), 5, (0, 0, 255), -1) # Red circles
            cv2.imwrite('output_corners.jpg', img_original)
            print("Original image with detected corners saved as output_corners.jpg")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_chess_vision.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    main(image_path) 