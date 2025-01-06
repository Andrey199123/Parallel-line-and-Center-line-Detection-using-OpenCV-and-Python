# Andrey Vasilyev 12/26/24
import cv2
import numpy as np
import math

### IMPORTANT ###
slope_value = 0.35
slope_value2 = 0.3

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Greyscale cuz we don't know the color of lines nor paper
    canny_image = cv2.Canny(grey, 100, 120) # Edge detection

    """Find the square mask region given screen dimensions"""
    rows, columns = frame.shape[:2] 
    square_size = min(columns, rows)*3/5
    center_column, center_row = columns // 2, rows // 2
    half_square = square_size // 2
    vertices = np.array([[
        [center_column - half_square, center_row + half_square],
        [center_column - half_square, center_row - half_square],
        [center_column + half_square, center_row - half_square],
        [center_column + half_square, center_row + half_square]
    ]], dtype=np.int32)

    """Provides a square through which parallel lines get detected"""
    # Inspired by: https://www.youtube.com/watch?v=G0cHyaP9HaQ
    mask = np.zeros_like(canny_image)
    cv2.fillPoly(mask, vertices, 255) # Make everything outside of red square  white so it doesn't get detected
    hough_img = cv2.bitwise_and(canny_image, mask)

    # Found from: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    parallel_lines = cv2.HoughLinesP(
        hough_img,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=100,
        maxLineGap=50
    )
    
    if parallel_lines is None:
        parallel_lines = []
    else:
        # Converts 2D array into 1D array
        parallel_lines = parallel_lines[:, 0, :]
    
    filtered_lines = []
    midpoints = []

    for line in parallel_lines:

        x1, y1, x2, y2 = line

        if y2 < y1: # Order y coordinates in increasing order for each line
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            
        line = np.array([x1,y1,x2,y2])
        mp = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Check if midpoint is at least 100 pixels away from all other midpoints to not get lines that are too close to each other
        if all(math.sqrt((mp[0] - existing_mp[0])**2 + (mp[1] - existing_mp[1])**2) >= 100 for existing_mp in midpoints):
            filtered_lines.append(line)
            midpoints.append(mp)

    filtered_lines = np.array(filtered_lines, dtype=np.int32) # numpy arrays are easier to unpack

    if filtered_lines.size > 0: # Checks to see if something is detected to get sorted 
        filtered_lines = sorted(filtered_lines, key=lambda line: min(line[0], line[2])) # Also sorts the lines by x coordinates

    if filtered_lines is not None and len(filtered_lines) > 0: # Check if any lines are detected to be drawn
        for x1, y1, x2, y2 in filtered_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Draw parallel lines here

        # Here, we sort the lines and their coordinates so that the slopes are found correctly (the origin is at the top-left)
        if len(filtered_lines) >= 2:
            x1_0, y1_0, x2_0, y2_0 = filtered_lines[0]
            if x2_0 < x1_0:
                x1_0, x2_0 = x2_0, x1_0
                y1_0, y2_0 = y2_0, y1_0
            if x2_0 - x1_0 != 0:
                first_slope = (y2_0 - y1_0) / (x2_0 - x1_0)
            else:
                first_slope = float('inf') # Avoid division by 0
            x1_1, y1_1, x2_1, y2_1 = filtered_lines[1]
            if x2_1 < x1_1:
                x1_1, x2_1 = x2_1, x1_1
                y1_1, y2_1 = y2_1, y1_1
            if x2_1 - x1_1 != 0:
                second_slope = (y2_1 - y1_1) / (x2_1 - x1_1)
            else:
                second_slope = float('inf')

            if (abs(first_slope) < slope_value or abs(second_slope) < slope_value and abs(first_slope + second_slope)/2 < 2 * math.sqrt(3)) or (abs(abs(first_slope) - abs(second_slope)) < slope_value2 and abs(first_slope) < 1 and abs(second_slope) < 1):
                """If the slopes of the lines are less than a certain value, then we can find the center between the two lines and draw a line with the average length of the two lines"""
        
                line1_length = math.sqrt((filtered_lines[0][2] - filtered_lines[0][0])**2 + (filtered_lines[0][3] - filtered_lines[0][1])**2)
                line2_length = math.sqrt((filtered_lines[1][2] - filtered_lines[1][0])**2 + (filtered_lines[1][3] - filtered_lines[1][1])**2)

                # Some trig using the slopes to go from the center to the endpoints of centerline
                angle = np.arctan((first_slope + second_slope) / 2)
                end_x1 = int((filtered_lines[0][0] + filtered_lines[1][0] + filtered_lines[0][2] + filtered_lines[1][2]) // 4 - ((line1_length + line2_length) / 4) * np.cos(angle))
                end_y1 = int((filtered_lines[0][1] + filtered_lines[1][1] + filtered_lines[0][3] + filtered_lines[1][3]) // 4 - ((line1_length + line2_length) / 4) * np.sin(angle))
                end_x2 = int((filtered_lines[0][0] + filtered_lines[1][0] + filtered_lines[0][2] + filtered_lines[1][2]) // 4 + ((line1_length + line2_length) / 4) * np.cos(angle))
                end_y2 = int((filtered_lines[0][1] + filtered_lines[1][1] + filtered_lines[0][3] + filtered_lines[1][3]) // 4 + ((line1_length + line2_length) / 4) * np.sin(angle))
                
                # Centerline for relatively horizontal cases
                cv2.line(frame, (end_x1, end_y1), (end_x2, end_y2), (0, 0, 255), 3)
            else:
                """Otherwise, if the slopes are too large, we need to average the x coordinaes and get the minimum and maximum y-value since the order in which the coordinates are givne (detected) is random"""
                # Centerline for relatively vertical cases
                cv2.line(frame, ((filtered_lines[0][0] + filtered_lines[1][0]) // 2, (filtered_lines[0][1] + filtered_lines[1][1]) // 2), ((filtered_lines[0][2] + filtered_lines[1][2]) // 2, (filtered_lines[0][3] + filtered_lines[1][3]) // 2), (0, 0, 255), 3)

    # Found from: https://www.geeksforgeeks.org/python-opencv-cv2-polylines-method/
    cv2.polylines(frame, vertices, isClosed=True, color=(0, 255, 255), thickness=2) # Red square where parallel lines are looked for (Basically a mask)

    cv2.imshow("Parallel line and Centerline detection by Andrey Vasilyev", frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
