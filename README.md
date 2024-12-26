# Parallel Line and Centerline Detection

This Python script detects parallel lines drawn on paper using OpenCV and calculates their centerline. It uses webcam input, converts frames to grayscale, applies Canny edge detection, and isolates a central region using a square mask. Hough transform is used for line detection, and detected lines are filtered and sorted. The centerline is calculated based on the slopes and lengths of the detected lines, handling both near-horizontal and near-vertical line orientations. The detected parallel lines and the calculated centerline are then overlaid on the original frame for visualization.

**Key Features:**

*   Webcam input.
*   Grayscale conversion and Canny edge detection.
*   Square mask for region of interest.
*   Hough transform for line detection.
*   Line filtering and sorting.
*   Centerline calculation for various line orientations using trigonometry.
*   Visual overlay of detected lines and centerline.

**Usage:**

1.  Save the code as a Python file (e.g., `parallel_lines.py`).
2.  Run the script: `python parallel_lines.py`.
3.  Press `q` to quit.
