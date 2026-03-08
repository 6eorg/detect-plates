import cv2
def find_contour_method_1(frame1, frame2):
    # Compute the absolute difference between the current frame and the next frame
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # used to merge a nice picture
    prepared_image = None
    extracted_region = None
    text_to_display = None
    x, y, w, h = None, None, None, None

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate the threshold image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours of the moving objects
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours,_


def find_contour_method_2(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Morphological operations to separate objects better
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dilated = cv2.dilate(opening, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, _