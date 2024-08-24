import cv2
import pytesseract
import numpy as np
import clean_up
import white_plate

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def detect_license_plate_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 100, 200, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate_contour = None

    for contour in contours:
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)

            aspect_ratio = float(w) / h
            if 2 <= aspect_ratio <= 5 and w * h > 500:  # Adjust these parameters based on your needs
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [approx], -1, 255, thickness=cv2.FILLED)
                white_content = cv2.countNonZero(cv2.bitwise_and(blur, blur, mask=mask))

                if white_content > 0.7 * (w * h):  # At least 70% white pixels in the region
                    license_plate_contour = approx
                    break

    if license_plate_contour is None:
        return None

    corners = [(point[0][0], point[0][1]) for point in license_plate_contour]
    return corners
def transform_perspective(image, corners):
    """
    Applies a perspective transformation to the image to make the license plate appear front-facing.

    Parameters:
    image (numpy.ndarray): Input image with the license plate.
    corners (list of tuple): Coordinates of the four corners of the license plate.

    Returns:
    numpy.ndarray: The perspective-transformed image.
    """

    if corners is None:
        return
    # Ensure corners are ordered consistently (top-left, top-right, bottom-right, bottom-left)
    corners = np.array(corners, dtype="float32")

    # Determine the new width and height of the transformed image
    (tl, tr, br, bl) = corners
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Set the destination points for the perspective transform
    destination_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Get the transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(corners, destination_points)

    # Warp the image using the transformation matrix
    warped_image = cv2.warpPerspective(image, transformation_matrix, (max_width, max_height))

    return warped_image

def rotate_if_vertical(image):
    """
    Checks if the image is vertical (taller than it is wide) and, if so, returns the image rotated
    90 degrees clockwise and counterclockwise.

    Parameters:
    image (numpy.ndarray): Input image to be checked and rotated if needed.

    Returns:
    tuple: The original image and the rotated images (if needed).
    """
    h, w = image.shape[:2]
    if h > w:  # Image is vertical
        rotated_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_counterclockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image, rotated_clockwise, rotated_counterclockwise
    else:
        return image, None, None  # No rotation needed

def is_mirrored(corners):
    """
    Detects if the license plate transformation resulted in a mirrored image based on the corner coordinates.

    Parameters:
    corners (numpy.ndarray): Coordinates of the four corners of the license plate.

    Returns:
    bool: True if the image is mirrored, False otherwise.
    """
    tl, tr, br, bl = corners

    # Convert tuples to NumPy arrays for subtraction
    tl = np.array(tl)
    tr = np.array(tr)
    br = np.array(br)
    bl = np.array(bl)

    # Calculate vectors for top and bottom edges
    vector_top = tr - tl
    vector_bottom = br - bl

    # Check if the y-component of the top vector is greater than the bottom vector (indicating a mirror)
    if vector_top[1] > vector_bottom[1]:
        return True

    return False

def analyze_and_correct_orientation(image, corners):
    """
    Analyzes the orientation of the image based on corner positions and corrects it if necessary.
    This function checks for 90-degree rotation and flips the image if needed.

    Parameters:
    image (numpy.ndarray): Input image to be analyzed and corrected.
    corners (list of tuple): Coordinates of the four corners of the license plate.

    Returns:
    numpy.ndarray: The corrected image.
    """
    # Convert corner points to numpy arrays
    tl, tr, br, bl = [np.array(pt) for pt in corners]

    # Determine if the image is vertically oriented (taller than wide)
    width = np.linalg.norm(tr - tl)
    height = np.linalg.norm(bl - tl)

    if height > width:  # If the image is vertically oriented
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        corners = [bl, tl, tr, br]  # Adjust corner positions after rotation

    # Check if the image is mirrored after rotation
    if is_mirrored(corners):
        image = cv2.flip(image, 0)  # Flip vertically to correct mirroring

    return image


def check_and_correct_orientation(image):
    """
    Checks if the image is turned 90 degrees (i.e., taller than wide) and corrects it by rotating 90 degrees counterclockwise and then mirroring vertically.

    Parameters:
    image (numpy.ndarray): The input image to be checked and corrected.

    Returns:
    numpy.ndarray: The corrected image.
    """
    h, w = image.shape[:2]

    # Check if the image is taller than wide (i.e., turned 90 degrees)
    if h > w:
        # Rotate 90 degrees counterclockwise
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Mirror the image horizontally
        image = cv2.flip(image, 0)  # Flip around the x-axis (vertical flip)

    return image

def extract_specific_region(image):
    """
    Extracts a specific region from the image:
    - From the lower 50% to the lower 80% of the image height.
    - 20% from each side of the image width.

    Parameters:
    image (numpy.ndarray): Input image from which the region is to be extracted.

    Returns:
    numpy.ndarray: The extracted region of the image.
    """

    height, width = image.shape[:2]

    # Calculate the coordinates of the region to extract
    start_row = int(height * 0.4)
    end_row = int(height * 0.8)
    start_col = int(width * 0.14)
    end_col = int(width * 0.9)

    # Extract the region from the image
    extracted_region = image[start_row:end_row, start_col:end_col]

    return extracted_region


def extract_text_from_image(image):
    """
    Detects the region with text in an image, frames it, crops it, and extracts text using OCR.

    Parameters:
    image (numpy.ndarray): Input image to be processed.

    Returns:
    str: The extracted text from the detected region.
    """

    if image is not None:
        # Show the cropped image with the detected text region

        # Define the whitelist of characters
        custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.- --psm 6'
        custom_config = None
        # Extract text using pytesseract with the custom configuration
        extracted_text = pytesseract.image_to_string(image, config=custom_config)

        # Print and return the extracted text
        return extracted_text
    else:
        print("No text region found.")
        return ""


def prepare_image_for_ocr(image):
    """
    Prepares an image for OCR by sharpening it, converting it to black and white,
    and applying thresholding if needed.

    Parameters:
    image (numpy.ndarray): Input image to be prepared for OCR.

    Returns:
    numpy.ndarray: The preprocessed image suitable for OCR.
    """

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a kernel to sharpen the image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    # Apply thresholding to make the image purely black and white
    _, binary_image = cv2.threshold(sharpened, 120, 255, cv2.THRESH_BINARY)

    # Alternatively, adaptive thresholding can be used for better results on varying lighting conditions
    # binary_image = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                     cv2.THRESH_BINARY, 11, 2)

    return binary_image



def merge_images(large_image, small_image_1, small_image_2, x, y, w, h, text=None):
    """
    Merges two small images onto the large image:
    - The first small image is placed at the top-right corner of the bounding box.
    - The second small image is placed at the top-left corner of the large image (0,0).

    Parameters:
    large_image (numpy.ndarray): The larger image.
    small_image_1 (numpy.ndarray): The first smaller image to overlay at the top-right corner of the bounding box.
    small_image_2 (numpy.ndarray): The second smaller image to overlay at the top-left corner of the large image.
    x (int): The x-coordinate of the bounding box.
    y (int): The y-coordinate of the bounding box.
    w (int): The width of the bounding box.
    h (int): The height of the bounding box.

    Returns:
    numpy.ndarray: The merged image.
    """
    if small_image_1 is None or small_image_2 is None:
        return large_image

    # Ensure the large image has 3 channels (convert to BGR if grayscale)
    if len(large_image.shape) == 2 or large_image.shape[2] == 1:
        large_image = cv2.cvtColor(large_image, cv2.COLOR_GRAY2BGR)

    # Ensure the small images have 3 channels (convert to BGR if grayscale)
    if len(small_image_1.shape) == 2 or small_image_1.shape[2] == 1:
        small_image_1 = cv2.cvtColor(small_image_1, cv2.COLOR_GRAY2BGR)
    if len(small_image_2.shape) == 2 or small_image_2.shape[2] == 1:
        small_image_2 = cv2.cvtColor(small_image_2, cv2.COLOR_GRAY2BGR)

    # Place the first small image at the top-right corner of the bounding box
    top_right_x = x + w - small_image_1.shape[1]
    top_right_y = y

    if top_right_x >= 0 and top_right_y >= 0 and \
       (top_right_y + small_image_1.shape[0] <= large_image.shape[0]) and \
       (top_right_x + small_image_1.shape[1] <= large_image.shape[1]):
        large_image[top_right_y:top_right_y + small_image_1.shape[0], top_right_x:top_right_x + small_image_1.shape[1]] = small_image_1
    else:
        raise ValueError("The first small image does not fit within the bounds of the large image.")

    # Place the second small image at the top-left corner of the large image
    if small_image_2.shape[0] <= large_image.shape[0] and small_image_2.shape[1] <= large_image.shape[1]:
        large_image[0:small_image_2.shape[0], 0:small_image_2.shape[1]] = small_image_2
    else:
        raise ValueError("The second small image does not fit within the bounds of the large image.")

    # Add text above the first small image if provided
    if text is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        text_color = (180, 250, 180)  # Light green color in BGR
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = top_right_x
        text_y = max(0, top_right_y - text_size[1])  # Ensure the text is within the image bounds
        cv2.putText(large_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

    return large_image

# Function to detect motion in a video and frame each object separately
def detect_motion(video_path=0): #when no input it takes the webcam !
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize the first frame for motion detection
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        # Compute the absolute difference between the current frame and the next frame
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        #used to merge a nice picture
        prepared_image = None
        extracted_region = None
        text_to_display = None
        x,y,w,h = None, None,None,None


        # Blur the image to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold the image to create a binary image
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        # Dilate the threshold image to fill in holes
        dilated = cv2.dilate(thresh, None, iterations=3)

        # Find contours of the moving objects
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 40000:  # Filter small contours
                continue

            # Get the bounding box coordinates around the moving object
            x, y, w, h = cv2.boundingRect(contour)

            # Draw the bounding box around each detected object
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the object from the frame
            cropped_object = frame1[y:y + h, x:x + w]

            # Display the cropped object

            extracted_region = extract_specific_region(cropped_object)
            cv2.imshow("extracted region",extracted_region)

            ### TEST FOR WHTE
            """
            white_img = white_plate.apply_treshhold(extracted_region)
            white_text = extract_text_from_image(white_img)
            if white_text != "":
                print("text from white: ", white_text)
                cleaned = clean_up.clean_swiss_license_plate(white_text)
                print("text from white cleaned: ",cleaned )
                collected_plate_nrs.add(cleaned)
            
            """

            ##END TEST FOR WHITE

            ## TEST READ DIRECT FROM UNTRANSFORMED IMAGE
            text_from_extracted = extract_text_from_image(extracted_region)
            if text_from_extracted != "":
                print("text from extracted", text_from_extracted)
                cleaned = clean_up.clean_swiss_license_plate(text_from_extracted)
                print("text from extracted cleaned: ", cleaned )
                collected_plate_nrs.add(cleaned)

            ## END TEST


            # Detect the corners of the license plate
            corners = detect_license_plate_corners(extracted_region)

            # Apply perspective transformation to correct the license plate orientation
            transformed_image = transform_perspective(extracted_region, corners)


            if transformed_image is not None:
                corrected = check_and_correct_orientation(transformed_image)
                cv2.imshow("turned", corrected)
                prepared_image = prepare_image_for_ocr(corrected)
                cv2.imshow("prepared_image", prepared_image)

                text = extract_text_from_image(prepared_image)
                print("prepared:", text.rstrip('\n'))
                print("prepared - cleaned:", clean_up.clean_swiss_license_plate(text))
                text = extract_text_from_image(corrected)
                print("corrected:", text.rstrip('\n'))
                print("corrected - cleaned", clean_up.clean_swiss_license_plate(text))
                text_to_display = clean_up.clean_swiss_license_plate(text)

                ## read with easy_ocr
            #    ocr_easy_ocr.extract_with_easy_ocr(corrected)

                showThat = False
                if showThat:
                    cv2.imshow(text, transformed_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


        # Display the frame with the detected motion
        cv2.imshow("Motion Detection", merge_images(frame1, prepared_image, extracted_region, x,y,w,h, text_to_display))


        # Wait for a short period and break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Update the frames
        frame1 = frame2
        ret, frame2 = cap.read()

        if not ret:
            break

    # Release the video capture object and close the windows
    cap.release()
    cv2.destroyAllWindows()

# Path to the video file
video_path = 'video1.mp4'
collected_plate_nrs = set()

# Call the motion detection function
detect_motion(video_path)
print("collected: ", collected_plate_nrs)