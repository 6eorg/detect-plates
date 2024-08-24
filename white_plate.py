import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_treshhold(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel operator to find edges
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

    # Compute the magnitude of the gradients and make edges more pronounced
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(np.clip((magnitude / magnitude.max()) * 255, 0, 255))

    # Apply a stronger threshold to keep only the strongest edges
    _, thresholded = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshhold", thresholded)
    return image
