import cv2
import numpy as np

def update_image(y_min, y_max, u_min, u_max, v_min, v_max):
    # Read the image
    image = cv2.imread('raw_autopilot_borroweddrone/55887660.jpg')
    
    # Convert image to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Define the lower and upper thresholds for YUV
    lower_yuv = np.array([y_min, u_min, v_min])
    upper_yuv = np.array([y_max, u_max, v_max])
    
    # Create a mask using the thresholds
    mask = cv2.inRange(yuv_image, lower_yuv, upper_yuv)
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Display the result
    cv2.imshow('Filtered Image', result)

def on_trackbar_change(dummy=None):
    y_min = cv2.getTrackbarPos('Y Min', 'Trackbars')
    y_max = cv2.getTrackbarPos('Y Max', 'Trackbars')
    u_min = cv2.getTrackbarPos('U Min', 'Trackbars')
    u_max = cv2.getTrackbarPos('U Max', 'Trackbars')
    v_min = cv2.getTrackbarPos('V Min', 'Trackbars')
    v_max = cv2.getTrackbarPos('V Max', 'Trackbars')
    
    update_image(y_min, y_max, u_min, u_max, v_min, v_max)

# Create a window and trackbars
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Y Min', 'Trackbars', 0, 255, on_trackbar_change)
cv2.createTrackbar('Y Max', 'Trackbars', 255, 255, on_trackbar_change)
cv2.createTrackbar('U Min', 'Trackbars', 0, 255, on_trackbar_change)
cv2.createTrackbar('U Max', 'Trackbars', 255, 255, on_trackbar_change)
cv2.createTrackbar('V Min', 'Trackbars', 0, 255, on_trackbar_change)
cv2.createTrackbar('V Max', 'Trackbars', 255, 255, on_trackbar_change)

# Call on_trackbar_change to initially display the image
on_trackbar_change()

cv2.waitKey(0)
cv2.destroyAllWindows()