import cv2

def perspective_correction(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Original', gray)
    
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #print(contours)
    
    contour = max(contours, key=cv2.contourArea)
    
    print(contour)
    
    cv2.drawContours(image, [contour], -1, (0,255,0), 1)
    
    # Display the image with detected corners
    cv2.imshow('Detected Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()