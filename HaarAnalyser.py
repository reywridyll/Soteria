#Model Reviewer
import cv2
import numpy as np

# Load video
video = cv2.VideoCapture(f'data/distance_test.mp4')

# Load cascade
body_cascade = cv2.CascadeClassifier('haar/cascade.xml')

# Configure blob detection
params = cv2.SimpleBlobDetector_Params() 

# Change thresholds
params.minThreshold = 200
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 30

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.35

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.4

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1

# Initialize blob detection
detector = cv2.SimpleBlobDetector_create(params)

# Set kernel
kernel = np.array([[1,1,1],
                   [0,0,0],
                   [1,1,1]], np.uint8)

counter = 0;
skip = input("Skip first N frames: ")
while(video.isOpened()):
    #skip early frames for demo purposes!!
    counter += 1
    ret, im = video.read()
    if counter > int(skip):
        
        # Convert to grey scale
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        grey_denoised = cv2.fastNlMeansDenoising(grey)
        
        # Detect objects
        bodies = body_cascade.detectMultiScale(grey_denoised)

        # Check if cascade detected something
        if len(bodies) > 0 :
            
            for body in bodies:
                x, y, w ,h = body
                im = cv2.rectangle(im,(x,y ,w , h),(36,255,12),2)
                cv2.putText(im, "target", (x,  y- 10), cv2.FONT_HERSHEY_SIMPLEX , 0.9, (36,255,12), 2)
        
        # Fall back to blob detection if cascade has no detections  
        else:
            line_frame = cv2.Canny(grey_denoised, 100,150)    
            filled_frame = cv2.morphologyEx(line_frame, cv2.MORPH_CLOSE, kernel, iterations=5)
            retval, threshold = cv2.threshold(filled_frame, 230, 255, cv2.THRESH_BINARY_INV)
            keypoints = detector.detect(threshold)
            im = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1) #wait until any key is pressed
        
        cv2.imshow('img',im)
    else:
        print("Frame skipped!")
        
video.release()
cv2.destroyAllWindows()