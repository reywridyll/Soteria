#- import the necessary packages -#
import numpy as np
import cv2

# Load the class labels the YOLO model was trained on
LABELS = open("yolo/objects.names").read().strip().split("\n")

# Load config and weights
net = cv2.dnn.readNetFromDarknet("yolo/yolocfg.cfg", "yolo/yolocfg_best.weights")

# GPU OR CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Load video
video = cv2.VideoCapture('data/distance_test.mp4')

# Video dimensions 
H, W = 1080, 1920

# Skip early frames for demo purposes
skip = input("Skip first N frames: ")
counter = 0
while(video.isOpened()):
    
    counter += 1
    ret, image = video.read()
    
    if counter > int(skip):
        
        # Determine the output layer
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        # Construct a blob from the input image and then perform a forward pass
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # Lists for detected bounding boxes, confidences, and classIDs 
        boxes = []
        confidences = []
        classIDs = []

        # Loop over each of the layer outputs
        for output in layerOutputs:
            
            # Loop over each of the detections
            for detection in output:
                
                # Extract the class ID and confidence
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                
                # Filter out weak predictions
                if confidence > 0.2:
                    
                    # Store the bounding boxes, confidence, and classids for filtered detection 
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Apply non maximum suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.6)

        # Check if a something is detected.
        if len(idxs) > 0:
            
            # Loop over detections
            for i in idxs.flatten():
                
                # Extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                # Define a colour
                color = (0,0,255)
                
                # Scale the boxes slightly to make it easier to see the object
                scaled_w = (x + w)*1.005
                scaled_h = (y + h)*1.005
                
                # Draw a bounding box and label on the image
                cv2.rectangle(image, (x, y), (int(scaled_w), int(scaled_h)), color, 1)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Define a key to break out of the loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # Display frame
        cv2.imshow('img',image)

    else:
        print("Skipped!")

video.release()
cv2.destroyAllWindows()