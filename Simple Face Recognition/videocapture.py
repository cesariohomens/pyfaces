# To install this script in Ubuntu 18.04
#
# Video face detection in Python3
# 
# Human detection -> https://github.com/opencv/opencv/tree/master/data/haarcascades
#
# sudo apt install python3 python3-pip
# pip3 install opencv-contrib-python opencv-python

# Libraries
import cv2

# Use webcam
cap = cv2.VideoCapture(0)

# Face detection
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Converting the frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray image using haarcascade_frontalface_default.xml
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Number of faces detected
    text = "Found {0} faces!".format(len(faces))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10,30), font, .5, (0,255,0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Face detection', frame)

    # Exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
