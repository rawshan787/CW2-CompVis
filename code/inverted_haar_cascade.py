from __future__ import print_function
import cv2 as cv
import argparse

def detectAndDisplay(frame):
    # Invert the colors of the frame
    frame_inverted = cv.bitwise_not(frame)
    
    # Convert the inverted frame to grayscale
    frame_gray = cv.cvtColor(frame_inverted, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        # Draw the ellipse on the inverted frame
        frame_inverted = cv.ellipse(frame_inverted, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
    
    # Display the inverted frame with detected faces
    cv.imshow('Capture - Face detection', frame_inverted)

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default=r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\code\models\haarcascade_frontalface_alt.xml")
parser.add_argument('--camera', help='Camera device number.', type=int, default=0)
args = parser.parse_args()

# Load the face cascade
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

# Initialize the video capture
camera_device = args.camera
cap = cv.VideoCapture(camera_device)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

# Main loop to capture frames and process them
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    
    # Detect and display faces on the inverted frame
    detectAndDisplay(frame)
    
    # Exit on ESC key press
    if cv.waitKey(10) == 27:
        break

# Release the video capture and close windows
cap.release()
cv.destroyAllWindows()