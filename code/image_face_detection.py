from __future__ import print_function
import cv2 as cv
import os

def process_frame(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        face_roi = frame_gray[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eyes_cascade.detectMultiScale(face_roi)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
    
    return frame

# Hardcoded input and output directories
input_dir   = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\images\occlusions"         # Path to the directory containing input images
output_dir  = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\images\occlusions_output"        # Path to the directory to save processed images

# Hardcoded cascade classifier paths
face_cascade_path = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\code\models\haarcascade_frontalface_alt.xml"
eyes_cascade_path = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\code\models\haarcascade_eye_tree_eyeglasses.xml"

# Load cascade classifiers
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile(face_cascade_path)):
    print('Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_path)):
    print('Error loading eyes cascade')
    exit(0)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in input directory
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
processed_count = 0

for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read and process image
        frame = cv.imread(input_path)
        if frame is None:
            print(f"Could not read image: {input_path}")
            continue
        
        processed_frame = process_frame(frame)
        
        # Save processed image
        if cv.imwrite(output_path, processed_frame):
            processed_count += 1
            print(f"Processed: {filename}")
        else:
            print(f"Failed to save: {filename}")

print(f"\nProcessing complete. {processed_count} images processed.")
print(f"Output saved to: {os.path.abspath(output_dir)}")