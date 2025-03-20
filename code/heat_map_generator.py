from __future__ import print_function
import cv2 as cv
import os
import numpy as np

# Define input and output directories
input_dir = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\images\myface_edited"
output_dir = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\images\myface_edited_heatmap_output"

# Haar cascade classifier path
face_cascade_path = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\code\models\haarcascade_frontalface_alt.xml"

# Load face cascade
face_cascade = cv.CascadeClassifier()
if not face_cascade.load(cv.samples.findFile(face_cascade_path)):
    print('Error loading face cascade')
    exit(0)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to generate heatmap
def generate_heatmap(frame_gray, faces):
    heatmap = np.zeros_like(frame_gray, dtype=np.float32)
    for (x, y, w, h) in faces:
        heatmap[y:y+h, x:x+w] += 1.0  # Increase heat in detected regions
    heatmap = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv.applyColorMap(heatmap, cv.COLORMAP_JET)  # Apply colormap
    return heatmap_color

# Process each image in input directory
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
processed_count = 0

for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        frame = cv.imread(input_path)
        if frame is None:
            print(f"Could not read image: {input_path}")
            continue
        
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(frame_gray)
        
        # Generate heatmap
        heatmap = generate_heatmap(frame_gray, faces)
        
        # Overlay heatmap on original image
        overlay = cv.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        
        # Save processed image
        if cv.imwrite(output_path, overlay):
            processed_count += 1
            print(f"Processed: {filename}")
        else:
            print(f"Failed to save: {filename}")

print(f"\nProcessing complete. {processed_count} images processed.")
print(f"Output saved to: {os.path.abspath(output_dir)}")