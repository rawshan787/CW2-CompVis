import cv2
import numpy as np
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

class HaarCascadeSaliencyMap:
    def __init__(self, cascade_file=None):
        """Initialize with a specific cascade file or use the default frontalface_alt"""
        if cascade_file is None:
            print('Error loading face cascade')
            exit(0)
        
        self.cascade_file = cascade_file
        self.cascade = cv2.CascadeClassifier(cascade_file)
        self.stages = self._parse_stages()
        print(f"Loaded cascade with {len(self.stages)} stages")
        
    def _parse_stages(self):
        """Parse the XML cascade file to extract stage information"""
        tree = ET.parse(self.cascade_file)
        root = tree.getroot()
        
        stages = []
        for stage in root.findall(".//stages/_"):
            stage_data = {
                'threshold': float(stage.find('stageThreshold').text)
            }
            stages.append(stage_data)
        
        return stages
    
    def generate_saliency_map(self, image, scale_factor=1.1, min_neighbors=0, min_size=(30, 30)):
        """
        Generate a saliency map using a simplified approach
        
        Args:
            image: Input image
            scale_factor: Factor to increase window size at each scale
            min_neighbors: Minimum number of neighbors
            min_size: Minimum size of face
        
        Returns:
            Overlay image with heatmap and face detection
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Initialize heatmap with zeros
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Get normal face detection results for reference
        #faces = self.cascade.detectMultiScale(gray, scaleFactor=scale_factor, 
        #                                    minNeighbors=7, minSize=min_size)
        faces = self.cascade.detectMultiScale(gray)
        
        # Now run detection with relaxed parameters to get more candidates
        all_candidates = []
        
        # Use multiple scales to get comprehensive coverage
        #scale_values = [1.05, 1.1, 1.15, 1.2]
        scale_values = [1.1]
        
        for scale in scale_values:
            candidates = self.cascade.detectMultiScale(
                gray, 
                scaleFactor=scale,
                minNeighbors=min_neighbors,  # Get all candidates
                minSize=min_size
            )
            
            if len(candidates) > 0:
                all_candidates.extend(candidates)
        
        print(f"Found {len(all_candidates)} candidate regions")
        
        # Process each candidate and count overlaps
        for (x, y, w, h) in all_candidates:
            # Add uniform weight to heatmap in window area
            y_end = min(y + h, height)
            x_end = min(x + w, width)
            
            if y_end > y and x_end > x:
                heatmap[y:y_end, x:x_end] += 1
        
        # Normalize to 0-1 range
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Create color heatmap
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
        
        # Draw regular face detection bounding boxes for comparison
        for (x, y, w, h) in faces:
            #cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), 6)
            center = (x + w // 2, y + h // 2)
            cv2.ellipse(overlay, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 6)
        
        return overlay
    
    def process_directory(self, input_dir, output_dir, file_ext=('.jpg', '.jpeg', '.png')):
        """Process all images in a directory and save results to output directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in file_ext:
            image_files.extend(list(input_path.glob(f'*{ext}')))
        
        print(f"Found {len(image_files)} images to process")
        
        for img_path in image_files:
            print(f"Processing {img_path.name}...")
            start_time = time.time()
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Could not read {img_path}")
                continue
            
            # Generate saliency map overlay
            overlay = self.generate_saliency_map(img)
            
            # Save only the overlay result
            base_name = img_path.stem
            cv2.imwrite(str(output_path / f"{base_name}_overlay.jpg"), overlay)
            
            elapsed_time = time.time() - start_time
            print(f"Saved overlay for {img_path.name} in {elapsed_time:.2f} seconds")

def main():
    # Define input and output directories
    input_dir = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\images\myface_occlusions"
    output_dir = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\images\myface_occlusions_heatmap_output"

    # Haar cascade classifier path
    face_cascade_path = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\code\models\haarcascade_frontalface_alt.xml"
    
    print("Starting saliency map generation...")
    start_time = time.time()
    
    # Create saliency map generator
    generator = HaarCascadeSaliencyMap(face_cascade_path)
    
    # Process directory
    generator.process_directory(input_dir, output_dir)
    
    elapsed_time = time.time() - start_time
    print(f"Processing complete! Total time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()