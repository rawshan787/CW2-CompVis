import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time

class HaarCascadeSaliencyMap:
    def __init__(self, cascade_file=None):
        """Initialize with a specific cascade file or use the default frontalface_alt"""
        if cascade_file is None:
            cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
        
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
        Generate a saliency map using a more efficient approach
        
        Args:
            image: Input image
            scale_factor: Factor to increase window size at each scale
            min_neighbors: Minimum number of neighbors
            min_size: Minimum size of face
        
        Returns:
            Original image, saliency map, overlay
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Initialize heatmap with zeros
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # For efficiency, we'll use an approximation approach by running detectMultiScale
        # with different parameters and accumulating the results
        
        # First, get normal face detection results for reference
        faces = self.cascade.detectMultiScale(gray, scaleFactor=scale_factor, 
                                            minNeighbors=5, minSize=min_size)
        
        # Now run detection with relaxed parameters to get more candidates
        # We set minNeighbors to 0 to get all candidates
        all_candidates = []
        
        # Use multiple scales to get comprehensive coverage
        scale_values = [1.05, 1.1, 1.15, 1.2]
        
        for scale in scale_values:
            candidates = self.cascade.detectMultiScale(
                gray, 
                scaleFactor=scale,
                minNeighbors=0,  # Get all candidates
                minSize=(20, 20)
            )
            
            if len(candidates) > 0:
                all_candidates.extend(candidates)
        
        print(f"Found {len(all_candidates)} candidate regions")
        
        # Process each candidate and calculate a confidence-like score
        # based on how many neighbors it has (approximating stage confidence)
        confidence_map = {}
        
        for (x, y, w, h) in all_candidates:
            key = f"{x},{y},{w},{h}"
            if key not in confidence_map:
                confidence_map[key] = 0
            confidence_map[key] += 1
        
        # Create a weighted heatmap based on overlapping regions
        for key, confidence in confidence_map.items():
            x, y, w, h = map(int, key.split(','))
            
            # Normalize confidence (higher values = higher confidence)
            norm_confidence = min(confidence / 20, 1.0)  # Cap at 1.0
            
            # Add weighted confidence to heatmap in window area
            # Use a Gaussian-like weight distribution to center on facial features
            y_indices, x_indices = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            
            # Create a 2D Gaussian weight centered on the window
            dist_from_center = ((y_indices - center_y) ** 2) / (2 * (h/4) ** 2) + \
                              ((x_indices - center_x) ** 2) / (2 * (w/4) ** 2)
            weight = np.exp(-dist_from_center)
            
            # Apply weighted confidence to heatmap
            y_end = min(y + h, height)
            x_end = min(x + w, width)
            h_used = y_end - y
            w_used = x_end - x
            
            if h_used > 0 and w_used > 0:
                heatmap[y:y_end, x:x_end] += weight[:h_used, :w_used] * norm_confidence
        
        # Normalize to 0-1 range
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Create color heatmap
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
        
        # Draw regular face detection bounding boxes for comparison
        for (x, y, w, h) in faces:
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        return image, heatmap, overlay
    
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
            
            # Generate saliency map
            original, heatmap, overlay = self.generate_saliency_map(img)
            
            # Save results
            base_name = img_path.stem
            cv2.imwrite(str(output_path / f"{base_name}_original.jpg"), original)
            cv2.imwrite(str(output_path / f"{base_name}_heatmap.jpg"), 
                       (heatmap * 255).astype(np.uint8))
            cv2.imwrite(str(output_path / f"{base_name}_overlay.jpg"), overlay)
            
            # Create a figure with all three images for easier comparison
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap, cmap='jet')
            plt.title('Saliency Heatmap')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title('Overlay with Face Detection')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(str(output_path / f"{base_name}_comparison.jpg"))
            plt.close()
            
            elapsed_time = time.time() - start_time
            print(f"Saved results for {img_path.name} in {elapsed_time:.2f} seconds")

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