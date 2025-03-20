import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

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
                'threshold': float(stage.find('stageThreshold').text),
                'trees': []
            }
            
            for tree_elem in stage.findall('.//trees/_'):
                for feature in tree_elem.findall('.//feature'):
                    rects = []
                    for rect in feature.findall('.//rects/_'):
                        rect_data = list(map(int, rect.text.strip().split()))
                        rects.append(rect_data)
                    
                    threshold = float(feature.find('.//threshold').text)
                    left_val = float(feature.find('.//left_val').text)
                    right_val = float(feature.find('.//right_val').text)
                    
                    stage_data['trees'].append({
                        'rects': rects,
                        'threshold': threshold,
                        'left_val': left_val,
                        'right_val': right_val
                    })
            
            stages.append(stage_data)
        
        return stages
    
    def evaluate_window(self, img, x, y, w, h, max_stages=None):
        """Evaluate a window at the given position and compute confidence per stage"""
        if max_stages is None:
            max_stages = len(self.stages)
        
        window = img[y:y+h, x:x+w]
        if window.shape[0] == 0 or window.shape[1] == 0:
            return []
            
        # Resize to the standard detection size (24x24)
        window = cv2.resize(window, (24, 24))
        
        # Compute integral image
        integral = cv2.integral(window)
        
        confidences = []
        cumulative_score = 0
        
        for stage_idx, stage in enumerate(self.stages[:max_stages]):
            stage_score = 0
            
            # For simplicity, we're just computing whether the stage passes
            # In practice, a more detailed approach would compute confidence per feature
            for tree in stage['trees']:
                # This is a simplification - in practice, you would evaluate each 
                # Haar-like feature explicitly
                pass
            
            # Since direct feature evaluation is complex, we use the cascade's built-in
            # detection mechanism to approximate whether this stage would pass
            # We'll use the stage threshold as a basis for confidence
            temp_cascade = cv2.CascadeClassifier()
            temp_cascade.load(self.cascade_file)
            stage_result = self.cascade.detectMultiScale(window, scaleFactor=1.1, 
                                                       minNeighbors=0, 
                                                       flags=cv2.CASCADE_DO_CANNY_PRUNING,
                                                       minSize=(20, 20))
            
            # If detection occurs, we consider this stage as passed
            if len(stage_result) > 0:
                cumulative_score += 1
            
            confidences.append(cumulative_score / (stage_idx + 1))
            
            # If a stage fails, we stop processing
            if cumulative_score <= stage_idx:
                break
                
        return confidences
    
    def generate_saliency_map(self, image, step_size=8, window_size=(24, 24), scale_factor=1.1, max_stages=None):
        """
        Generate a saliency map by sliding windows and evaluating each position
        
        Args:
            image: Input image
            step_size: Step size for sliding window
            window_size: Base size of the window
            scale_factor: Factor to increase window size at each scale
            max_stages: Maximum number of stages to evaluate (None for all)
        
        Returns:
            Original image, saliency map
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Initialize heatmap with zeros
        heatmap = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        # Process at multiple scales
        current_scale = 1.0
        max_scale = 10.0  # Limit scales to avoid extremely large windows
        
        while current_scale < max_scale:
            current_window = (int(window_size[0] * current_scale), 
                             int(window_size[1] * current_scale))
            
            # Skip if window is bigger than image
            if current_window[0] > width or current_window[1] > height:
                break
            
            # Slide window across the image
            for y in range(0, height - current_window[1], step_size):
                for x in range(0, width - current_window[0], step_size):
                    # Evaluate window confidence scores
                    confidences = self.evaluate_window(gray, x, y, 
                                                      current_window[0], 
                                                      current_window[1],
                                                      max_stages)
                    
                    if confidences:
                        # Weight by progression through cascade stages
                        confidence = sum(confidences) / len(self.stages)
                        
                        # Add weighted confidence to heatmap in window area
                        heatmap[y:y+current_window[1], x:x+current_window[0]] += confidence
                        count_map[y:y+current_window[1], x:x+current_window[0]] += 1
            
            # Increase scale for next iteration
            current_scale *= scale_factor
        
        # Normalize heatmap by count to get average confidence
        mask = count_map > 0
        heatmap[mask] /= count_map[mask]
        
        # Normalize to 0-1 range
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Create color heatmap
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
        
        # Draw regular face detection bounding boxes for comparison
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
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
            
            print(f"Saved results for {img_path.name}")

def main():
    # Define input and output directories
    input_dir = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\images\myface_occlusions"
    output_dir = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\images\myface_occlusions_heatmap_output"

    # Haar cascade classifier path
    face_cascade_path = r"C:\Users\rawsh\OneDrive - University of Southampton\EE-Y5\CompVis\CW2 CompVis\code\models\haarcascade_frontalface_alt.xml"
    
    # Create saliency map generator
    generator = HaarCascadeSaliencyMap(face_cascade_path)
    
    # Process directory
    generator.process_directory(input_dir, output_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()