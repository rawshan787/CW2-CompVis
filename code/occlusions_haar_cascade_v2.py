import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
from PIL import Image, ImageTk
import uuid
import os

class OcclusionObject:
    def __init__(self, id=None):
        self.id = id if id else str(uuid.uuid4())
    
    def contains_point(self, x, y):
        pass
    
    def draw(self, frame):
        pass
    
    def move(self, dx, dy):
        pass


class Rectangle(OcclusionObject):
    def __init__(self, x1, y1, x2, y2, color=(0, 0, 0)):
        super().__init__()
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.color = color
    
    def contains_point(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def draw(self, frame):
        # Make sure the rectangle is within the frame boundaries
        x1 = max(0, self.x1)
        y1 = max(0, self.y1)
        x2 = min(frame.shape[1], self.x2)
        y2 = min(frame.shape[0], self.y2)
        
        # Draw the rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
    
    def move(self, dx, dy):
        self.x1 += dx
        self.y1 += dy
        self.x2 += dx
        self.y2 += dy
    
    def __str__(self):
        color_name = "Black" if self.color == (0, 0, 0) else "White" if self.color == (255, 255, 255) else f"RGB{self.color}"
        return f"{color_name} Rectangle ({self.x2-self.x1}x{self.y2-self.y1})"


class ImageOcclusion(OcclusionObject):
    def __init__(self, image_path, x, y):
        super().__init__()
        self.image_path = image_path
        self.x = x
        self.y = y
        
        # Load the image
        self.pil_image = Image.open(image_path)
        self.cv_image = cv2.cvtColor(np.array(self.pil_image), cv2.COLOR_RGB2BGR)
        self.height, self.width = self.cv_image.shape[:2]
    
    def contains_point(self, x, y):
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height
    
    def draw(self, frame):
        # Create a region of interest
        roi = frame[self.y:self.y + self.height, self.x:self.x + self.width]
        
        # Check if ROI is within frame bounds
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            # Handle images with transparency (PNG)
            if self.cv_image.shape[2] == 4:
                # Make sure dimensions match before trying to blend
                img_height, img_width = self.cv_image.shape[:2]
                roi_height, roi_width = roi.shape[:2]
                
                # Calculate the valid region for blending
                valid_h = min(img_height, roi_height)
                valid_w = min(img_width, roi_width)
                
                # Extract the alpha channel (ensure it's in range 0-1)
                alpha = self.cv_image[:valid_h, :valid_w, 3] / 255.0
                
                # Create a 3D alpha for broadcasting with RGB channels
                alpha_3d = np.dstack([alpha] * 3)
                
                # Get the RGB part of the image
                rgb = self.cv_image[:valid_h, :valid_w, :3]
                
                # Get the ROI part that matches our image dimensions
                roi_part = roi[:valid_h, :valid_w]
                
                # Blend using the alpha channel: result = foreground * alpha + background * (1 - alpha)
                blended = (rgb * alpha_3d + roi_part * (1 - alpha_3d)).astype(np.uint8)
                
                # Place the blended image back into the frame
                frame[self.y:self.y + valid_h, self.x:self.x + valid_w] = blended
            else:
                # No alpha channel, just overlay the image
                # Use smaller of the dimensions to avoid out-of-bounds errors
                h = min(self.cv_image.shape[0], roi.shape[0])
                w = min(self.cv_image.shape[1], roi.shape[1])
                frame[self.y:self.y + h, self.x:self.x + w] = self.cv_image[:h, :w]
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
    
    def __str__(self):
        filename = os.path.basename(self.image_path)
        return f"Image: {filename} ({self.width}x{self.height})"


class FaceOcclusionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Configure main window
        self.window.geometry("1200x700")
        self.window.configure(background="#f0f0f0")
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video capture device")
            self.window.destroy()
            return
        
        # Get video dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create main frame for layout
        self.main_frame = ttk.Frame(window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left frame for video display
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create canvas for video display
        self.canvas = tk.Canvas(self.video_frame, width=self.width, height=self.height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create right frame for controls
        self.controls_frame = ttk.Frame(self.main_frame, width=300)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        
        # Drawing and interaction variables
        self.occlusion_objects = []
        self.drawing = False
        self.moving = False
        self.start_x = 0
        self.start_y = 0
        self.current_rectangle = None
        self.selected_object = None
        self.current_color = (0, 0, 0)  # Default black
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Create control widgets
        self.create_widgets()
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Update display
        self.update()
        
        # Set up window close callback
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_widgets(self):
        # Color selection
        ttk.Label(self.controls_frame, text="Occlusion Controls", font=("Arial", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        # Object type frame
        obj_type_frame = ttk.Frame(self.controls_frame)
        obj_type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(obj_type_frame, text="Type:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Rectangle button
        self.rect_btn = ttk.Button(obj_type_frame, text="Draw Rectangle", command=self.set_rectangle_mode)
        self.rect_btn.pack(side=tk.LEFT, padx=5)
        
        # Import Image button
        self.image_btn = ttk.Button(obj_type_frame, text="Import Image", command=self.import_image)
        self.image_btn.pack(side=tk.LEFT, padx=5)
        
        # Color frame
        color_frame = ttk.Frame(self.controls_frame)
        color_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(color_frame, text="Color:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Black button
        self.black_btn = ttk.Button(color_frame, text="Black", command=lambda: self.set_color((0, 0, 0)))
        self.black_btn.pack(side=tk.LEFT, padx=5)
        
        # White button
        self.white_btn = ttk.Button(color_frame, text="White", command=lambda: self.set_color((255, 255, 255)))
        self.white_btn.pack(side=tk.LEFT, padx=5)
        
        # Custom color button
        self.custom_color_btn = ttk.Button(color_frame, text="Custom", command=self.choose_custom_color)
        self.custom_color_btn.pack(side=tk.LEFT, padx=5)
        
        # Current color indicator
        self.color_indicator = tk.Canvas(self.controls_frame, width=50, height=20, bg="#000000", highlightthickness=1)
        self.color_indicator.pack(anchor=tk.W, pady=5)
        
        # Interaction info
        interaction_frame = ttk.Frame(self.controls_frame)
        interaction_frame.pack(fill=tk.X, pady=10)
        
        interaction_text = "Interactions:\n• Click & drag to draw/move\n• Click on object to select\n• Move selected objects with mouse"
        ttk.Label(interaction_frame, text=interaction_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Object list
        ttk.Label(self.controls_frame, text="Occlusion Objects:", font=("Arial", 12)).pack(anchor=tk.W, pady=(15, 5))
        
        # Scrollable frame for objects
        self.obj_frame = ttk.Frame(self.controls_frame)
        self.obj_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self.obj_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox
        self.obj_listbox = tk.Listbox(self.obj_frame, height=15, width=35, selectmode=tk.SINGLE)
        self.obj_listbox.pack(fill=tk.BOTH, expand=True)
        self.obj_listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.obj_listbox.yview)
        self.obj_listbox.bind('<<ListboxSelect>>', self.on_select_object)
        
        # Buttons frame
        btn_frame = ttk.Frame(self.controls_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Remove selected button
        self.remove_btn = ttk.Button(btn_frame, text="Remove Selected", command=self.remove_selected)
        self.remove_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear all button
        self.clear_btn = ttk.Button(btn_frame, text="Clear All", command=self.clear_objects)
        self.clear_btn.pack(side=tk.LEFT)
        
        # Face Detection Controls
        ttk.Label(self.controls_frame, text="Face Detection Settings", font=("Arial", 14, "bold")).pack(anchor=tk.W, pady=(20, 10))
        
        # Scale factor for detection
        scale_frame = ttk.Frame(self.controls_frame)
        scale_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scale_frame, text="Scale Factor:").pack(side=tk.LEFT, padx=(0, 10))
        self.scale_factor = tk.DoubleVar(value=1.1)
        self.scale_factor_entry = ttk.Spinbox(scale_frame, from_=1.01, to=1.5, increment=0.05, textvariable=self.scale_factor, width=5)
        self.scale_factor_entry.pack(side=tk.LEFT)
        
        # Minimum neighbors
        neighbors_frame = ttk.Frame(self.controls_frame)
        neighbors_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(neighbors_frame, text="Min Neighbors:").pack(side=tk.LEFT, padx=(0, 10))
        self.min_neighbors = tk.IntVar(value=5)
        self.min_neighbors_entry = ttk.Spinbox(neighbors_frame, from_=1, to=10, textvariable=self.min_neighbors, width=5)
        self.min_neighbors_entry.pack(side=tk.LEFT)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.controls_frame, textvariable=self.status_var, font=("Arial", 10))
        self.status_label.pack(anchor=tk.W, pady=(20, 0))
    
    def set_rectangle_mode(self):
        self.status_var.set("Rectangle drawing mode: click and drag to create")
    
    def import_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Create an image occlusion at the center of the canvas
                center_x = self.width // 4
                center_y = self.height // 4
                img_occlusion = ImageOcclusion(file_path, center_x, center_y)
                
                # Add to objects list
                self.occlusion_objects.append(img_occlusion)
                self.update_object_list()
                self.status_var.set(f"Added image: {os.path.basename(file_path)}")
                
                # Select the newly added image
                self.selected_object = img_occlusion
                self.obj_listbox.selection_clear(0, tk.END)
                self.obj_listbox.selection_set(len(self.occlusion_objects) - 1)
                
            except Exception as e:
                self.status_var.set(f"Error loading image: {str(e)}")
    
    def set_color(self, color):
        self.current_color = color
        
        # Update color indicator
        color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        self.color_indicator.config(bg=color_hex)
    
    def choose_custom_color(self):
        color = colorchooser.askcolor(title="Choose Color")
        if color[0] is not None:
            # Convert RGB floats to RGB integers
            rgb_color = tuple(int(c) for c in color[0])
            self.set_color(rgb_color)



    def update_object_list(self):
        self.obj_listbox.delete(0, tk.END)
        for i, obj in enumerate(self.occlusion_objects):
            self.obj_listbox.insert(tk.END, f"Object {i+1}: {obj}")
    
    def on_select_object(self, event):
        selection = self.obj_listbox.curselection()
        if selection:
            idx = selection[0]
            if 0 <= idx < len(self.occlusion_objects):
                self.selected_object = self.occlusion_objects[idx]
                self.status_var.set(f"Selected: Object {idx+1}")
            else:
                self.selected_object = None
        else:
            self.selected_object = None
    
    def remove_selected(self):
        if self.selected_object is not None:
            self.occlusion_objects.remove(self.selected_object)
            self.selected_object = None
            self.update_object_list()
            self.status_var.set("Object removed")
    
    def clear_objects(self):
        self.occlusion_objects.clear()
        self.selected_object = None
        self.update_object_list()
        self.status_var.set("All objects cleared")
    
    def on_mouse_down(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
        # Check if clicked on an existing object
        for obj in reversed(self.occlusion_objects):  # Check from front to back
            if obj.contains_point(event.x, event.y):
                # Select this object
                idx = self.occlusion_objects.index(obj)
                self.obj_listbox.selection_clear(0, tk.END)
                self.obj_listbox.selection_set(idx)
                self.obj_listbox.see(idx)
                self.on_select_object(None)
                self.status_var.set(f"Selected: Object {idx+1}")
                self.selected_object = obj
                self.moving = True
                return
        
        # If not on an existing object, start drawing a new rectangle
        self.drawing = True
        self.moving = False
        self.start_x = event.x
        self.start_y = event.y
        self.current_rectangle = Rectangle(event.x, event.y, event.x, event.y, self.current_color)
    
    def on_mouse_move(self, event):
        if self.moving and self.selected_object:
            # Calculate the movement delta
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            
            # Move the selected object
            self.selected_object.move(dx, dy)
            
            # Update last mouse position
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            
        elif self.drawing and self.current_rectangle:
            self.current_rectangle.x2 = event.x
            self.current_rectangle.y2 = event.y
    
    def on_mouse_up(self, event):
        if self.moving:
            self.moving = False
            self.status_var.set(f"Moved object")
        
        elif self.drawing and self.current_rectangle:
            # Finalize the rectangle - adjust min/max coordinates
            self.current_rectangle.x1 = min(self.start_x, event.x)
            self.current_rectangle.y1 = min(self.start_y, event.y)
            self.current_rectangle.x2 = max(self.start_x, event.x)
            self.current_rectangle.y2 = max(self.start_y, event.y)
            
            # Only add if it has some size
            if (self.current_rectangle.x2 - self.current_rectangle.x1) > 5 and \
               (self.current_rectangle.y2 - self.current_rectangle.y1) > 5:
                self.occlusion_objects.append(self.current_rectangle)
                self.update_object_list()
                self.status_var.set(f"Added new rectangle")
        
        self.drawing = False
        self.current_rectangle = None
    
    def detect_faces(self, frame):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor.get(),
            minNeighbors=self.min_neighbors.get(),
            minSize=(30, 30)
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return frame, len(faces)
    
    def update(self):
        # Capture frame
        ret, frame = self.cap.read()
        
        if ret:
            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Create a copy of the frame for occlusions
            occluded_frame = frame.copy()
            
            # Draw all objects
            for obj in self.occlusion_objects:
                obj.draw(occluded_frame)
            
            # Draw current rectangle if drawing
            if self.drawing and self.current_rectangle:
                self.current_rectangle.draw(occluded_frame)
            
            # Highlight selected object
            if self.selected_object in self.occlusion_objects:
                if isinstance(self.selected_object, Rectangle):
                    # Draw a yellow border around the selected rectangle
                    cv2.rectangle(occluded_frame, 
                                (self.selected_object.x1, self.selected_object.y1),
                                (self.selected_object.x2, self.selected_object.y2),
                                (0, 255, 255), 2)
                elif isinstance(self.selected_object, ImageOcclusion):
                    # Draw a yellow border around the selected image
                    cv2.rectangle(occluded_frame,
                                (self.selected_object.x, self.selected_object.y),
                                (self.selected_object.x + self.selected_object.width, 
                                 self.selected_object.y + self.selected_object.height),
                                (0, 255, 255), 2)
            
            # Detect faces on the occluded frame
            result_frame, face_count = self.detect_faces(occluded_frame)
            
            # Update status with face count
            if not self.drawing and not self.moving:
                self.status_var.set(f"Detected {face_count} face{'s' if face_count != 1 else ''}")
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.canvas.config(width=rgb_frame.shape[1], height=rgb_frame.shape[0])
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk  # Keep a reference to prevent garbage collection
        
        # Schedule the next update
        self.window.after(15, self.update)
    
    def on_close(self):
        # Release the camera
        if self.cap.isOpened():
            self.cap.release()
        
        # Close the window
        self.window.destroy()


def main():
    # Create the main window
    root = tk.Tk()
    
    # Create the application
    app = FaceOcclusionApp(root, "Face Occlusion Experiment")
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()