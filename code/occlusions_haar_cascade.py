import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, colorchooser
from PIL import Image, ImageTk
import uuid

class Rectangle:
    def __init__(self, x1, y1, x2, y2, color=(0, 0, 0)):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.color = color
        self.id = str(uuid.uuid4())
    
    def contains_point(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def draw(self, frame):
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), self.color, -1)
    
    def __str__(self):
        color_name = "Black" if self.color == (0, 0, 0) else "White" if self.color == (255, 255, 255) else f"RGB{self.color}"
        return f"{color_name} ({self.x2-self.x1}x{self.y2-self.y1})"


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
        
        # Drawing variables
        self.rectangles = []
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rectangle = None
        self.selected_rectangle = None
        self.current_color = (0, 0, 0)  # Default black
        
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
        
        # Rectangle list
        ttk.Label(self.controls_frame, text="Occlusion Rectangles:", font=("Arial", 12)).pack(anchor=tk.W, pady=(15, 5))
        
        # Scrollable frame for rectangles
        self.rect_frame = ttk.Frame(self.controls_frame)
        self.rect_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self.rect_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox
        self.rect_listbox = tk.Listbox(self.rect_frame, height=15, width=35, selectmode=tk.SINGLE)
        self.rect_listbox.pack(fill=tk.BOTH, expand=True)
        self.rect_listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.rect_listbox.yview)
        self.rect_listbox.bind('<<ListboxSelect>>', self.on_select_rectangle)
        
        # Buttons frame
        btn_frame = ttk.Frame(self.controls_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Remove selected button
        self.remove_btn = ttk.Button(btn_frame, text="Remove Selected", command=self.remove_selected)
        self.remove_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear all button
        self.clear_btn = ttk.Button(btn_frame, text="Clear All", command=self.clear_rectangles)
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
    
    def update_rectangle_list(self):
        self.rect_listbox.delete(0, tk.END)
        for i, rect in enumerate(self.rectangles):
            self.rect_listbox.insert(tk.END, f"Rectangle {i+1}: {rect}")
    
    def on_select_rectangle(self, event):
        selection = self.rect_listbox.curselection()
        if selection:
            idx = selection[0]
            if 0 <= idx < len(self.rectangles):
                self.selected_rectangle = self.rectangles[idx]
                self.status_var.set(f"Selected: Rectangle {idx+1}")
            else:
                self.selected_rectangle = None
        else:
            self.selected_rectangle = None
    
    def remove_selected(self):
        if self.selected_rectangle is not None:
            self.rectangles.remove(self.selected_rectangle)
            self.selected_rectangle = None
            self.update_rectangle_list()
            self.status_var.set("Rectangle removed")
    
    def clear_rectangles(self):
        self.rectangles.clear()
        self.selected_rectangle = None
        self.update_rectangle_list()
        self.status_var.set("All rectangles cleared")
    
    def on_mouse_down(self, event):
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        
        # Check if clicked on an existing rectangle
        for rect in reversed(self.rectangles):  # Check from front to back
            if rect.contains_point(event.x, event.y):
                # Select this rectangle
                idx = self.rectangles.index(rect)
                self.rect_listbox.selection_clear(0, tk.END)
                self.rect_listbox.selection_set(idx)
                self.rect_listbox.see(idx)
                self.on_select_rectangle(None)
                self.status_var.set(f"Selected: Rectangle {idx+1}")
                return
        
        # If not on an existing rectangle, start drawing a new one
        self.current_rectangle = Rectangle(event.x, event.y, event.x, event.y, self.current_color)
    
    def on_mouse_move(self, event):
        if self.drawing and self.current_rectangle:
            self.current_rectangle.x2 = event.x
            self.current_rectangle.y2 = event.y
    
    def on_mouse_up(self, event):
        if self.drawing and self.current_rectangle:
            # Finalize the rectangle - adjust min/max coordinates
            self.current_rectangle.x1 = min(self.start_x, event.x)
            self.current_rectangle.y1 = min(self.start_y, event.y)
            self.current_rectangle.x2 = max(self.start_x, event.x)
            self.current_rectangle.y2 = max(self.start_y, event.y)
            
            # Only add if it has some size
            if (self.current_rectangle.x2 - self.current_rectangle.x1) > 5 and \
               (self.current_rectangle.y2 - self.current_rectangle.y1) > 5:
                self.rectangles.append(self.current_rectangle)
                self.update_rectangle_list()
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
            
            # Draw all rectangles
            for rect in self.rectangles:
                rect.draw(occluded_frame)
            
            # Draw current rectangle if drawing
            if self.drawing and self.current_rectangle:
                self.current_rectangle.draw(occluded_frame)
            
            # Highlight selected rectangle
            if self.selected_rectangle in self.rectangles:
                cv2.rectangle(occluded_frame, 
                              (self.selected_rectangle.x1, self.selected_rectangle.y1),
                              (self.selected_rectangle.x2, self.selected_rectangle.y2),
                              (0, 255, 255), 2)
            
            # Detect faces on the occluded frame
            result_frame, face_count = self.detect_faces(occluded_frame)
            
            # Update status with face count
            self.status_var.set(f"Detected {face_count} face{'s' if face_count != 1 else ''}")
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
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