from PIL import Image, ImageOps
import tkinter as tk
from tkinter import filedialog

def invert_image_colors(image_path):
    # Open the image
    image = Image.open(image_path)
    
    # Invert the colors
    inverted_image = ImageOps.invert(image)
    
    # Save the inverted image in the same directory
    inverted_image_path = image_path.replace('.', '_inverted.')
    inverted_image.save(inverted_image_path)
    
    print(f"Inverted image saved as {inverted_image_path}")

def select_image():
    # Open file explorer to select an image
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    
    if file_path:
        invert_image_colors(file_path)
    else:
        print("No image selected.")

if __name__ == "__main__":
    select_image()