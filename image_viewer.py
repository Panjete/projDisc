import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, Label
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ImageGridViewer:
    def __init__(self, root, final_nums):
        self.root = root
        self.root.title("Image Grid Viewer")
        self.final_nums = final_nums
        self.image_frames = []

    def add_image_frame(self, image_path):
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)

        image_frame = tk.Label(self.root, image=photo)
        image_frame.image = photo  # Keep a reference to the image to prevent it from being garbage collected
        image_frame.grid(row= 1, column=len(self.image_frames))

        self.image_frames.append(image_frame)

def main_viewer(image_files):
    root = tk.Tk()
    root.resizable(width = True, height = True)
    viewer = ImageGridViewer(root, 6)
    # Example list of image file paths
    for image_file in image_files:
        viewer.add_image_frame(image_file)

    root.mainloop()


def mv2(image_paths):
    window = tk.Tk()
    window.title("Image Viewer")

    photo_images = []
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        photo_images.append(photo)

    #for i, photo in enumerate(photo_images):
        label = Label(window, image=photo)
        label.grid(row=0, column=i)

    # Start the Tkinter main loop
    window.mainloop()


def mv3(image_paths):

    # Create a figure with subplots
    num_images = len(image_paths)
    rows, cols = 1, num_images
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3))

    # Load and display each image
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        axes[i].imshow(img)
        axes[i].axis('off')

    # Show the images
    plt.show()

if __name__ == "__main__":
    main_viewer(['photos/2.png', 'photos/3.png', 'photos/4.png'])
    