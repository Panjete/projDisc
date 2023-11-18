import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
#import sv_ttk


#from inception_df.query import Nearest_images
#from vgg_df.query_vgg import Nearest_images
#from fashion200.query_200 import Nearest_images

from inception_df_color.query import Nearest_images

class ImageGridViewer:
    def __init__(self, root, final_nums):
        self.root = root
        self.root.title("Image Grid Viewer")
        self.final_nums = final_nums
        self.image_frames = []

    def add_image_frame(self, image_path):
        image = Image.open(image_path)

        image = image.resize((150, 250), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        image_frame = tk.Label(self.root, image=photo)
        image_frame.image = photo  # Keep a reference to the image to prevent it from being garbage collected
        image_frame.place(x=175*int(len(self.image_frames)%5) + 150, y=150 + 270 * int(len(self.image_frames)/5))
        
        self.image_frames.append(image_frame)

def process_queries(text_query, image_path):
    # Add your processing logic here based on the text query and image path
    print("Text Query:", text_query)
    print("Image Path:", image_path)

    return Nearest_images(image_path, text_query)

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png"),("Image files", "*.jpeg") ])
    if file_path:
        update_display(process_queries(text_entry.get(), file_path))

def update_display(image_list):
    # Update the GUI to display the processed images
    viewer = ImageGridViewer(root, 6)
    for image_file in image_list:
        viewer.add_image_frame(image_file)

# GUI setup
root = tk.Tk()

#sv_ttk.set_theme("light")
root.title("Text and Image Query Processing")


width = 1080
height = 720
root.geometry(f"{width}x{height}")

label_frame = tk.Frame(root)
label_frame.place(x=250, y=20)

my_font1=('century schoolbook l', 20, 'bold')
frame_label = tk.Label(label_frame, text="COL764 Project", font=my_font1, bg="#f0f0f0", fg="black", bd=0, highlightbackground="#f0f0f0", highlightcolor="#f0f0f0")
frame_label.pack(side=tk.LEFT, padx=5, pady=5)

query_frame = tk.Frame(root)
query_frame.place(x=800, y=20)

# Label for the text query
my_font1=('century schoolbook l', 15, 'bold')
text_query_label = tk.Label(query_frame, text="Query Text : ", font=my_font1, bg="#f0f0f0", fg="black", bd=0, highlightbackground="#f0f0f0", highlightcolor="#f0f0f0")
text_query_label.pack(side="left", padx=15, pady=10)

# Entry widget for the text query
text_entry = tk.Entry(query_frame, width=30)
text_entry.pack(side="left")

# Open File button to trigger the file dialog
open_file_button = tk.Button(root, text="Open Image", font=my_font1,command=open_file_dialog, borderwidth=5)
open_file_button.place(x=500, y=20)

# Display images label
display_label = tk.Label(root, text="Retrieved Images:", font=my_font1)
display_label.place(x=500, y=100)

root.mainloop()
