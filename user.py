import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import sv_ttk

## Choose which Env to Run by uncommenting relevant import statement

#from inception_df.query import Nearest_images
from vgg_df.query_vgg import Nearest_images
#from fashion200.query_200 import Nearest_images
#from inception_df_color.query import Nearest_images

class ImageGridViewer:
    def __init__(self, root, final_nums):
        self.root = root
        self.root.title("Image Grid Viewer")
        self.final_nums = final_nums
        self.image_frames = []

    def add_image_frame(self, image_path):
        image = Image.open(image_path)

#Variant 1
        new_height = image.height
        new_width = image.width
        while new_width > 150:
            new_width //=2
            new_height //=2

#Variant2
        # new_height = 300
        # new_width = int(image.width * new_height / image.height)

        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        image_frame = tk.Label(self.root, image=photo, highlightthickness=1, highlightbackground="black")
        image_frame.image = photo  # Keep a reference to the image to prevent it from being garbage collected
        # image_frame.grid(row= int(len(self.image_frames)/3) + 3, column=len(self.image_frames)%3)
        image_frame.place(x=157*int(len(self.image_frames)%5) + 112, y=112 + 247 * int(len(self.image_frames)/5))
        
        self.image_frames.append(image_frame)

def process_queries(text_query, image_path, w):
    # Add your processing logic here based on the text query and image path
    print("Text Query:", text_query)
    print("Image Path:", image_path)
    print("Weight : ", w)

    if  w is None or w == "":
        return Nearest_images(image_path, text_query)
    return Nearest_images(image_path, text_query, int(w))


def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png"),("Image files", "*.jpeg"),("Image files", "*.jpg") ])
    if file_path:
        image = Image.open(file_path)

        new_width = image.width
        new_height = image.height

        while new_width > 250:
            new_width //= 1.1
            new_height //= 1.1

        image = image.resize((int(new_width), int(new_height)), Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(image)
        image_frame = tk.Label(root, image=photo, border=8, highlightthickness=2, highlightbackground="black")
        image_frame.image = photo
        image_frame.place(x=975, y = 112)
        update_display(process_queries(text_entry.get(), file_path, w_entry.get()))

def update_display(image_list):
    # Update the GUI to display the processed images
    viewer = ImageGridViewer(root, 6)
    for image_file in image_list:
        viewer.add_image_frame(image_file)

# GUI setup
root = tk.Tk()

sv_ttk.set_theme("light")
root.title("Text and Image Query Processing")

# background_image = Image.open("bg.jpeg")
# background_photo = ImageTk.PhotoImage(background_image)

# # Create a label for the background image
# background_label = tk.Label(root, image=background_photo)
# background_label.place(relwidth=1, relheight=1)

width = 720
height = 540
root.geometry(f"{width}x{height}")

label_frame = tk.Frame(root)
# label_frame.pack(side=tk.LEFT, padx=20, pady=100)
label_frame.place(x=187, y=15)

my_font1=('century schoolbook l', 17, 'bold')
frame_label = tk.Label(label_frame, text="Multi-Modal Search", font=my_font1, bg="#f0f0f0", fg="black", bd=0, highlightbackground="#f0f0f0", highlightcolor="#f0f0f0")
frame_label.pack(side=tk.LEFT, padx=5, pady=5)

query_frame = tk.Frame(root)
query_frame.place(x=600, y=15)

# Label for the text query
my_font1=('century schoolbook l', 15, 'bold')
text_query_label = tk.Label(query_frame, text="Query Text : ", font=my_font1, bg="#f0f0f0", fg="black", bd=0, highlightbackground="#f0f0f0", highlightcolor="#f0f0f0")
text_query_label.pack(side="left", padx=12, pady=7)

# Entry widget for the text query
text_entry = tk.Entry(query_frame, width=25)
text_entry.pack(side="left")

w_frame = tk.Frame(root)
w_frame.place(x=900, y=15)

# Label for the text query
my_font1=('century schoolbook l', 15, 'bold')
w_label = tk.Label(w_frame, text="Weight to text : ", font=my_font1, bg="#f0f0f0", fg="black", bd=0, highlightbackground="#f0f0f0", highlightcolor="#f0f0f0")
w_label.pack(side="left", padx=12, pady=7)

w_entry = tk.Entry(w_frame, width=7)
w_entry.pack(side="left")



# Open File button to trigger the file dialog
open_file_button = tk.Button(root, text="Open Image", font=my_font1,command=open_file_dialog, borderwidth=5)
open_file_button.place(x=375, y=15)
# open_file_button.pack(padx=10, pady=10)

# Display images label
display_label = tk.Label(root, text="Retrieved Images:", font=my_font1)
display_label.place(x=375, y=75)

display_label = tk.Label(root, text="Query Image:", font=my_font1)
display_label.place(x=975, y=75)

root.mainloop()