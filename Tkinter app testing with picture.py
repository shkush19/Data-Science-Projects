import tkinter as tk
from PIL import Image, ImageTk

def on_button_click():
    # Update the label text
    label.config(text="Here is your comparison of Ridge and Lasso with different alpha values, " + entry.get())
    
    # Open and display the image
    image = Image.open("S:\\Naresh IT\\30th April\\Self Learning\\comparision.jpg")  # Replace "path_to_your_image.jpg" with the path to your image file
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection
    image_label.pack()

# Create the main window
root = tk.Tk()
root.title("Simple Tkinter App")

# Create a label
label = tk.Label(root, text="Type 'Comparison'")
label.pack()

# Create an entry widget
entry = tk.Entry(root)
entry.pack()

# Create a button
button = tk.Button(root, text="Click Me", command=on_button_click)
button.pack()
root.mainloop()

