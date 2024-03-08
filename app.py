import tkinter as tk
import customtkinter as ctk 

from PIL import ImageTk

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

# Create the app
app = tk.Tk()
app.geometry("1200x700")
app.title("Birthday Image Generator") 
authorization_token = "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"

# Headline 
title = ctk.CTkLabel(app, text="Birthday Image Generator", padx=10, pady=40, height=90, width=50, text_color="black")
title.configure(font=("bold", 40))
title.pack()

# Left Input
head_left = ctk.CTkLabel(app, text="Fill your information", height=50, width=50, text_color="black")
head_left.place(x=50, y=110)
head_left.configure(font=("bold", 25))

date = ctk.CTkEntry(app, placeholder_text="Enter your Date", height=40, width=300, text_color="black", fg_color="white")
date.place(x=10, y=160)
date.configure(font=("Arial", 20))

month = ctk.CTkEntry(app, placeholder_text="Enter your Month", height=40, width=300, text_color="black", fg_color="white") 
month.place(x=10, y=210)
month.configure(font=("Arial", 20))

year = ctk.CTkEntry(app, placeholder_text="Enter your Year", height=40, width=300, text_color="black", fg_color="white") 
year.place(x=10, y=260)
year.configure(font=("Arial", 20))

resolution = ctk.CTkEntry(app, placeholder_text="Enter Resolution Do you want", height=40, width=300, 
                          text_color="black", fg_color="white") 
resolution.place(x=10, y=310)
resolution.configure(font=("Arial", 20))

# Show Picture
lmain = ctk.CTkLabel(app, height=512, width=512, text = None)
lmain.pack()

# Model
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, 
                                               use_auth_token = authorization_token) 
pipe.to(device) 

# Function Generate Picture
def generate(): 
    sumprompt = date.get() + " " + month.get() + " " + year.get() + ", " + resolution.get() + ", Animal"
    with autocast(device): 
        image = pipe(sumprompt, guidance_scale=8.5).images[0]
    
    # Right Input
    head_right = ctk.CTkLabel(app, text="If you want to save image \n Please fill filename and \n Click Save button", 
                              height=50, width=50, text_color="black")
    head_right.place(x=890, y=120)
    head_right.configure(font=("bold", 25))
    
    name_file = ctk.CTkEntry(app, placeholder_text="Enter your file name ", height=40, width=300, text_color="black", fg_color="white") 
    name_file.place(x=890, y=220)
    name_file.configure(font=("Arial", 20))

    def save_image():
        filename = name_file.get() 
        if filename:
            image.save(f"{filename}.png") 

    save_button = ctk.CTkButton(app, height=50, width=120, text_color="white", fg_color="blue", command=save_image) 
    save_button.configure(text="Save", font=("Arial", 20))
    save_button.place(x=970, y=270)
    
    # Send image for show
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

# Generate Button
trigger = ctk.CTkButton(app, height=50, width=120, text_color="white", fg_color="blue", command=generate) 
trigger.configure(text="Generate", font=("Arial", 20))
trigger.place(x=90, y=360) 

app.mainloop()