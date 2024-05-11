import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk 
from tkinter import font
import cv2
import numpy as np 
import random 
import tensorflow as tf 
from keras.models import load_model
import tensorflow_hub as hub
import tkinter.messagebox as messagebox
import time 

class ImageAugmenterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Leprosy Detection AI Model")
        self.bold_font = font.Font(weight="bold")
        self.image_path = ""
        self.original_image = None
        self.modified_image = None
        self.create_widgets()
    

    def load_model(self):
        self.model = tf.keras.models.load_model('new_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        self.featwts = self.model.layers[0].get_weights()[0][0,0]
        messagebox.showinfo("Success", "Model loaded successfully!")

    def create_widgets(self):
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        left_frame = tk.Frame(self.master, width=400, height=500, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.model_button = tk.Button(left_frame, bg='#8FBCBB', text="Load Model", command=self.load_model, height=2, width=20, font=self.bold_font)
        self.model_button.pack(padx=40, pady=(90, 20))  # Adjusted pady for the model button
            
        self.add_image_button = tk.Button(left_frame, bg='#8FBCBB', text="Load Image", command=self.load_image, height=2, width=20, font=self.bold_font)
        self.add_image_button.pack(padx=40, pady=20)  # Adjusted pady for the add image button

        self.predict_button = tk.Button(left_frame, bg='#8FBCBB', text="Predict", command=self.predict, height=2, width=20, font=self.bold_font)
        self.predict_button.pack(padx=40, pady=(20, 90))  # Adjusted pady for the predict button

        right_frame = tk.Frame(self.master, width=600, height=400)
        right_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        real_canvas_label = tk.Label(right_frame, text="Original Image",  font=self.bold_font)
        real_canvas_label.grid(row=0, column=0, padx=10, pady=10)  # Adjusted padx and pady for the label
        
        grayscale_canvas_label = tk.Label(right_frame, text="Grayscale Image", font=self.bold_font)
        grayscale_canvas_label.grid(row=0, column=1, padx=10, pady=10)  #    Adjusted padx and pady for the label
        
        processed_canvas_label = tk.Label(right_frame, text="Augmented Image", font=self.bold_font)
        processed_canvas_label.grid(row=2, column=0, padx=10, pady=10)  # Adjusted padx and pady for the label
        
        bw_canvas_label = tk.Label(right_frame, text="Detection",  font=self.bold_font)
        bw_canvas_label.grid(row=2, column=1, padx=10, pady=10)  # Adjusted padx and pady for the label

        self.real_canvas = tk.Canvas(right_frame, width=300, height=300, bg='#f0f0f0')
        self.real_canvas.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.grayscale_canvas = tk.Canvas(right_frame, width=300, height=300, bg='#f0f0f0')
        self.grayscale_canvas.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        self.processed_canvas = tk.Canvas(right_frame, width=300, height=300, bg='#f0f0f0')
        self.processed_canvas.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')

        self.bw_canvas = tk.Canvas(right_frame, width=300, height=300, bg='#f0f0f0')
        self.bw_canvas.grid(row=3, column=1, padx=10, pady=10, sticky='nsew')

        self.prediction_label = tk.Label(right_frame, text="", font=self.bold_font)
        self.prediction_label.grid(row=5, column=0, padx=10, pady=10, sticky='nsew')


        for i in range(6):  
            right_frame.grid_rowconfigure(i, weight=1)

        for i in range(2):
            right_frame.grid_columnconfigure(i, weight=1)


    def convert_to_grayscale(self, img_array):
        if len(img_array.shape) == 2:
            return img_array 

        grayscale_image_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return grayscale_image_array


    def display_grayscale_image(self, image):
        grayscale_image = self.convert_to_grayscale(image)
        self.display_image(grayscale_image, self.grayscale_canvas)    

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if self.image_path:
            self.original_image = Image.open(self.image_path)
            self.display_image(self.original_image, self.real_canvas)

    def display_image(self, image, canvas):
        image_array = np.array(image)
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        desired_size = min(screen_width, screen_height) // 4
        resized_array = cv2.resize(image_array, (desired_size, desired_size))
        resized_image = Image.fromarray(resized_array)
        photo = ImageTk.PhotoImage(image=resized_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo

    def display_bw_image(self, image, canvas):
        image_array = np.array(image)
        image_array = ((image_array - image_array.min()) * (1/(image_array.max() - image_array.min()) * 255)).astype('uint8')
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        desired_size = min(screen_width, screen_height) // 4
        image = Image.fromarray(image_array)
        resized_image = image.resize((desired_size, desired_size))
        photo = ImageTk.PhotoImage(image=resized_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo


    def random_saturation(self, img, saturation_factor):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation_factor, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    def binarize_image(self):
        if self.original_image:
            cv_image = np.array(self.original_image)
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
            
            # Binarization using Otsu's thresholding
            _, binarized_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            self.modified_image = Image.fromarray(binarized_image)
            self.display_image(self.modified_image, self.processed_canvas)

    def random_augmentation_values(self):
        saturation_factor = random.uniform(0.5, 2.0)  
        rotation_angle = random.uniform(-30, 30)   
        zoom_factor = random.uniform(0.7, 1.3)       
        return saturation_factor, rotation_angle, zoom_factor

    def random_crop(self, img):
        original_height, original_width = img.shape[:2]
        crop_width = int(original_width * 0.8)
        crop_height = int(original_height * 0.8)
        start_x = (original_width - crop_width) // 2
        start_y = (original_height - crop_height) // 2
        cropped_img = img[start_y:start_y+crop_height, start_x:start_x+crop_width]
        return cropped_img

    def output_image(self, single_img, stride=1):
        lo_reshaped =  self.featwts[:16, :16]
        print(lo_reshaped.shape)
        img_height, img_width, num_channels = single_img.shape
        lo_height, lo_width = lo_reshaped.shape
        out_height = (img_height - lo_height) // stride + 1
        out_width = (img_width - lo_width) // stride + 1
        feature_map = np.zeros((out_height, out_width, num_channels))
        for c in range(num_channels):
            for i in range(0, img_height - lo_height + 1, stride):
                for j in range(0, img_width - lo_width + 1, stride):
                    roi = single_img[i:i+lo_height, j:j+lo_width, c]
                    feature_map[i//stride, j//stride, c] = np.sum(np.multiply(roi, lo_reshaped))
        return feature_map

    def predict(self):
        self.binarize_image()  # Using binarized image instead of augmented image
        time.sleep(2)
        img = Image.open(self.image_path)
        img = img.resize((128, 128))
        img_array = np.array(img)
        self.display_grayscale_image(self.convert_to_grayscale(img_array))
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        img_data = np.asarray(img) / 255.0
        img_data1 = cv2.GaussianBlur(img_data.squeeze(), (5, 5), 0)
        
        img_data = np.expand_dims(img_data1, axis=0)
        out = self.model.predict(img_data)[0, 0]
        output = self.output_image(img_data.squeeze())
        
        output_bw = output.astype(np.float16)
        output_bw = np.median(output_bw, axis=-1)
        
        if out < 0: 
            print("Leprosy")
            self.prediction_label.config(text="Leprosy")
        else: 
            print("Non Leprosy")
            self.prediction_label.config(text="Non Leprosy")
        self.display_bw_image(output_bw, self.bw_canvas)

                
def main():
    root = tk.Tk()
    root.configure(bg="#f0f0f0")
    app = ImageAugmenterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
