import tkinter as tk
from tkinter import Button, Label, Tk, filedialog
from classification_emotions import Classifictions
import cv2
from model import Model
from PIL import Image, ImageTk

# Initialize the model
model = Model.create_model()
model.load_weights('./model_2022-01-08.h5')


class App:
    def __init__(self):
        self.window = Tk()
        self.window.geometry('1200x850+150+150')
        self.window.title("Emotion Detection")

        # Buttons
        self._exit_btn = Button(
            self.window,
            text="Exit",
            width=20,
            command=self.window.quit,
            highlightbackground='lightgray'
        )
        self._select_btn = Button(
            self.window,
            text="Upload Image",
            command=self._load_image,
            width=20,
            highlightbackground='lightgray'
        )
        self._live_btn = Button(
            self.window,
            text="Live Detection",
            command=self._live_func,
            width=20,
            highlightbackground='lightgray'
        )

        # Positioning Buttons
        self._select_btn.place(x=580, y=650, anchor=tk.N)
        self._live_btn.place(x=580, y=680, anchor=tk.N)
        self._exit_btn.place(x=580, y=710, anchor=tk.N)

        # Frame for video and image display
        self.video_frame = tk.Label(self.window, text="Upload Image or Start Live Detection", bg="grey")
        self.video_frame.place(relx=0.5, rely=0.4, anchor='center')

        # Initialize video capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Use AVFoundation for macOS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        self.window.mainloop()

    def live_face_expression(self):
        # Capture a frame
        ret, img = self.cap.read()

        # Check if the frame was successfully captured
        if not ret or img is None:
            print("Failed to capture frame from camera.")
            return

        # Flip the image horizontally for a mirror effect
        img = cv2.flip(img, 1)

        # Detect and classify emotion
        Classifictions.get_expression_classified(img, model)

        # Convert the image for Tkinter display
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

        # Schedule the next frame
        self.video_frame.after(10, self.live_face_expression)

    def _load_image(self):
        # Allow user to upload an image
        filenames = filedialog.askopenfilenames(
            title="Choose a file",
            filetypes=[("Image Files", "*.png;*.jpg")]
        )

        if filenames:
            filename = filenames[0]
            if filename != '':
                img = cv2.imread(filename)
                if img is None:
                    print("Unable to load the selected image.")
                    return

                # Resize the image for better display
                scale = 100
                while scale * img.shape[1] / 100 >= 800 or scale * img.shape[0] / 100 >= 600:
                    scale -= 1
                width = int(img.shape[1] * scale / 100)
                height = int(img.shape[0] * scale / 100)
                img = cv2.resize(img, (width, height))

                # Detect and classify emotion
                Classifictions.get_expression_classified(img, model)

                # Convert the image for Tkinter display
                cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
            else:
                print("No image selected!")
        else:
            print("Choose an Image!")

    def _live_func(self):
        # Start live detection
        if not self.cap.isOpened():
            print("Camera is not accessible.")
            return
        self.live_face_expression()


if __name__ == '__main__':
    App()