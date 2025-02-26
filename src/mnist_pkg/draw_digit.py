from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Event, Frame, Label
from PIL import Image, ImageDraw

import numpy as np
import torch

from mnist_pkg.cnn_model import Net
from mnist_pkg.constants import MODELS_PATH
from mnist_pkg.utils import load_model


class DigitDrawer:
    predict_timer: int = 0
    delay: int = 1000
    lastx: int
    lasty: int
    canvas: Canvas
    model: Net
    guess: str = "Guess: -"

    def _save_position(self, event: Event) -> None:
        self.lastx, self.lasty = event.x, event.y

    def _clear_canvas(self, event) -> None:
        self.canvas.delete("all")
        self.guess_label.config(text="Guess: -")

    def _add_line(self, event: Event) -> None:
        self.canvas.create_line(
            self.lastx,
            self.lasty,
            event.x,
            event.y,
            fill="white",
            width=20,
            capstyle=tk.ROUND,
            smooth=True,
            splinesteps=36,
        )
        self._save_position(event)

        if self.predict_timer:
            try:
                self.root.after_cancel(self.predict_timer)
            except ValueError:
                pass

        self.predict_timer = self.root.after(self.delay, self._predict_digit)

    def _predict_digit(self) -> None:
        image = Image.new("L", (280, 280), color=0)
        draw = ImageDraw.Draw(image)

        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            draw.line(coords, fill=255, width=20)

        image = image.resize((28, 28))
        image_array = np.array(image)
        image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor / 255.0

        # image_tensor = image_tensor / 255.0
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output)
            self.guess = f"Guess: {str(prediction.item())}"
            self.guess_label.config(text=self.guess)

    def __init__(self) -> None:
        if not Path(MODELS_PATH / "best_model.pth").exists():
            raise FileNotFoundError(
                "The trained model is missing. "
                "Please run model_train() in train_model.py "
                "to generate the best model."
            )

        self.root = tk.Tk()
        self.root.title("Draw a digit")

        # Load the model
        self.model = Net()
        opt = torch.optim.Adam(self.model.parameters())
        load_model(self.model, opt, MODELS_PATH / "best_model.pth")
        self.model.eval()

        # Canvas
        self.canvas = Canvas(self.root, width=280, height=280, background="black")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self._save_position)
        self.canvas.bind("<B1-Motion>", self._add_line)

        # Make canvas focusable
        self.canvas.focus_set()  # Give canvas focus
        self.canvas.focus_force()  # Force focus

        # Clear canvas on C/c press
        self.root.bind("C", self._clear_canvas)
        self.root.bind("c", self._clear_canvas)

        # Button frame
        button_frame = Frame(self.root)
        button_frame.grid(row=1, column=0, pady=10)

        # Buttons
        clear_label = Label(button_frame, text="Press C to clear.")
        clear_label.grid(row=0, column=0, padx=0)

        self.guess_label = Label(button_frame, text=self.guess)
        self.guess_label.grid(row=0, column=2, padx=20)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = DigitDrawer()
    app.run()
