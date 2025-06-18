# one_shot_object_detection/ui/visualizer.py
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

class ResultsVisualizer:
    """Handles drawing and displaying detection results."""
    def __init__(self):
        try:
            self.font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            self.font = ImageFont.load_default()

    def draw_detections(self, image: Image.Image, results: dict, color: str = 'red', width: int = 3) -> Image.Image:
        """Draw bounding boxes and scores on an image."""
        if "boxes" not in results or "scores" not in results:
            return image
        
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        boxes = results["boxes"].tolist()
        scores = results["scores"].tolist()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
            draw.text((x1, y1 - 20), f"Score: {score:.2f}", fill=color, font=self.font)
        return img_draw

    def display_image(self, image: Image.Image, title: str):
        """Display a single image."""
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def display_results_grid(self, images_with_titles: list):
        """Display multiple images with their titles in a grid."""
        n_images = len(images_with_titles)
        if n_images == 0:
            return
            
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (image, title) in enumerate(images_with_titles):
            axes[i].imshow(image)
            axes[i].set_title(title)
            axes[i].axis('off')
        
        # Turn off unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()
