# one_shot_object_detection/core/image_handler.py
from PIL import Image

class ImageHandler:
    """Handles loading and cropping of images."""
    def __init__(self):
        self.current_image = None

    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from a file path."""
        self.current_image = Image.open(image_path).convert("RGB")
        return self.current_image

    def crop_bbox_region(self, image: Image.Image, bbox: list) -> Image.Image:
        """Crop a bounding box region from an image."""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        return image.crop((x1, y1, x2, y2))
