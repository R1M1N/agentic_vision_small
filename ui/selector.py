# one_shot_object_detection/ui/selector.py
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

class BoundingBoxSelector:
    """An interactive UI for selecting a bounding box on an image."""
    def __init__(self, image: Image.Image):
        self.image = image
        self.bbox = None
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.selector = None

    def _onselect(self, eclick, erelease):
        """Callback for when the rectangle is selected."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        print(f"Selected BBox: {self.bbox}")
        plt.close(self.fig)

    def select_bbox(self) -> list:
        """Display the image and activate the selector."""
        self.ax.imshow(self.image)
        self.ax.set_title("Draw a bounding box around the object of interest, then close this window.")
        self.ax.axis('off')
        
        self.selector = RectangleSelector(
            self.ax, self._onselect, useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
        
        plt.show(block=True)
        return self.bbox
