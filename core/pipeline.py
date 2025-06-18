# one_shot_object_detection/core/pipeline.py
from typing import List
from .image_handler import ImageHandler
from .detector import OWLViTDetector

class DetectionPipeline:
    """Orchestrates the detection workflow."""
    def __init__(self):
        self.detector = OWLViTDetector()
        self.image_handler = ImageHandler()

    def process_same_image_detection(self, image_path: str, bbox: list, threshold: float = 0.1) -> tuple:
        """Find similar objects within the same image."""
        image = self.image_handler.load_image(image_path)
        query_image = self.image_handler.crop_bbox_region(image, bbox)

        results = self.detector.detect_similar_objects(
            target_image=image,
            query_image=query_image,
            threshold=threshold
        )
        # In pipeline.py, after results = self.detector.detect_similar_objects(...)
        if results["scores"].numel() == 0:  # Check if any detections were made
            print("No objects detected")
            return original_image, []  # Return empty results
        return image, results

    def process_cross_image_detection(self, reference_image_path: str, reference_bbox: list,
                                      target_image_paths: List[str], threshold: float = 0.1) -> dict:
        """Find similar objects across a list of different images."""
        ref_image = self.image_handler.load_image(reference_image_path)
        query_image = self.image_handler.crop_bbox_region(ref_image, reference_bbox)

        all_results = {}
        for target_path in target_image_paths:
            target_image = self.image_handler.load_image(target_path)
            detection_results = self.detector.detect_similar_objects(
                target_image=target_image,
                query_image=query_image,
                threshold=threshold
            )
            all_results[target_path] = (target_image, detection_results)

        return all_results
    def process_text_prompt(self, image_path: str, query_text: str, threshold: float = 0.1) -> tuple:
        """Find objects in an image using a text prompt."""
        image = self.image_handler.load_image(image_path)
        results = self.detector.detect_from_text(image, query_text, threshold=threshold)
        return image, results

