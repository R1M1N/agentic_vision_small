import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class OWLViTDetector:
    def __init__(self, model_name: str = "google/owlvit-base-patch32"):
        print("Loading OWL-ViT model...")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("Model loaded successfully.")

    def detect_similar_objects(self, target_image: Image.Image, query_image: Image.Image,
                              threshold: float = 0.1, nms_threshold: float = 0.3) -> dict:
        """Detect objects in a target image that are similar to a query image."""
        inputs = self.processor(
            images=target_image,
            query_images=query_image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.image_guided_detection(**inputs)

        target_sizes = torch.tensor([target_image.size[::-1]]).to(self.device)
        processed_outputs = self.processor.post_process_image_guided_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            nms_threshold=nms_threshold
        )[0]

        if processed_outputs is None:
            return {
                "scores": torch.tensor([]),
                "labels": torch.tensor([], dtype=torch.long),
                "boxes": torch.tensor([])
            }

        return {
            "scores": processed_outputs["scores"].cpu() if processed_outputs["scores"] is not None else torch.tensor([]),
            "labels": processed_outputs["labels"].cpu() if processed_outputs.get("labels") is not None else torch.tensor([], dtype=torch.long),
            "boxes": processed_outputs["boxes"].cpu() if processed_outputs["boxes"] is not None else torch.tensor([])
        }

    def detect_from_text(self, target_image: Image.Image, query_text: str,
                        threshold: float = 0.1, nms_threshold: float = 0.3) -> dict:
        """Detect objects in a target image using a text prompt."""
        inputs = self.processor(
            text=query_text,
            images=target_image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([target_image.size[::-1]]).to(self.device)
        processed_outputs = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]


        if processed_outputs is None:
            return {
                "scores": torch.tensor([]),
                "labels": torch.tensor([], dtype=torch.long),
                "boxes": torch.tensor([])
            }

        return {
            "scores": processed_outputs["scores"].cpu() if processed_outputs["scores"] is not None else torch.tensor([]),
            "labels": processed_outputs["labels"].cpu() if processed_outputs.get("labels") is not None else torch.tensor([], dtype=torch.long),
            "boxes": processed_outputs["boxes"].cpu() if processed_outputs["boxes"] is not None else torch.tensor([])
        }
