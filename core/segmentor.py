# core/segmentor.py
import torch
import numpy as np
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamPredictor
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class Segmentor:
    def __init__(self, sam_checkpoint_path="sam_vit_h_4b8939.pth", sam_model_type="vit_h", owlvit_model_name="google/owlvit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load SAM model
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

        # Load OWL-ViT for text-to-box conversion
        self.owlvit_processor = OwlViTProcessor.from_pretrained(owlvit_model_name)
        self.owlvit_model = OwlViTForObjectDetection.from_pretrained(owlvit_model_name).to(self.device)

    def _visualize_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Applies a segmentation mask to an image."""
        
        annotated_image = image.copy().convert("RGBA")
        
        color = np.random.randint(0, 255, 3)
        mask_img = Image.new('RGBA', image.size, (color[0], color[1], color[2], 0))
        mask_draw = ImageDraw.Draw(mask_img)

        bitmap = Image.fromarray((mask * 255).astype(np.uint8))
        mask_draw.bitmap((0, 0), bitmap, fill=(color[0], color[1], color[2], 128))
        
        annotated_image.alpha_composite(mask_img)
        return annotated_image

    def segment_with_points(self, image: Image.Image, points: list, labels: list) -> Image.Image:
        """Segments an object using point prompts."""
        self.sam_predictor.set_image(np.array(image))
        masks, _, _ = self.sam_predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=False,
        )
        return self._visualize_mask(image, masks[0])

    def segment_with_box(self, image: Image.Image, box: list) -> Image.Image:
        """Segments an object using a bounding box prompt."""
        self.sam_predictor.set_image(np.array(image))
        masks, _, _ = self.sam_predictor.predict(
            box=np.array(box),
            multimask_output=False,
        )
        return self._visualize_mask(image, masks[0])

    def segment_with_text(self, image: Image.Image, text_prompt: str) -> Image.Image:
        """Segments an object using a text prompt by first detecting it with OWL-ViT."""
        inputs = self.owlvit_processor(text=[text_prompt], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.owlvit_model(**inputs)
        
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.owlvit_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)[0]
        
        if len(results["boxes"]) == 0:
            return image # Return original image if no object is detected

        # Use the box with the highest score as the prompt for SAM
        best_box = results["boxes"][0].tolist()
        return self.segment_with_box(image, best_box)
