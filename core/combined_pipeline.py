# core/combined_pipeline.py

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image, ImageDraw, ImageFont
import requests
import numpy as np
import io

class OwlViT_SAM_Pipeline:
    def __init__(self, owlvit_model_name="google/owlvit-base-patch32", sam_checkpoint_path="sam_vit_h_4b8939.pth", sam_model_type="vit_h"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device} for combined pipeline.")

        print("Loading OWL-ViT model...")
        self.owlvit_processor = OwlViTProcessor.from_pretrained(owlvit_model_name)
        self.owlvit_model = OwlViTForObjectDetection.from_pretrained(owlvit_model_name).to(self.device)
        
        print("Loading SAM model...")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        print("Combined pipeline models loaded.")

    def parse_prompt(self, prompt: str):
        detect_queries = []
        segment_queries = []
        # Simple parsing logic - can be replaced with a more advanced method
        if "box around the cat" in prompt or "detect the cat" in prompt:
            detect_queries.append("a cat")
        if "segment the dog" in prompt:
            segment_queries.append("a dog")
        # Add more rules as needed
        return detect_queries, segment_queries

    def run(self, image: Image.Image, prompt: str, threshold: float = 0.1):
        detect_queries, segment_queries = self.parse_prompt(prompt)
        all_queries = list(set(detect_queries + segment_queries))
        
        if not all_queries:
            return image, {}, {}

        inputs = self.owlvit_processor(text=all_queries, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.owlvit_model(**inputs)
        
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.owlvit_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

        detected_boxes = {}
        segmentation_masks = {}
        result_set = results[0]

        for box, score, label_idx in zip(result_set["boxes"], result_set["scores"], result_set["labels"]):
            query = all_queries[label_idx]
            box_coords = [round(i, 2) for i in box.tolist()]
            
            if query in detect_queries and query not in detected_boxes:
                detected_boxes[query] = box_coords
            
            if query in segment_queries and query not in segmentation_masks:
                self.sam_predictor.set_image(np.array(image))
                masks, _, _ = self.sam_predictor.predict(box=np.array(box_coords)[None, :], multimask_output=False)
                segmentation_masks[query] = masks[0]

        return self.visualize_results(image, detected_boxes, segmentation_masks)

    def visualize_results(self, image, detected_boxes, segmentation_masks):
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image, "RGBA")
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        for query, mask in segmentation_masks.items():
            color = np.random.randint(0, 255, 3)
            mask_img = Image.new('RGBA', image.size, (color[0], color[1], color[2], 0))
            mask_draw = ImageDraw.Draw(mask_img)
            mask_draw.bitmap((0,0), Image.fromarray(mask * 255), fill=(color[0], color[1], color[2], 128))
            annotated_image.paste(mask_img, (0,0), mask_img)

        for query, box in detected_boxes.items():
            draw.rectangle(box, outline="green", width=3)
            draw.text((box[0], box[1] - 20), query, fill="green", font=font)
        
        return annotated_image
