# one_shot_object_detection/batch_process.py
import os
import argparse
from pathlib import Path
from PIL import Image
from core.pipeline import DetectionPipeline
from ui.selector import BoundingBoxSelector
from ui.visualizer import ResultsVisualizer

def main(args):
    if not os.path.exists(args.reference_image):
        print(f"Error: Reference image not found at {args.reference_image}")
        return

    pipeline = DetectionPipeline()
    visualizer = ResultsVisualizer()

    # Select BBox on reference image
    ref_image_for_selection = Image.open(args.reference_image).convert("RGB")
    selector = BoundingBoxSelector(ref_image_for_selection)
    reference_bbox = selector.select_bbox()

    if not reference_bbox:
        print("No bounding box selected. Aborting batch process.")
        return
    
    # Prepare directories
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    image_files = list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {target_dir}")
        return
        
    print(f"\nStarting batch processing on {len(image_files)} images...")

    # Process detections
    results = pipeline.process_cross_image_detection(
        reference_image_path=args.reference_image,
        reference_bbox=reference_bbox,
        target_image_paths=[str(p) for p in image_files],
        threshold=args.threshold
    )

    # Save annotated images
    for path_str, (image, detections) in results.items():
        annotated_image = visualizer.draw_detections(image.copy(), detections)
        output_filename = output_dir / f"annotated_{Path(path_str).name}"
        annotated_image.save(output_filename)
        print(f"Saved annotated image to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images for one-shot object detection.")
    parser.add_argument("--reference_image", required=True, help="Path to the reference image containing the query object.")
    parser.add_argument("--target_dir", default="data/target", help="Directory containing target images to process.")
    parser.add_argument("--output_dir", default="output/annotated_images", help="Directory to save annotated images.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection confidence threshold.")
    
    args = parser.parse_args()
    main(args)
