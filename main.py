# one_shot_object_detection/main.py
import os
from core.pipeline import DetectionPipeline
from ui.selector import BoundingBoxSelector
from ui.visualizer import ResultsVisualizer
from PIL import Image

def run_same_image_scenario():
    print("\n--- Running Same-Image Detection Scenario ---")
    image_path = "data/target/jar.jpg" # Make sure this image exists
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    pipeline = DetectionPipeline()
    visualizer = ResultsVisualizer()
    
    # Let user select a bounding box
    image_for_selection = Image.open(image_path).convert("RGB")
    selector = BoundingBoxSelector(image_for_selection)
    selected_bbox = selector.select_bbox()

    if not selected_bbox:
        print("No bounding box selected. Aborting.")
        return

    # Process detection
    original_image, results = pipeline.process_same_image_detection(
        image_path=image_path,
        bbox=selected_bbox,
        threshold=0.5
    )

    # Visualize results
    result_image = visualizer.draw_detections(original_image.copy(), results)
    visualizer.display_image(result_image, "Detections in the Same Image")

def run_cross_image_scenario():
    print("\n--- Running Cross-Image Detection Scenario ---")
    reference_image_path = "data/reference/coke.jpeg" # Make sure this exists
    target_image_paths = ["data/target/test.jpg"] # Make sure these exist

    if not os.path.exists(reference_image_path):
        print(f"Error: Reference image not found at {reference_image_path}")
        return

    pipeline = DetectionPipeline()
    visualizer = ResultsVisualizer()
    
    # Let user select bbox on reference image
    ref_image_for_selection = Image.open(reference_image_path).convert("RGB")
    selector = BoundingBoxSelector(ref_image_for_selection)
    reference_bbox = selector.select_bbox()

    if not reference_bbox:
        print("No bounding box selected. Aborting.")
        return

    # Process detection
    all_results = pipeline.process_cross_image_detection(
        reference_image_path=reference_image_path,
        reference_bbox=reference_bbox,
        target_image_paths=target_image_paths,
        threshold=0.5
    )

    # Visualize results
    images_to_display = []
    for path, (image, detections) in all_results.items():
        result_image = visualizer.draw_detections(image.copy(), detections)
        images_to_display.append((result_image, f"Detections in {os.path.basename(path)}"))
    
    visualizer.display_results_grid(images_to_display)

def run_text_prompt_scenario():
    print("\n--- Running Text-Prompt Detection Scenario ---")
    image_path = "data/target/jar.jpg"  # or prompt user for image path
    query_text = input("Enter your text prompt (e.g., 'a jar', 'cat', 'person'): ")

    pipeline = DetectionPipeline()
    visualizer = ResultsVisualizer()

    original_image, results = pipeline.process_text_prompt(image_path, query_text)
    result_image = visualizer.draw_detections(original_image.copy(), results)
    visualizer.display_image(result_image, f"Detections for prompt: '{query_text}'")


if __name__ == "__main__":
    print("Welcome to the One-Shot Object Detector!")
    print("1. Run Same-Image Detection (select object in one image to find others in it)")
    print("2. Run Cross-Image Detection (select object in one image to find it in others)")
    print("3. Run Text-Prompt Detection (enter a text description to find objects)")
    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == '1':
        run_same_image_scenario()
    elif choice == '2':
        run_cross_image_scenario()
    elif choice == '3':
        run_text_prompt_scenario()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
