# Imports
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import numpy as np
import argparse


def extract_polygon_region(image_np, polygon_points):
    """
    Extract a rectangular region containing a polygon from an image and mask everything outside
    the polygon with black background.
    
    Args:
        image_np: numpy array of image (height, width, channels)
        polygon_points: numpy array of points defining the polygon, shape (N, 2)
    
    Returns:
        numpy array containing the extracted region with black background
    """
    # Get image dimensions
    height, width = image_np.shape[:2]
    
    # Create a mask for the polygon
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert polygon points to integer array for cv2
    polygon_points_int = polygon_points.astype(np.int32)
    
    # Draw the polygon on the mask
    cv2.fillPoly(mask, [polygon_points_int], 255)
    
    # Get bounding box of the polygon
    x_min, y_min = np.min(polygon_points_int, axis=0)
    x_max, y_max = np.max(polygon_points_int, axis=0)
    
    # Extract the region of interest from the mask
    mask_region = mask[y_min:y_max, x_min:x_max]
    
    # Create black background image of same size as bounding box
    result = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    
    # Extract the region of interest from the image
    image_region = image_np[y_min:y_max, x_min:x_max]
    
    # Apply the mask to each channel
    for c in range(3):
        result[:, :, c] = image_region[:, :, c] * (mask_region / 255)
    
    return result

def main(input_dir,model_path):
    # Directories
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path('./generate_output')
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = output_dir / 'overlays'
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_prefix = "" # Image prefix, can be empty
    overlay_suffix = "" # Image suffix, can be empty

    detection_dir = output_dir / 'detections'
    detection_dir.mkdir(parents=True, exist_ok=True)
    detection_prefix = ""  # Text prefix, can be empty
    detection_suffix = ""  # Text suffix, can be empty

    mask_dir = output_dir / 'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_prefix = ""  # Text prefix, can be empty
    mask_suffix = ""  # Text suffix, can be empty

    # Add this after initializing other directories
    crop_dir = output_dir / 'crops'
    crop_dir.mkdir(parents=True, exist_ok=True)


    # Load your trained model

    model = YOLO(model_path)

    print(model.task)
    #detect,segment,obb

    # Mode selection: detection or segmentation
    mode = "segmentation"

    # Detect all classes or selected classes only
    detect_all_classes = True  # Set to True to detect all classes, False to detect only specific classes below

    # Classes to detect
    # Example: ['SpeechBalloons', 'General_speech', 'hit_sound', 'blast_sound', 'narration speech', 'thought_speech', 'roar']
    selected_classes = ['comic_frame']

    # Class override mapping, treats the left side of the mapping as if it was the class of the right side
    # Example: thought_speech annotations will be treated as SpeechBalloons annotations.
    class_overrides = {
        'thought_speech': 'SpeechBalloons',
    }

    # Confidence threshold
    confidence_threshold = 0.3

    # Label settings
    label_boxes = True  # Draw class names or just boxes
    font_size = 30  # Font size for the class labels

    try:
        font = ImageFont.truetype("arial.ttf", 30)  # Update font size as needed
    except IOError:
        font = ImageFont.load_default()
        print("Default font will be used, as custom font not found.")

    # Label colors by index
    predefined_colors_with_text = [
        ((204, 0, 0),     'white'),  # Darker red, white text
        ((0, 204, 0),     'black'),  # Darker green, black text
        ((0, 0, 204),     'white'),  # Darker blue, white text
        ((204, 204, 0),   'black'),  # Darker yellow, black text
        ((204, 0, 204),   'white'),  # Darker magenta, white text
        ((0, 204, 204),   'black'),  # Darker cyan, black text
        ((153, 0, 0),     'white'),  # Darker maroon, white text
        ((0, 153, 0),     'white'),  # Darker green, white text
        ((0, 0, 153),     'white'),  # Darker navy, white text
        ((153, 153, 0),   'black'),  # Darker olive, black text
        # Add more color pairs if needed
    ]

    # Assign colors to each class, wrapping around if there are more classes than colors
    class_colors = {class_name: predefined_colors_with_text[i % len(predefined_colors_with_text)][0] for i, class_name in enumerate(selected_classes)}
    text_colors = {class_name: predefined_colors_with_text[i % len(predefined_colors_with_text)][1] for i, class_name in enumerate(selected_classes)}


    # Store input images in a variable
    image_paths = []
    for extension in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(input_dir.glob(extension))

    # Segmentation class
    class YOLOSEG:
        def __init__(self, model_path):
            self.model = YOLO(model_path)

        def detect(self, img):
            height, width, _ = img.shape
            results = self.model.predict(source=img.copy(), save=False, save_txt=False)
            result = results[0]

            segmentation_contours_idx = []
            if len(result) > 0:
                for seg in result.masks.xy:
                    segment = np.array(seg, dtype=np.float32)
                    segmentation_contours_idx.append(segment)

            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
            scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
            return bboxes, class_ids, segmentation_contours_idx, scores

    ys = YOLOSEG(model_path)

    # Function to estimate text size
    def estimate_text_size(label, font_size):
        approx_char_width = font_size * 0.6
        text_width = len(label) * approx_char_width
        text_height = font_size
        return text_width, text_height

    def write_detections_to_file(image_path, detections):
        # Create a text file named after the image
        text_file_path = detection_dir / f"{detection_prefix}{image_path.stem}{detection_suffix}.txt"

        with open(text_file_path, 'w') as file:
            for detection in detections:
                file.write(f"{detection}\n")

    # Process images with progress bar
    print(f"Generating outputs in {mode} mode.")
    for image_path in tqdm(image_paths, desc='Processing Images'):
        if mode == "detection":
            img_cv = cv2.imread(str(image_path))  # Load the image with OpenCV for cropping
            mask_img = np.zeros(img_cv.shape[:2], dtype=np.uint8)  # Initialize a blank mask for all detections

            img_pil = Image.open(image_path)  # Load the image with PIL for overlay generation
            #results = model.predict(img_pil)
            results = model(img_pil)
            draw = ImageDraw.Draw(img_pil)
            detections = []

            if len(results) > 0 and results[0].boxes.xyxy is not None:
                for idx, segm_mask in enumerate(results[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, segm_mask[:4].tolist())  # Ensure coordinates are integers
                    cls_id = int(results[0].boxes.cls[idx].item())
                    conf = results[0].boxes.conf[idx].item()
                    cls_name = results[0].names[cls_id] if 0 <= cls_id < len(results[0].names) else "Unknown"
                    cls_name = class_overrides.get(cls_name, cls_name)

                    if (cls_name in selected_classes or detect_all_classes) and conf >= confidence_threshold:
                        # Draw bounding boxes and labels
                        box_color = class_colors.get(cls_name, (255, 0, 0))
                        text_color = text_colors.get(cls_name, 'black')
                        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=7)

                        # Fill mask image for this detection
                        cv2.rectangle(mask_img, (x1, y1), (x2, y2), 255, thickness=-1)

                        if label_boxes:
                            label = f"{cls_name}: {conf:.2f}"
                            text_size = estimate_text_size(label, font_size)
                            draw.rectangle([x1, y1 - text_size[1] - 5, x1 + text_size[0], y1], fill=box_color)
                            draw.text((x1, y1 - text_size[1] - 5), label, fill=text_color, font=font)

                        # Add detection data to the list
                        detections.append(f"{cls_name} {conf:.2f} {x1} {y1} {x2} {y2}")

                        # Crop the detected region
                        crop = img_cv[y1:y2, x1:x2]  # Crop the detection from the image

                        # Calculate aspect ratio
                        h, w = crop.shape[:2]
                        target_ratio = 1.5
                        current_ratio = w / h

                        # Skip padding if within the acceptable range
                        if 1 / target_ratio <= current_ratio <= target_ratio:
                            padded_crop = crop  # No padding needed
                        elif current_ratio < 1 / target_ratio:  # Too tall, add padding to width
                            target_width = int(h / target_ratio)
                            padding = target_width - w
                            padding_left = padding // 2
                            padding_right = padding - padding_left
                            padded_crop = cv2.copyMakeBorder(crop, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        elif current_ratio > target_ratio:  # Too wide, add padding to height
                            target_height = int(w / target_ratio)
                            padding = target_height - h
                            padding_top = padding // 2
                            padding_bottom = padding - padding_top
                            padded_crop = cv2.copyMakeBorder(crop, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                        # Resize to 1 megapixel (approximately 1,000,000 pixels)
                        target_area = 1_000_000
                        aspect_ratio = padded_crop.shape[1] / padded_crop.shape[0]
                        target_width = int((target_area * aspect_ratio) ** 0.5)
                        target_height = int(target_area / target_width)
                        resized_image = cv2.resize(padded_crop, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

                        # Save the processed crop
                        crop_filename = crop_dir / f"{image_path.stem}_{cls_name}_{idx}.png"
                        cv2.imwrite(str(crop_filename), resized_image)

            # Save overlay images
            img_pil.save(overlay_dir / f"{overlay_prefix}{image_path.stem}{overlay_suffix}{image_path.suffix}")

            # Write detections to a text file
            write_detections_to_file(image_path, detections)

            # Save the combined mask image
            mask_output_path = mask_dir / f"{mask_prefix}{image_path.stem}{mask_suffix}.png"
            cv2.imwrite(str(mask_output_path), mask_img)
        
        if mode == "segmentation":
            img_cv = cv2.imread(str(image_path))  # Load the image with OpenCV for cropping
            mask_img = np.zeros(img_cv.shape[:2], dtype=np.uint8)  # Initialize a blank mask for all detections

            img_pil = Image.open(image_path)  # Load the image with PIL for overlay generation
            #results = model.predict(img_pil)
            results = model(img_pil)
            draw = ImageDraw.Draw(img_pil)
            detections = []

            if len(results) > 0 and results[0].masks.xy is not None:
                for idx, segm_mask32 in enumerate(results[0].masks.xy):
                    segm_mask = segm_mask32.astype(int)
                    cls_id = int(results[0].boxes.cls[idx].item())
                    conf = results[0].boxes.conf[idx].item()
                    cls_name = results[0].names[cls_id] if 0 <= cls_id < len(results[0].names) else "Unknown"
                    cls_name = class_overrides.get(cls_name, cls_name)

                    if (cls_name in selected_classes or detect_all_classes) and conf >= confidence_threshold:
                        # Draw bounding boxes and labels
                        box_color = class_colors.get(cls_name, (255, 0, 0))
                        text_color = text_colors.get(cls_name, 'black')
                        #draw.polygon(segm_mask, outline=box_color, width=7)

                        # Fill mask image for this detection
                        #cv2.rectangle(mask_img, (x1, y1), (x2, y2), 255, thickness=-1)
                        cv2.fillPoly(mask_img, pts=[segm_mask], color=(255, 0, 0))


                        if label_boxes:
                            label = f"{cls_name}: {conf:.2f}"
                            text_size = estimate_text_size(label, font_size)
                            draw.polygon(segm_mask, outline="red")
                            #draw.rectangle([x1, y1 - text_size[1] - 5, x1 + text_size[0], y1], fill=box_color)
                            #draw.text((x1, y1 - text_size[1] - 5), label, fill=text_color, font=font)

                        # Add detection data to the list
                        #detections.append(f"{cls_name} {conf:.2f} {x1} {y1} {x2} {y2}")

                        # Crop the detected region
                        #crop = img_cv[y1:y2, x1:x2]  # Crop the detection from the image

                        #crop = img_cv[segm_mask]
                        #crop = segm_mask
                        crop_filename = crop_dir / f"{image_path.stem}_{cls_name}_{idx}.png"
                        cv2.imwrite(str(crop_filename), extract_polygon_region(img_cv,segm_mask))
                        
                        """ 
                        crop = segm_mask

                        # Calculate aspect ratio
                        _, _, w, h = cv2.boundingRect(segm_mask)
                        target_ratio = 1.5
                        current_ratio = w / h

                        # Skip padding if within the acceptable range
                        if 1 / target_ratio <= current_ratio <= target_ratio:
                            padded_crop = crop  # No padding needed
                        elif current_ratio < 1 / target_ratio:  # Too tall, add padding to width
                            target_width = int(h / target_ratio)
                            padding = target_width - w
                            padding_left = padding // 2
                            padding_right = padding - padding_left
                            padded_crop = cv2.copyMakeBorder(crop, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        elif current_ratio > target_ratio:  # Too wide, add padding to height
                            target_height = int(w / target_ratio)
                            padding = target_height - h
                            padding_top = padding // 2
                            padding_bottom = padding - padding_top
                            padded_crop = cv2.copyMakeBorder(crop, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                        # Resize to 2 megapixel (approximately 1,000,000 pixels)
                        target_area = 2_000_000
                        aspect_ratio = padded_crop.shape[1] / padded_crop.shape[0]
                        target_width = int((target_area * aspect_ratio) ** 0.5)
                        target_height = int(target_area / target_width)
                        resized_image = cv2.resize(padded_crop, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

                        # Save the processed crop
                        crop_filename = crop_dir / f"{image_path.stem}_{cls_name}_{idx}.png"
                        cv2.imwrite(str(crop_filename), resized_image) """


            # Save overlay images
            img_pil.save(overlay_dir / f"{overlay_prefix}{image_path.stem}{overlay_suffix}{image_path.suffix}")

            # Write detections to a text file
            #write_detections_to_file(image_path, detections)

            # Save the combined mask image
            mask_output_path = mask_dir / f"{mask_prefix}{image_path.stem}{mask_suffix}.png"
            cv2.imwrite(str(mask_output_path), mask_img)



    print(f"Processed {len(image_paths)} images. Overlays saved to '{overlay_dir}', Detections saved to '{detection_dir}', and Masks saved to '{mask_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Generating script',
                    description='Script to generate masks')
    parser.add_argument("input_dir", nargs='?', help="Path to input images", type=str)
    parser.add_argument("model_name", nargs='?', help="model name in models folder", type=str)
    args = parser.parse_args()
    #print(args.input_dir.replace('\\','/'))
    #print(f'./models/{args.model_name}.pt')
    
    main(args.input_dir.replace('\\','/'),f'./models/{args.model_name}.pt')
