from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import numpy as np
import depth_pro
import os
import torch
import cv2

# Configuration

input_folder = "Z:/kccacheproxy/src/proxy/input"
# input_folder = "input"
output_folder = "Z:/midas"
# output_folder = "output"

FACE_SEG_WEIGHTS = "checkpoints/FaceSeg.pt"  # Replace with actual path to the model weights
BREAST_SEG_WEIGHTS = "checkpoints/BreastSeg.pt"  # Replace with actual path to the model weights
BREAST_SIZE_WEIGHTS = "checkpoints/BreastSize.pt"  # Replace with actual path to the model weights
CONF_THRESHOLD = 0.5  # Confidence threshold for detection
IMG_SIZE = 640  # Default input size for YOLOv11 (adjust if needed)

def get_average_intensity(image, mask):
    """
    Compute the average intensity of a single-channel image within a binary mask.
    
    Args:
        image: Single-channel image (grayscale, uint8 or float)
        mask: Binary mask (uint8, 0 for excluded pixels, 255 for included pixels)
    
    Returns:
        float: Average intensity in the masked area, or 0 if the mask is empty/invalid
    """
    try:        
        # Convert mask to boolean (True for non-zero pixels)
        mask_bool = mask > 0
        # Check if any pixels are included in the mask
        if not np.any(mask_bool):
            return 0.0
        masked_pixels = image[mask_bool]
        average_intensity = np.mean(masked_pixels)
        return float(average_intensity)
    except Exception as e:
        print(f"Error computing average intensity: {e}")
        return 0.0

def load_yolo_model(weights_path):
    """Load the YOLOv11 model."""
    try:
        model = YOLO(weights_path)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model from {weights_path}: {e}")


def brightest_neighbor_filter(image, kernel_size=7):
    """
    Applies a max filter (brightest neighbor filter) using OpenCV's dilate.
    
    Args:
        image: Input image (grayscale or color).
        kernel_size: Size of the neighborhood (should be odd).
    
    Returns:
        Filtered image with maximum values in each neighborhood.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    if image.ndim == 2:  # Grayscale
        return cv2.dilate(image, kernel)
    elif image.ndim == 3:  # Color image
        # Apply dilation to each channel independently
        channels = cv2.split(image)
        dilated_channels = [cv2.dilate(c, kernel) for c in channels]
        return cv2.merge(dilated_channels)
    else:
        raise ValueError("Unsupported image format")

def fast_box_blur_integral(image, kernel_size=41):
    """Apply a fast box blur to a single-channel image using an integral image."""
    try:
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be odd and positive")
        if image is None:
            raise ValueError("Input image is None")
        if len(image.shape) != 2:
            raise ValueError("Image must be single-channel (2D array)")
        
        # Ensure uint8
        image = image.astype(np.uint8) if image.dtype != np.uint8 else image
        height, width = image.shape
        
        # Compute integral image (default type, typically float64)
        integral = cv2.integral(image)
        blurred = np.zeros_like(image, dtype=np.float32)
        half_k = kernel_size // 2
        
        # Compute blur
        for i in range(height):
            for j in range(width):
                y1 = max(0, i - half_k)
                y2 = min(height, i + half_k + 1)
                x1 = max(0, j - half_k)
                x2 = min(width, j + half_k + 1)
                area_sum = (
                    integral[min(y2 + 1, height), min(x2 + 1, width)] -
                    integral[min(y2 + 1, height), x1] -
                    integral[y1, min(x2 + 1, width)] +
                    integral[y1, x1]
                )
                area = (y2 - y1) * (x2 - x1)
                blurred[i, j] = area_sum / area if area > 0 else image[i, j]
        
        return np.clip(blurred, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error in integral box blur: {e}")
        return image  # Return original to avoid None

def preprocess_image(image_path, img_size):
    """Preprocess the input image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    return img

def perform_segmentation(model, img, conf_threshold, img_size):
    """Perform face segmentation and extract the mask."""
    # Run inference
    results = model.predict(source=img, conf=conf_threshold, imgsz=img_size, verbose=False)
    
    # Initialize an empty mask
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Process segmentation results
    if results[0].masks is not None:
        for mask_data in results[0].masks.data:
            # Convert mask to numpy array and resize to original image size
            mask_single = mask_data.cpu().numpy()
            mask_single = cv2.resize(mask_single, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_single = (mask_single > 0).astype(np.uint8) * 255
            mask = np.maximum(mask, mask_single)  # Combine masks if multiple faces
    
    return mask

def detect_breast_size(model, img, conf_threshold, img_size):
    """Get the breast size class of the image.
    :return: 0-4 from small to big, or -1 if no breast is found."""
    results = model.predict(source=img, conf=conf_threshold, imgsz=img_size, verbose=True)
    
    detections = []
    height, width = img.shape[:2]
    
    for result in results:
        if result.boxes is not None:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                confidence = float(conf)
                class_id = int(cls)
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class': class_id
                })
    print(detections)
    if not detections:
        return -1
    max_detection = max(detections, key=lambda x: x['confidence'])
    return max_detection['class']


# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load model and preprocessing transform
try:
    depth_model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), precision=torch.float16)
    depth_model.eval()
    print(f"Model is running on: {next(depth_model.parameters()).device}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
    
face_seg_model = load_yolo_model(FACE_SEG_WEIGHTS)
breast_seg_model = load_yolo_model(BREAST_SEG_WEIGHTS)
breast_size_model = load_yolo_model(BREAST_SIZE_WEIGHTS)

# Get list of image files in input folder
image_extensions = ('.png')
input_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

for image_file in input_files:
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, f"{image_file}")

    # Check if output files already exist
    if os.path.exists(output_path) and os.path.exists(output_path):
        print(f"Skipping {image_file}: Output files already exist")
        continue

    # Preprocess image
    img = preprocess_image(input_path, IMG_SIZE)
    
    # Perform segmentation
    face_mask = perform_segmentation(face_seg_model, img, CONF_THRESHOLD, IMG_SIZE)
    breast_size = detect_breast_size(breast_size_model, img, CONF_THRESHOLD, IMG_SIZE)
    if breast_size != -1:
        breast_mask = perform_segmentation(breast_seg_model, img, CONF_THRESHOLD, IMG_SIZE)
        breast_mask = fast_box_blur_integral(breast_mask, 41)
    

    print(f"Processing {image_file}...")

    # Load and preprocess image
    try:
        image, _, f_px = depth_pro.load_rgb(input_path)
        image = transform(image)
        print(f"Input tensor for {image_file} is on: {image.device}")
    except Exception as e:
        print(f"Error loading or preprocessing {image_file}: {e}")
        continue

    print(f"Run inference...")
    # Run inference
    try:
        prediction = depth_model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m]
        print(f"Depth tensor for {image_file} is on: {depth.device}")
    except Exception as e:
        print(f"Error during inference for {image_file}: {e}")
        continue
    print(f"Run inference done...")

    # Convert PyTorch tensor to NumPy array
    depth = depth.cpu().numpy()

    # Normalize depth map to 0-255 for visualization
    inverse_depth = 1 / depth

    inverse_depth_min = np.min(inverse_depth)
    inverse_depth_max = np.max(inverse_depth)
    if inverse_depth_max > inverse_depth_min:  # Avoid division by zero
        depth_normalized = (inverse_depth - inverse_depth_min) / (inverse_depth_max - inverse_depth_min) * 255
    else:
        depth_normalized = np.zeros_like(depth)  # Handle case where depth is constant
    depth_normalized = depth_normalized.astype(np.uint8)

    # Post-processing with OpenCV
    print(f"Post-processing with OpenCV...")
    try:
        # Load original image with alpha channel
        original_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if original_image.shape[2] == 4:  # Check if image has alpha channel
            alpha_channel = original_image[:, :, 3]
        else:
            # If no alpha channel, assume fully opaque
            alpha_channel = np.full_like(depth_normalized, 255, dtype=np.uint8)

        print(f"GaussianBlur...")
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(depth_normalized, (7, 7), cv2.BORDER_DEFAULT)
        print(f"brightest_neighbor_filter...")
        blur = brightest_neighbor_filter(depth_normalized)

        # Create 4-channel RGBA output
        height, width = depth_normalized.shape
        
        print(f"get_average_intensity...")
        face_average_depth = get_average_intensity(depth_normalized, face_mask)
        if face_average_depth == 0:
            face_average_depth = 128


        print(f"Assign pixel values based on alpha channel...")

        # Select depth_value: use `blur` where alpha < 255, otherwise use `depth_normalized`
        depth_value = np.where(alpha_channel < 255, blur, depth_normalized)

        # Compute the red channel
        i_coords = np.arange(height).reshape(-1, 1)  # column vector for row index
        r_channel = 140 + ((depth_value - face_average_depth) / 5.0) + (20.0 * i_coords / height)
        r_channel = np.clip(r_channel, 0, 255).astype(np.uint8)

        # Compute the blue channel if breast size is detected
        if breast_size != -1:
            b_channel = (breast_mask / 255.0) * (10 + breast_size * 25)
            b_channel[b_channel < 7] = 0
            b_channel = b_channel.astype(np.uint8)
        else:
            b_channel = np.zeros((height, width), dtype=np.uint8)

        # Green channel is always zero
        g_channel = np.zeros((height, width), dtype=np.uint8)

        # Merge channels into final image
        output_image = np.stack([b_channel, g_channel, r_channel], axis=-1)


        print(f"Save the processed image...")
        # Save the processed image
        cv2.imwrite(output_path, output_image)
        print(f"Processed depth map saved as {output_path}")
    except Exception as e:
        print(f"Error during post-processing or saving for {image_file}: {e}")
        continue

print("Processing complete!")