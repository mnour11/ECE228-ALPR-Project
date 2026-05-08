import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="ALPR System | ECE 228", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("Team Members")
st.sidebar.markdown("""
- Mohamed Ahmed
- Mohie Eldein
- Mahmoud Ahmed
- Mohamed Adel
- Moaz Mohamed
- Mohamed Elsayed
- Mohamed Mesbah
- Ibrahim Khaled
""")

st.sidebar.title("Course Information")
st.sidebar.markdown("""
**Zagazig University**  
Course: ECE 228  
Submitted to: Dr. Azhar
""")

# --- MAIN UI ---
st.title("Automatic License Plate Recognition (ALPR)")
st.write("A purely classical Computer Vision approach using OpenCV (No Deep Learning).")

uploaded_file = st.file_uploader("Upload Car Image", type=['jpg', 'jpeg', 'png', 'bmp'])

def fix_image_orientation(image):
    """Fix image orientation using PIL EXIF data."""
    return ImageOps.exif_transpose(image)

def process_pipeline(image_np):
    """Apply classical CV pipeline to locate and crop the license plate."""
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 1. CLAHE enhancement (mimicking MATLAB adapthisteq)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 2. Canny Edge Detection with Bilateral Filtering to reduce noise
    blur = cv2.bilateralFilter(enhanced, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    
    # 3. Morphological filtering (imclose and imfill equivalent)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_contour = None
    max_area = 0
    
    # 4. Aspect-Ratio based plate localization
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        
        # Typical aspect ratio for plates is between 2 and 6
        if 2.0 < aspect_ratio < 6.0 and area > 800:
            if area > max_area:
                max_area = area
                plate_contour = (x, y, w, h)
                
    if plate_contour is None:
        return None, None
        
    x, y, w, h = plate_contour
    # Crop from the grayscale image for better binarization later
    plate_crop = gray[y:y+h, x:x+w]
    return plate_crop, plate_contour

def match_templates(plate_crop, templates_dir='Templates'):
    """Recognize characters using template matching (mimicking corr2)."""
    if not os.path.exists(templates_dir):
        return "ERROR: 'Templates' folder not found."
        
    # Binarize plate crop
    _, thresh = cv2.threshold(plate_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Ensure characters are white on black background
    if np.sum(thresh) > (thresh.shape[0] * thresh.shape[1] * 255) / 2:
        thresh = cv2.bitwise_not(thresh)
        
    # Find contours for characters
    char_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes
    bounding_boxes = [cv2.boundingRect(c) for c in char_contours]
    if not bounding_boxes:
        return "No characters found"
        
    # Sort contours from left to right
    contours_with_boxes = zip(char_contours, bounding_boxes)
    contours_with_boxes = sorted(contours_with_boxes, key=lambda b: b[1][0])
    
    recognized_text = ""
    template_files = [f for f in os.listdir(templates_dir) if f.endswith(('.bmp', '.png', '.jpg'))]
    
    if not template_files:
        return "ERROR: No templates found in 'Templates' folder."
        
    # Pre-load and process templates
    templates = {}
    for t_file in template_files:
        char_name = os.path.splitext(t_file)[0]
        # Handle cases where template names might have trailing characters (e.g., A_1.bmp)
        if len(char_name) > 1:
            char_name = char_name[0]
            
        t_path = os.path.join(templates_dir, t_file)
        t_img = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)
        if t_img is not None:
            _, t_thresh = cv2.threshold(t_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Standardize size for correlation matching
            t_thresh = cv2.resize(t_thresh, (24, 42)) 
            templates[char_name] = t_thresh

    # Process each detected character
    for cnt, bbox in contours_with_boxes:
        x, y, w, h = bbox
        aspect_ratio = w / float(h)
        area = w * h
        
        # Filter out noise (non-characters)
        if aspect_ratio > 1.5 or h < 10 or area < 20:
            continue
            
        # Extract character ROI
        char_roi = thresh[y:y+h, x:x+w]
        char_roi = cv2.resize(char_roi, (24, 42))
        
        best_score = -1
        best_char = "?"
        
        # Template matching (Normalized Cross-Correlation ~ corr2)
        for char_name, t_img in templates.items():
            res = cv2.matchTemplate(char_roi, t_img, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]
            
            if score > best_score:
                best_score = score
                best_char = char_name
                
        # Only accept if the correlation score is reasonable
        if best_score > 0.4:
            recognized_text += best_char
            
    return recognized_text if recognized_text else "UNRECOGNIZED"

if uploaded_file is not None:
    # Read image using PIL to handle EXIF properly
    pil_image = Image.open(uploaded_file)
    pil_image = fix_image_orientation(pil_image)
    
    # Convert to OpenCV format (NumPy Array - RGB)
    image_np = np.array(pil_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_container_width=True)
        
    with st.spinner('Processing...'):
        plate_crop, plate_contour = process_pipeline(image_np)
    
    if plate_crop is not None:
        with col2:
            st.subheader("Detected Plate Crop")
            st.image(plate_crop, cmap='gray', use_container_width=True)
            
        st.subheader("Final Recognized Text")
        recognized_text = match_templates(plate_crop)
        
        if "ERROR" in recognized_text:
            st.error(recognized_text)
        else:
            st.success(f"### {recognized_text}")
    else:
        st.error("No License Plate Detected.")
