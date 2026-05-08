# Automatic License Plate Recognition (ALPR) - Python Web App

A Python Web Application for Automatic License Plate Recognition built using Streamlit, OpenCV, and purely classical Computer Vision techniques. This project replicates a robust MATLAB ALPR pipeline without relying on deep learning models.

## Course Information

- **University:** Zagazig University
- **Course:** ECE 228
- **Instructor:** Dr. Azhar

## Team Members

- Mohamed Ahmed
- Mohie Eldein
- Mahmoud Ahmed
- Mohamed Adel
- Moaz Mohamed
- Mohamed Elsayed
- Mohamed Mesbah
- Ibrahim Khaled

## Features

- **EXIF Orientation Handling:** Automatically corrects image rotation issues caused by smartphone cameras using the `Pillow` library.
- **Classical Image Processing Pipeline:**
  - **Pre-processing:** Grayscale conversion and CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement.
  - **Edge Detection:** Bilateral filtering combined with Canny edge detection.
  - **Morphological Filtering:** Custom rectangular kernels to isolate and close plate-like structures.
  - **Localization:** Aspect-ratio and area-based contour filtering to accurately extract the license plate bounding box.
- **Character Recognition:** Template matching using normalized cross-correlation (equivalent to MATLAB's `corr2`) against a standard `Templates` directory.
- **Streamlit Web UI:** A clean, academic, and user-friendly interface for uploading images and visualizing the pipeline steps.

## Installation & Usage Locally

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install dependencies:**
   It is recommended to use a Python virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

3. **Add Templates:**
   Ensure that a folder named `Templates` is present in the root directory and contains binary character images (e.g., `A.bmp`, `1.bmp`) used for template matching.

4. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## Deployment

This project is structured specifically to be easily deployed on **Streamlit Cloud**. Simply link your GitHub repository to Streamlit Cloud, and it will automatically detect `app.py` and install the dependencies from `requirements.txt`.

## Architecture

- `app.py`: Contains the main Streamlit application and the Computer Vision processing logic using OpenCV.
- `requirements.txt`: Defines the required Python dependencies.
- `Templates/`: (Required) Directory containing character templates.
