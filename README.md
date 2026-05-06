# Medical Image Processing Web Application

A web application using the Streamlit library (https://streamlit.io/), created for a VSCode virtual environment (venv) and deployed with Streamlit Community Cloud.

You can view a **Live demo** here: [Medical Image Processing WebApp](https://cm-medical-image-processing.streamlit.app/)


---

## Overview
`MedImageProc_WebApp.py` contains several library imports, CSS coding for the page configuration, functions that are purely used for image loading and processing, and functions that use the Streamlit library to set up the page and user interface.

The image types the app can work with are: `.bmp`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, and most importantly, DICOM images (`.dcm`). The app can also display DICOM Metadata (information about the patient, the acquisition protocol, the imaging modality, the image size, etc.).
Provided with the script is a folder called `images`, which contains some samples of medical images to process with the app. The samples have been reproduced from *[Siemens Healthineers-Protocols](https://www.magnetomworld.siemens-healthineers.com/clinical-corner/protocols/dicom-images/deep-resolve)*, and are anonymous. The app also lets the user upload a file from a different directory, by copy-pasting the path, or through drag and drop.

---

## Project Structure

```
├── MedImageProc_WebApp.py   # Main application script
├── requirements.txt         # Python dependencies
├── images/                  # Sample medical images
└── README.md
```

---
## Chapters

The app contains four different chapters:

## 1 · Display Modification:
Three different categories:
- **Linear Windowing** — Optimal Display, Simple Window, Broken Window, Double Window
- **Non-Linear Windowing** — Inverse, Logarithmic, Inverse Logarithmic, Power, Sine-Window, Exp-Window
- **Histogram Manipulation** — Histogram Equalization, CDF Equalization, CLAHE

## 2 · Filters in the Space Domain:
Application of different filters (kernels) on the images:
- Smoothing, Laplacian, High-Emphasis, Median

## 3 · Filters in the Frequency Domain:
- Low-pass, High-pass, Band-pass, and Band-reject filters in different "shapes" (Ideal, Butterworth, Exponential, Gaussian)
- This chapter also lets the user restore the image from a degraded/"noisy" version of it using different restoration filters (Wiener, Unsupervised Wiener, Richardson-Lucy)

## 4 · Tomographic Reconstruction:
Uses different projections to create the image's sinogram, and then reconstruct it, either with the **FBP** (Filtered Back-Projection) or the **ART** (Algebraic Reconstruction Technique) method.

---

Throughout the chapters there is:
- The option to manipulate the window parameters in order to change the display in addition to processing it with other methods (such as filters)
- The option to download the processed images and their complementary graphs
- Processed images are downloaded in their original format, except DICOM ones, which are downloaded in `.png` format

---

## How to View the App in Your Browser

The script is meant to be run in VSCode with **Python 3.12.9**, after the installation of the right packages and the creation of a virtual environment.

### Step 1 — Download the repository files:
Download the files from the repository and make sure they are all in the same folder.

### Step 2 — Create and activate a virtual environment:
After making sure to create a venv and choosing the right interpreter, the command to activate the venv from the terminal is:

```
.\venv\Scripts\activate
```

### Step 3 — Install dependencies:
In order to install the necessary packages from the `requirements.txt` file, run this in your terminal:

```
pip install -r requirements.txt
```

### Step 4 — Launch the app:
Once everything is downloaded, launch the app through Streamlit:

```
streamlit run MedImageProc_WebApp.py
```

You can either follow the **Local URL** (e.g. `http://localhost:8501`) or the **Network URL** (e.g. `http://192.168.1.202:8501`).
