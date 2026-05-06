# -*- coding: utf-8 -*-
"""
Medical Image Processing Web-App
Streamlit application with four chapters:
  1. Display Modification Methods
  2. Filters in the Space Domain
  3. Filters in the Frequency Domain
  4. Tomographic Image Reconstruction
"""

import io
import os
import math
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tempfile
from matplotlib.figure import Figure
from PIL import Image as PILImage
from scipy import signal
import streamlit as st

warnings.filterwarnings("ignore")

try:
    from skimage.transform import radon, iradon, iradon_sart
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Image Processing Web-App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080e1a;
    color: #cdd9e5;
}
.stApp { background-color: #080e1a; }

h1, h2, h3, h4 {
    font-family: 'Space Mono', monospace;
    color: #e6edf3;
    letter-spacing: -0.3px;
}

section[data-testid="stSidebar"] {
    background-color: #0d1526;
    border-right: 1px solid #1c2d40;
}
section[data-testid="stSidebar"] * { color: #adbac7 !important; }
section[data-testid="stSidebar"] h2 { color: #6cb6ff !important; font-family: 'Space Mono', monospace !important; }

.sidebar-chapter {
    background: linear-gradient(135deg, #112240 0%, #0d1b33 100%);
    border: 1px solid #1f3a5f;
    border-left: 3px solid #388bfd;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 10px 0 6px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6cb6ff !important;
}

.chapter-header {
    background: linear-gradient(135deg, #0f2039 0%, #0a1828 100%);
    border: 1px solid #1c3a5c;
    border-left: 4px solid #388bfd;
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 20px;
}
.chapter-header h2 {
    margin: 0 0 4px 0;
    font-size: 1.3rem;
    color: #6cb6ff !important;
}
.chapter-header p {
    margin: 0;
    color: #768390;
    font-size: 0.88rem;
}

.metric-card {
    background: #0d1526;
    border: 1px solid #1c2d40;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label { font-size: 0.68rem; color: #768390; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.2rem; color: #6cb6ff; }

.stButton > button {
    background: linear-gradient(135deg, #3f85ee, #2f5fa3);
    color: white; border: none; border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem; letter-spacing: 0.04em;
    padding: 0.55rem 1.2rem; width: 100%;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

.stDownloadButton > button {
    background: #0d1526; color: #6cb6ff;
    border: 1px solid #1f3a5f; border-radius: 6px;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    padding: 0.5rem 1rem; width: 100%;
}
.stDownloadButton > button:hover { border-color: #388bfd; }

div[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace; color: #6cb6ff;
}

.stSelectbox label, .stSlider label, .stRadio label,
.stNumberInput label, .stFileUploader label, .stCheckbox label {
    color: #768390 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d1526;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #768390;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #1f3a5f;
    color: #6cb6ff !important;
}

.info-box {
    background: #091a2e;
    border: 1px solid #1c3a5c;
    border-radius: 8px;
    padding: 16px 20px;
    color: #97a4ad; 
    font-size: 0.88rem;
    margin-top: 16px;
}

hr { border-color: #1c2d40 !important; }

.stAlert { border-radius: 8px; }

div[data-testid="stExpander"] {
    background: #0d1526;
    border: 1px solid #1c2d40;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1526",
    "axes.facecolor": "#111f35",
    "axes.edgecolor": "#1c2d40",
    "axes.labelcolor": "#768390",
    "xtick.color": "#768390",
    "ytick.color": "#768390",
    "text.color": "#cdd9e5",
    "grid.color": "#1c2d40",
    "lines.color": "#387cdb",
    "font.family": "monospace",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titlecolor": "#6cb6ff",
})

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_DEPTH = 255
TONES = 256
DICOM_EXTS = (".dcm", ".dicom", ".ima")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def imNormalize(w, tones=256):
    mx, mn = np.max(w), np.min(w)
    if mx == mn:
        return np.zeros_like(w, dtype=float)
    return np.round((tones - 1) * (w - mn) / (mx - mn))

def to_uint8(arr):
    return np.asarray(np.clip(arr, 0, 255), dtype=np.uint8)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

def arr_to_bytes(arr, fmt="png"):
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    pil_img = PILImage.fromarray(arr_u8, mode="L")
    buf = io.BytesIO()
    pil_fmt = "JPEG" if fmt.lower() in ("jpg", "jpeg") else fmt.upper()
    try:
        pil_img.save(buf, format=pil_fmt)
    except Exception:
        pil_img.save(buf, format="PNG")
        fmt = "png"
    buf.seek(0)
    return buf.read(), fmt

def is_dicom_file(path):
    if path.lower().endswith(DICOM_EXTS):
        return True
    if "." not in os.path.basename(path):
        try:
            with open(path, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except Exception:
            return False
    return False

def load_dicom(path):
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept
    meta = {
        "Patient Name": str(getattr(ds, "PatientName", "N/A")),
        "Modality": str(getattr(ds, "Modality", "N/A")),
        "Study Date": str(getattr(ds, "StudyDate", "N/A")),
        "Rows x Cols": f"{getattr(ds, 'Rows', '?')} x {getattr(ds, 'Columns', '?')}",
        "Window Center": str(getattr(ds, "WindowCenter", "N/A")),
        "Window Width": str(getattr(ds, "WindowWidth", "N/A")),
        "Bits Stored": str(getattr(ds, "BitsStored", "N/A")),
    }
    return arr, meta, ds

def dicom_default_window(arr, ds=None):
    wc, ww = None, None
    if ds is not None:
        try:
            raw_wc = getattr(ds, "WindowCenter", None)
            raw_ww = getattr(ds, "WindowWidth", None)
            if raw_wc is not None and raw_ww is not None:
                wc = float(raw_wc[0]) if hasattr(raw_wc, "__iter__") else float(raw_wc)
                ww = float(raw_ww[0]) if hasattr(raw_ww, "__iter__") else float(raw_ww)
        except Exception:
            pass
    if wc is None:
        mn, mx = arr.min(), arr.max()
        wc = (mn + mx) / 2
        ww = mx - mn
    return wc, ww

def apply_dicom_window(arr, wc, ww):
    lo = wc - ww / 2
    hi = wc + ww / 2
    out = np.clip(arr, lo, hi)
    return (out - lo) / max(hi - lo, 1e-9) * 255.0

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADER (handles standard + DICOM via upload or folder)
# Returns: (gray_float_array, original_filename, extension, is_dicom, dicom_meta)
# ─────────────────────────────────────────────────────────────────────────────

def image_loader_widget(key_prefix="main"):
    """Renders sidebar image-load controls + unified 0-255 window sliders.
    Returns (im_0_255, wc, ww, filename, ext, is_dcm, dcm_meta)."""
    st.sidebar.markdown('<div class="sidebar-chapter">📂 Image Source</div>', unsafe_allow_html=True)
    # Default = Folder (index 0)
    source = st.sidebar.radio("Load from", ["Folder ./images/", "Upload file"],
                              key=f"{key_prefix}_source")

    im_raw = None
    filename = ""
    ext = "png"
    is_dcm = False
    dcm_meta = {}
    tmp_path = None

    accepted = ["bmp", "png", "jpg", "jpeg", "tif", "tiff"]
    if DICOM_AVAILABLE:
        accepted += ["dcm", "dicom"]

    if source == "Folder ./images/":
        img_dir = "./images"
        if os.path.isdir(img_dir):
            valid_ext = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
            if DICOM_AVAILABLE:
                valid_ext += tuple(DICOM_EXTS)
            files = sorted([f for f in os.listdir(img_dir)
                            if f.lower().endswith(valid_ext) and
                            os.path.isfile(os.path.join(img_dir, f))])
            if files:
                filename = st.sidebar.selectbox("Select image", files, key=f"{key_prefix}_pick")
                tmp_path = os.path.join(img_dir, filename)
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
                if DICOM_AVAILABLE and ext in [e.strip(".") for e in DICOM_EXTS]:
                    is_dcm = True
                if not is_dcm:
                    try:
                        pil = PILImage.open(tmp_path).convert("RGB")
                        im_raw = np.array(pil, dtype=float)
                    except Exception as e:
                        st.sidebar.error(f"Cannot open: {e}")
            else:
                st.sidebar.warning("No images found in ./images/")
        else:
            st.sidebar.warning("Folder ./images/ not found next to the script.")
    else:  # Other folder / Upload
        img_dir = st.sidebar.text_input("Directory path", value="", placeholder="e.g. C:/Users/me/Pictures", key=f"{key_prefix}_dir")
        if img_dir.strip():
            if os.path.isdir(img_dir):
                valid_ext = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
                if DICOM_AVAILABLE:
                    valid_ext += tuple(DICOM_EXTS)
                files = sorted([f for f in os.listdir(img_dir)
                                if f.lower().endswith(valid_ext) and
                                os.path.isfile(os.path.join(img_dir, f))])
                if files:
                    filename = st.sidebar.selectbox("Select image", files, key=f"{key_prefix}_pick")
                    tmp_path = os.path.join(img_dir, filename)
                    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
                    if DICOM_AVAILABLE and ext in [e.strip(".") for e in DICOM_EXTS]:
                        is_dcm = True
                    if not is_dcm:
                        try:
                            pil = PILImage.open(tmp_path).convert("RGB")
                            im_raw = np.array(pil, dtype=float)
                        except Exception as e:
                            st.sidebar.error(f"Cannot open: {e}")
                else:
                    st.sidebar.warning("No supported images found in that folder.")
            else:
                st.sidebar.error(f"Folder not found: `{img_dir}`")
        st.sidebar.markdown("**— or upload directly —**")
        up = st.sidebar.file_uploader("Choose image", type=accepted, key=f"{key_prefix}_upload")
        #
        if up is not None:
            filename = up.name
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, filename)
            with open(tmp_path, "wb") as fh:
                fh.write(up.getbuffer()) #changed from: see below # error: drag 'n drop didnt work
            #tmp_path = f"/tmp/{filename}"
            #with open(tmp_path, "wb") as fh:
                #fh.write(up.getbuffer())

            if DICOM_AVAILABLE and ext in [e.strip(".") for e in DICOM_EXTS]:
                is_dcm = True
            if not is_dcm:
                try:
                    pil = PILImage.open(tmp_path).convert("RGB")
                    im_raw = np.array(pil, dtype=float)
                except Exception as e:
                    st.sidebar.error(f"Cannot open image: {e}")

    ###############
    # ── DICOM: load and normalise pixel array to 0-255 ──────────────────────
    if is_dcm and tmp_path:
        try:
            dcm_arr, dcm_meta, dcm_ds = load_dicom(tmp_path)
            wc_dcm, ww_dcm = dicom_default_window(dcm_arr, dcm_ds)
            im_raw = apply_dicom_window(dcm_arr, wc_dcm, ww_dcm)
            ext = "png" # DICOM always saved as PNG
        except Exception as e:
            st.sidebar.error(f"DICOM error: {e}")
            is_dcm = False
    
        # ── Convert to grayscale float in 0-255 ─────────────────────────────────
    im = None
    if im_raw is not None:
        if im_raw.ndim == 3:
            im = rgb2gray(im_raw)
        else:
            im = im_raw.copy()
        im = np.asarray(im, dtype=float)
        # Normalise standard images to 0-255
        mn_i, mx_i = im.min(), im.max()
        if mx_i > mn_i:
            im = (im - mn_i) / (mx_i - mn_i) * 255.0

    # ── Unified window sliders — always 0-255, for every image type ─────────
    st.sidebar.markdown('<div class="sidebar-chapter">🪟 Window Parameters</div>', unsafe_allow_html=True)
    wc = st.sidebar.slider("Window Center (wc)", 0, 255, 128, key=f"{key_prefix}_wc",
                           help="Centre of the display window. Shifts brightness up or down.")
    ww = st.sidebar.slider("Window Width  (ww)", 1, 255, 255, key=f"{key_prefix}_ww",
                           help="Width of the display window. Narrows the visible intensity range, boosting contrast. Wider = softer contrast; narrower = harsher contrast.")

    return im, wc, ww, filename, ext, is_dcm, dcm_meta


def save_download_button(label, arr, filename_base, ext, key):
    """Renders a download button for a processed image."""
    save_ext = "png" if ext in ("dcm", "dicom", "tif", "tiff") else ext
    mime_map = {"png": "image/png", "bmp": "image/bmp",
                "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    mime = mime_map.get(save_ext, "image/png")
    data, actual_ext = arr_to_bytes(arr, save_ext)
    st.download_button(label, data=data,
                       file_name=f"{filename_base}_processed.{actual_ext}",
                       mime=mime, key=key)


# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 1 — DISPLAY MODIFICATION METHODS
# ─────────────────────────────────────────────────────────────────────────────

def ch1_display_methods():
    st.markdown("""
    <div class="chapter-header">
      <h2>Chapter 1 · Display Modification Methods</h2>
      <p>Linear windowing, non-linear tone mapping, and histogram manipulation techniques.</p>
    </div>""", unsafe_allow_html=True)

    im, wc, ww, filename, ext, is_dcm, dcm_meta = image_loader_widget("ch1")

    if im is None:
        st.markdown('<div class="info-box">👈 Load an image from the sidebar to get started.</div>',
                    unsafe_allow_html=True)
        return

    st.sidebar.markdown('<div class="sidebar-chapter">⚙️ Method</div>', unsafe_allow_html=True)

    category = st.sidebar.selectbox("Category",
        ["Linear Windowing", "Non-Linear Windowing", "Histogram Manipulation"], key="ch1_cat")

    LINEAR = ["Optimal Display", "Simple Window", "Broken Window", "Double Window"]
    NONLIN = ["Inverse", "Logarithmic", "Inverse Logarithmic", "Power", "Sine-Window", "Exp-Window"]
    HISTOG = ["Histogram Equalization", "CDF Equalization", "CLAHE"]

    # ── Params depending on category ──
    if category == "Linear Windowing":
        method = st.sidebar.selectbox("Method", LINEAR, key="ch1_linear_method")

        if method == "Simple Window":
            sw_wc = st.sidebar.slider("Window Center", 0, 255, 128, key="ch1_sw_wc",
                                      help="Centre of the linear window applied to the image.")
            sw_ww = st.sidebar.slider("Window Width", 1, 255, 200, key="ch1_sw_ww",
                                      help="Range of input intensities mapped to the full 0–255 output. Smaller = higher contrast.")
        elif method == "Broken Window":
            gray_val = st.sidebar.slider("Gray Value", 0, 255, 128, key="ch1_bw_gv",
                                         help="Output gray level at the break point — the intensity at which the slope changes.")
            im_val = st.sidebar.slider("Image Value", 1, 254, 70, key="ch1_bw_iv",
                                       help="Input intensity at the break point. Values below this use one slope; values above use another.")
        elif method == "Double Window":
            wl1 = st.sidebar.slider("Window Level 1", 0, 255, 50, key="ch1_dw_wl1",
                                    help="Centre of the first (lower) window — targets dark structures.")
            ww1 = st.sidebar.slider("Window Width 1", 1, 255, 100, key="ch1_dw_ww1",
                                    help="Width of the first window. Controls contrast in the dark region.")
            wl2 = st.sidebar.slider("Window Level 2", 0, 255, 200, key="ch1_dw_wl2",
                                    help="Centre of the second (upper) window — targets bright structures.")
            ww2 = st.sidebar.slider("Window Width 2", 1, 255, 100, key="ch1_dw_ww2",
                                    help="Width of the second window. Controls contrast in the bright region.")

    elif category == "Non-Linear Windowing":
        method = st.sidebar.selectbox("Function", NONLIN, key="ch1_nonlin_method")

    else:  # Histogram
        method = st.sidebar.selectbox("Method", HISTOG, key="ch1_hist_method")
        if method == "CLAHE":
            clip_limit = st.sidebar.slider("Clip Limit", 0.005, 0.10, 0.01, 0.01, key="ch1_clahe",
                                           help="Limits the height of local histograms to prevent the over-amplification of noise. 0.01 = Subtle contrast enhancement with minimal noise. 0.1 = Higher contrast, may show slight artifacts or increased noise.")
    run = st.sidebar.button("▶  Apply", key="ch1_run")

    # ── Display DICOM metadata ──
    if is_dcm and dcm_meta:
        with st.expander("🩻 DICOM Metadata", expanded=False):
            for k, v in dcm_meta.items():
                st.write(f"**{k}:** {v}")

    # Always show original with the display window applied
    im_preview = _simple_window(im, wc, ww)
    col_orig, col_proc = st.columns(2)
    with col_orig:
        st.subheader("Original")
        fig_o, ax_o = plt.subplots(figsize=(5, 5))
        ax_o.imshow(im_preview, cmap="gray", vmin=0, vmax=255); ax_o.axis("off")
        st.pyplot(fig_o, width='stretch')
        plt.close(fig_o)

    if not run:
        with col_proc:
            st.markdown('<div class="info-box">Configure method in sidebar and press Apply.</div>',
                        unsafe_allow_html=True)
        return

    # ── Processing ──
    im_norm = 255 * im / max(np.max(im), 1e-9)
    im1 = None
    y_init, w_func = None, None
    txt = method

    if category == "Linear Windowing":
        if method == "Optimal Display":
            vmn, vmx = np.min(im), np.max(im)
            im1 = np.round((TONES - 1) * (im - vmn) / max(vmx - vmn, 1e-9))
            wc_opt = int((vmx + vmn) / 2); ww_opt = int(vmx - vmn)
            y_init, w_func = _windowing_function(0, wc=wc_opt, ww=ww_opt)

        elif method == "Simple Window":
            im1 = _simple_window(im, sw_wc, sw_ww)
            y_init, w_func = _windowing_function(1, wc=sw_wc, ww=sw_ww)

        elif method == "Broken Window":
            im1 = _broken_window(im, gray_val, im_val)
            y_init, w_func = _windowing_function(2, gray_val=gray_val, im_val=im_val)

        elif method == "Double Window":
            im1 = _double_window(im, ww1, wl1, ww2, wl2)
            y_init, w_func = _windowing_function(3, ww1=ww1, wl1=wl1, ww2=ww2, wl2=wl2)

    elif category == "Non-Linear Windowing":
        idx = NONLIN.index(method)
        im1, y_init, w_func = _nonlinear_process(im, idx)

    else:  # Histogram
        if method == "Histogram Equalization":
            im1 = _hist_equalization(im_norm)
        elif method == "CDF Equalization":
            im1 = _cdf_equalization(im_norm)
        elif method == "CLAHE":
            try:
                from skimage import exposure
                im1 = exposure.equalize_adapthist(to_uint8(im_norm), clip_limit=clip_limit) * 255.0
            except ImportError:
                st.error("scikit-image required for CLAHE. Run: pip install scikit-image")
                return
        y_init = _hist_cumsum(im_norm)
        w_func = _hist_cumsum(im1)

    if im1 is None:
        return

    im1_disp = 255 * im1 / max(np.max(im1), 1e-9)
    # Apply the global window sliders to the processed image too
    im1_windowed = _simple_window(im1_disp, wc, ww, image_depth=255, tones=256)
    im1_windowed = np.clip(im1_windowed, 0, 255)

    with col_proc:
        st.subheader(f"Processed · {txt}") ### edited out = (wc:{wc}, ww:{ww})
        fig_p, ax_p = plt.subplots(figsize=(5, 5))
        ax_p.imshow(im1_windowed, cmap="gray", vmin=0, vmax=255)
        ax_p.axis("off")
        st.pyplot(fig_p, width='stretch')
        plt.close(fig_p)

    # ── Function / histogram plots ──
    tabs = st.tabs(["📈 Display Function", "📊 Histograms"])
    x_ax = np.arange(IMAGE_DEPTH)

    with tabs[0]:
        fig_f = Figure(figsize=(10, 4), facecolor="#0d1526")
        ax1 = fig_f.add_subplot(1, 2, 1)
        ax2 = fig_f.add_subplot(1, 2, 2)
        if y_init is not None:
            ax1.plot(x_ax, y_init, color="#6cb6ff", lw=1.5, label="Initial")
            ax1.set_title("Initial Display Function"); ax1.grid(True, alpha=0.3)
            ax1.set_xlabel("Input intensity"); ax1.set_ylabel("Output intensity")
        if w_func is not None:
            ax2.plot(x_ax if len(w_func) == IMAGE_DEPTH else np.arange(len(w_func)),
                     w_func, color="#f78166", lw=1.5, label="Processed")
            ax2.set_title(f"Processed · {txt}"); ax2.grid(True, alpha=0.3)
        fig_f.tight_layout(pad=1.0)
        st.pyplot(fig_f, width='stretch')

    with tabs[1]:
        h_init = _histogram(im_norm)
        h_proc = _histogram(im1_windowed)
        fig_h = Figure(figsize=(10, 4), facecolor="#0d1526")
        ax_h1 = fig_h.add_subplot(1, 2, 1)
        ax_h2 = fig_h.add_subplot(1, 2, 2)
        ax_h1.plot(h_init, color="#6cb6ff"); ax_h1.set_title("Initial Histogram"); ax_h1.grid(True, alpha=0.3)
        ax_h2.plot(h_proc, color="#f78166"); ax_h2.set_title(f"Processed Histogram (wc={wc}, ww={ww})"); ax_h2.grid(True, alpha=0.3)
        fig_h.tight_layout(pad=1.0)
        st.pyplot(fig_h, width='stretch')

    # ── Downloads ──
    st.markdown("---")
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        fig_full = Figure(figsize=(14, 5), facecolor="#0d1526")
        gss = fig_full.subplots(1, 3)
        gss[0].imshow(im_preview, cmap="gray", vmin=0, vmax=255); gss[0].set_title("Original"); gss[0].axis("off")
        gss[1].imshow(im1_windowed, cmap="gray", vmin=0, vmax=255)
        gss[1].set_title(f"Processed · {txt}  wc={wc}, ww={ww}"); gss[1].axis("off")
        if y_init is not None:
            gss[2].plot(x_ax, y_init, color="#6cb6ff", label="Initial")
        if w_func is not None:
            gss[2].plot(np.arange(len(w_func)), w_func, color="#f78166", label="Processed")
        gss[2].set_title("Display Function"); gss[2].legend(fontsize=7); gss[2].grid(True, alpha=0.3)
        fig_full.tight_layout()
        st.download_button("⬇ Download full plot (PNG)", data=fig_to_bytes(fig_full),
                           file_name=f"{base}_plot.png", mime="image/png", key="ch1_dl_plot")
        plt.close(fig_full)
    with dcol2:
        save_download_button("⬇ Download processed image", im1_windowed, base, ext, key="ch1_dl_img")


# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 2 — SPATIAL DOMAIN FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def ch2_spatial_filters():
    st.markdown("""
    <div class="chapter-header">
      <h2>Chapter 2 · Filters in the Space Domain</h2>
      <p>Convolution-based filters: smoothing, Laplacian, high-emphasis, and median filtering.</p>
    </div>""", unsafe_allow_html=True)

    im, wc, ww, filename, ext, is_dcm, dcm_meta = image_loader_widget("ch2")

    if im is None:
        st.markdown('<div class="info-box">👈 Load an image from the sidebar to get started.</div>',
                    unsafe_allow_html=True)
        return

    st.sidebar.markdown('<div class="sidebar-chapter">⚙️ Filter</div>', unsafe_allow_html=True)

    FILTERS = ["Smoothing", "Laplacian", "High Emphasis", "Median"]
    filter_choice = st.sidebar.selectbox("Filter type", FILTERS, key="ch2_filter",
                                         help= "Smoothing: Removes noise and artifacts, but worsens image definition, Laplacian: Shows borders/edges between two different structures, High Emphasis: Improves image definition, but enhances noise, Median: Acts as a smoothing filter.")

    # ── Kernel editors in sidebar ────────────────────────────────────────────
    he_choice = None
    c_center, c_cross, c_corner = 15, -2, -1

    if filter_choice == "Smoothing":
        st.sidebar.markdown('<div class="sidebar-chapter">🧩 Smoothing Kernel</div>', unsafe_allow_html=True)
        smooth_preset = st.sidebar.radio("Kernel preset",
            ["Cross (0,1 centre)", "Full 3×3 box", "Gaussian-like (1,2,1)", "Custom"],
            key="ch2_smooth_preset",
            help="Choose the kernel of the filter to be applied to the image.")
        if smooth_preset == "Custom":
            s_center = st.sidebar.number_input("Centre value", value=1, key="ch2_sc")
            s_cross  = st.sidebar.number_input("Cross value",  value=1, key="ch2_sx")
            s_corner = st.sidebar.number_input("Corner value", value=0, key="ch2_so")

    elif filter_choice == "High Emphasis":
        st.sidebar.markdown('<div class="sidebar-chapter">🧩 HE Kernel</div>', unsafe_allow_html=True)
        he_choice = st.sidebar.radio("Kernel preset",
            ["HE1 – cross −1, centre 5",
             "HE2 – full −1, centre 9",
             "HE3 – cross −2, corner −1, centre 13",
             "HE4 – Custom"], key="ch2_he_preset")
        if he_choice.startswith("HE4"):
            c_center = st.sidebar.number_input("Centre value", value=15, key="ch2_cc")
            c_cross  = st.sidebar.number_input("Cross value",  value=-2, key="ch2_cx")
            c_corner = st.sidebar.number_input("Corner value", value=-1, key="ch2_co")

    run = st.sidebar.button("▶  Apply Filter", key="ch2_run")

    if is_dcm and dcm_meta:
        with st.expander("🩻 DICOM Metadata", expanded=False):
            for k, v in dcm_meta.items():
                st.write(f"**{k}:** {v}")

    im_preview = _simple_window(im, wc, ww)
    col_orig, col_proc = st.columns(2)
    with col_orig:
        st.subheader("Original")
        fig_o, ax_o = plt.subplots(figsize=(5, 5))
        ax_o.imshow(im_preview, cmap="gray", vmin=0, vmax=255); ax_o.axis("off")
        st.pyplot(fig_o, width='stretch')
        plt.close(fig_o)

    if not run:
        with col_proc:
            st.markdown('<div class="info-box">Configure filter in sidebar and press Apply Filter.</div>',
                        unsafe_allow_html=True)
        return

    image_depth = 255
    im1 = None
    im1_windowed = None
    kernel_used = None

    if filter_choice == "Smoothing":
        smooth_kernels = {
            "Cross (0,1 centre)":    np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=float),
            "Full 3×3 box":          np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=float),
            "Gaussian-like (1,2,1)": np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float),
        }
        if smooth_preset == "Custom":
            kernel = np.array([[s_corner, s_cross,  s_corner],
                               [s_cross,  s_center, s_cross],
                               [s_corner, s_cross,  s_corner]], dtype=float)
        else:
            kernel = smooth_kernels[smooth_preset]
        sK = np.sum(kernel)
        if sK != 0: kernel_norm = kernel / sK
        else: kernel_norm = kernel
        im1 = signal.convolve2d(im, kernel_norm, mode="same")
        im1_windowed = _simple_window(im1, wc, ww, image_depth=image_depth, tones=256)
        kernel_used = kernel  # store unnormalised for display

    elif filter_choice == "Laplacian":
        kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
        im1 = signal.convolve2d(im, kernel, mode="same")
        im1 = np.clip(im1, 0, 255)
        im1 = imNormalize(im1, 256)
        im1_windowed = _simple_window(im1, wc, ww, image_depth=image_depth, tones=256)
        kernel_used = kernel

    elif filter_choice == "High Emphasis":
        kernels = {
            "HE1": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
            "HE2": np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=float),
            "HE3": np.array([[-1,-2,-1],[-2,13,-2],[-1,-2,-1]], dtype=float),
        }
        if he_choice.startswith("HE1"):
            kernel = kernels["HE1"]
        elif he_choice.startswith("HE2"):
            kernel = kernels["HE2"]
        elif he_choice.startswith("HE3"):
            kernel = kernels["HE3"]
        else:
            kernel = np.array([[c_corner, c_cross, c_corner],
                               [c_cross,  c_center, c_cross],
                               [c_corner, c_cross, c_corner]], dtype=float)
        sK = np.sum(kernel)     ## delete these two?
        if sK > 0: kernel /= sK ##
        im1 = signal.convolve2d(im, kernel, mode="same")
        im1 = np.clip(im1, 0, 255)
        im1 = imNormalize(im1, 256)
        im1_windowed = _simple_window(im1, wc, ww, image_depth=image_depth, tones=256)
        kernel_used = kernel

    elif filter_choice == "Median":
        im1 = signal.medfilt2d(im.astype(float), (5, 5))
        im1_windowed = _simple_window(im1, wc, ww, image_depth=image_depth, tones=256)

    # Display results
    with col_proc:
        st.subheader(f"Filtered · {filter_choice}")
        fig_p, ax_p = plt.subplots(figsize=(5, 5))
        disp = im1_windowed if im1_windowed is not None else im1
        ax_p.imshow(disp, cmap="gray", vmin=0, vmax=255)
        ax_p.axis("off")
        st.pyplot(fig_p, width='stretch')
        plt.close(fig_p)

    if kernel_used is not None:
        with st.expander("🔍 Kernel used", expanded=True):
            st.code(str(np.round(kernel_used, 4)))

    fig_cmp = Figure(figsize=(14, 5), facecolor="#0d1526")
    axes = fig_cmp.subplots(1, 3)
    axes[0].imshow(im_preview, cmap="gray", vmin=0, vmax=255); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(np.clip(im1, 0, 255), cmap="gray")
    axes[1].set_title(f"{filter_choice} (raw)"); axes[1].axis("off")
    axes[2].imshow(im1_windowed, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"+ Window wc={wc}, ww={ww}"); axes[2].axis("off")
    fig_cmp.tight_layout()
    st.pyplot(fig_cmp, width='stretch')
    plt.close(fig_cmp)

    # Downloads
    st.markdown("---")
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    final_arr = im1_windowed if im1_windowed is not None else im1
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        fig_dl = Figure(figsize=(12, 5), facecolor="#0d1526")
        a = fig_dl.subplots(1, 3 if im1_windowed is not None else 2)
        a[0].imshow(im_preview, cmap="gray", vmin=0, vmax=255); a[0].set_title("Original"); a[0].axis("off")
        a[1].imshow(np.clip(im1, 0, 255), cmap="gray")
        a[1].set_title(f"{filter_choice}"); a[1].axis("off")
        if im1_windowed is not None:
            a[2].imshow(im1_windowed, cmap="gray", vmin=0, vmax=255)
            a[2].set_title(f"+ Windowed"); a[2].axis("off")
        fig_dl.tight_layout()
        st.download_button("⬇ Download full plot (PNG)", data=fig_to_bytes(fig_dl),
                           file_name=f"{base}_spatial_filter_plot.png",
                           mime="image/png", key="ch2_dl_plot")
        plt.close(fig_dl)
    with dcol2:
        save_download_button("⬇ Download processed image", final_arr, base, ext, key="ch2_dl_img")


# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 3 — FREQUENCY DOMAIN FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def ch3_frequency_filters():
    st.markdown("""
    <div class="chapter-header">
      <h2>Chapter 3 · Filters in the Frequency Domain</h2>
      <p>Ideal, Butterworth, Exponential, and Gaussian filters — LP, HP, BP, BR bands. Plus image restoration.</p>
    </div>""", unsafe_allow_html=True)

    im, wc, ww, filename, ext, is_dcm, dcm_meta = image_loader_widget("ch3")

    if im is None:
        st.markdown('<div class="info-box">👈 Load an image from the sidebar to get started.</div>',
                    unsafe_allow_html=True)
        return

    st.sidebar.markdown('<div class="sidebar-chapter">⚙️ Filter Design</div>', unsafe_allow_html=True)
    filter_shape = st.sidebar.selectbox("Filter shape",
        ["Ideal", "Butterworth", "Exponential", "Gaussian"], key="ch3_shape")
    filter_type  = st.sidebar.selectbox("Filter band",
        ["Low Pass", "High Pass", "Band-Reject", "Band-Pass"], key="ch3_band")
    ndegree = st.sidebar.slider("Order / degree", 1, 6, 2, key="ch3_order")
    fco_pct = st.sidebar.slider("Cut-off freq (% of max)", 1, 60, 30, key="ch3_fco",
                                help="For Low-Pass and High-Pass Filters. Controls the cut-off frequency as a percentage of the maximum spatial frequency. Applies to all filter shapes.")

    st.sidebar.markdown('<div class="sidebar-chapter">🛠 Restoration</div>', unsafe_allow_html=True)
    do_restore = st.sidebar.checkbox("Enable image restoration", False, key="ch3_restore")
    pct_noise  = st.sidebar.slider("Noise level (%)", 0.0, 1.0, 0.05, 0.01,
                                   disabled=not do_restore, key="ch3_noise")
    rl_iters   = st.sidebar.slider("Richardson-Lucy iterations", 1, 20, 5,
                                   disabled=not do_restore, key="ch3_rl",
                                   help="Number of R-L deconvolution iterations. More iterations = sharper result but more noise. 5-15 typically works well.")
    wiener_c   = st.sidebar.slider("Wiener balance (c)", 0.0001, 0.1, 0.001, 0.0001,
                                   format="%.4f", disabled=not do_restore, key="ch3_wc2",
                                   help="Regularisation/balance parameter for the Wiener filter. Lower values = stronger restoration; higher values = more noise suppression.")

    run = st.sidebar.button("▶  Run Filter", key="ch3_run")

    if is_dcm and dcm_meta:
        with st.expander("🩻 DICOM Metadata", expanded=False):
            for k, v in dcm_meta.items():
                st.write(f"**{k}:** {v}")

    M, N_im = im.shape
    Flength = int(np.round(np.sqrt(M * M + N_im * N_im)))
    im_norm = imNormalize(im, 256).astype(float)

    FILTER_MAP = {"Low Pass": 1, "High Pass": 2, "Band-Reject": 3, "Band-Pass": 4}
    FILTER = FILTER_MAP[filter_type]
    fco    = int(round(Flength * fco_pct / 100))
    trans  = int(round(Flength * 0.25))
    w_bp, enh = 0, 0.0

    if FILTER == 1:   trans = int(round(Flength / 2))
    elif FILTER == 2: trans = int(round(Flength * 0.25)); enh = 0.4
    elif FILTER == 3: trans = int(round(Flength * 0.2)); enh = 0; w_bp = 15
    elif FILTER == 4: trans = int(round(Flength * 0.2))

    shape_map = {"Ideal": 1, "Butterworth": 2, "Exponential": 3, "Gaussian": 4}
    fs = shape_map[filter_shape]

    if fs == 1:   fh, sText = _make_ideal(Flength, fco, FILTER, enh, trans, w_bp)
    elif fs == 2: fh, sText = _make_butterworth(Flength, ndegree, fco, FILTER, trans)
    elif fs == 3: fh, sText = _make_exponential(Flength, ndegree, fco, FILTER, trans)
    else:         fh, sText = _make_gaussian(Flength, ndegree, fco, FILTER, trans)

    FH = _design_2d_filter(im_norm, fh)
    im_filtered = _filter_image(im_norm, FH)
    im_filtered = imNormalize(im_filtered, 256)
    im_windowed = _simple_window(im_filtered, wc, ww, image_depth=255, tones=256)
    im_windowed = np.clip(im_windowed, 0, 255)
    fft_orig    = _ampl_fft2(im_norm)

    col_m1, col_m2, col_m3 = st.columns(3)
    base = filename.rsplit(".", 1)[0] if "." in filename else filename

    for col, (lbl, val) in zip([col_m1, col_m2, col_m3], [
        ("Image", f"{M}×{N_im} px"),
        ("Filter", sText),
        ("Window", f"wc={wc}, ww={ww}"),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                        f'<div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

    tab_filter, tab_restore = st.tabs(["🧬 Frequency Filtering", "🛠 Restoration"])

    with tab_filter:
        fig1 = Figure(figsize=(16, 6), facecolor="#0d1526")
        gs1 = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.06)
        ax_o = fig1.add_subplot(gs1[0])
        ax_f = fig1.add_subplot(gs1[1])
        ax_s = fig1.add_subplot(gs1[2])
        ax_o.imshow(im_norm, cmap="gray", vmin=0, vmax=255); ax_o.axis("off"); ax_o.set_title("Original")
        ax_f.imshow(im_windowed, cmap="gray", vmin=0, vmax=255); ax_f.axis("off")
        ax_f.set_title(f"Filtered · {sText}  wc={wc}, ww={ww}")
        ax_s.imshow(fft_orig, cmap="bone"); ax_s.axis("off"); ax_s.set_title("FFT Spectrum")
        fig1.tight_layout(pad=0.8)
        st.pyplot(fig1, width='stretch')
        plt.close(fig1)

        fig2 = Figure(figsize=(16, 4), facecolor="#0d1526")
        gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.3)
        ax1d = fig2.add_subplot(gs2[0])
        ax2d = fig2.add_subplot(gs2[1])
        xax = np.arange(len(fh))
        ax1d.plot(xax, fh, color="#6cb6ff", lw=1.8, label="Filter")
        ax1d.plot(xax, np.fft.fftshift(fh), color="#f78166", lw=1.2, ls="--", label="Shifted")
        ax1d.axvline(len(fh)//2, color="#3ead4d", lw=1.5, ls=":", label="Centre")
        ax1d.set_title(f"1-D Filter: {sText}"); ax1d.grid(True, alpha=0.3)
        ax1d.legend(fontsize=8); ax1d.set_xlabel("Spatial frequency")
        ax2d.imshow(np.fft.fftshift(FH), cmap="bone"); ax2d.axis("off")
        ax2d.set_title(f"2-D Filter: {sText}")
        fig2.tight_layout(pad=0.8)
        st.pyplot(fig2, width='stretch')
        plt.close(fig2)

        st.markdown("---")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            fig_dl = Figure(figsize=(16, 10), facecolor="#0d1526")
            gs_dl = gridspec.GridSpec(2, 3, figure=fig_dl, wspace=0.1, hspace=0.3)
            dl_items = [(im_norm,"Original",{"cmap":"gray","vmin":0,"vmax":255}),
                        (im_windowed,f"Filtered · {sText}",{"cmap":"gray","vmin":0,"vmax":255}),
                        (fft_orig,"FFT Spectrum",{"cmap":"bone"}),
                        (np.fft.fftshift(FH),f"2-D Filter",{"cmap":"bone_r"})]
            for idx_dl, (arr_dl, ttl_dl, kw_dl) in enumerate(dl_items):
                r_dl, c_dl = divmod(idx_dl, 3)
                ax_dl = fig_dl.add_subplot(gs_dl[r_dl, c_dl])
                ax_dl.imshow(arr_dl, **kw_dl); ax_dl.set_title(ttl_dl); ax_dl.axis("off")
            ax_line = fig_dl.add_subplot(gs_dl[1, 1:3])
            ax_line.plot(xax, fh, color="#6cb6ff", lw=1.5, label="Filter")
            ax_line.plot(xax, np.fft.fftshift(fh), color="#f78166", lw=1, ls="--", label="Shifted")
            ax_line.axvline(len(fh)//2, color="#3fb950", lw=1.5, ls=":", label="Centre")
            ax_line.set_title(f"1-D Filter: {sText}"); ax_line.grid(True, alpha=0.3); ax_line.legend(fontsize=7)
            fig_dl.tight_layout()
            st.download_button("⬇ Full plot (PNG)", data=fig_to_bytes(fig_dl),
                               file_name=f"{base}_freq_filter_plot.png",
                               mime="image/png", key="ch3_dl_plot")
            plt.close(fig_dl)
        with dcol2:
            save_download_button("⬇ Processed image", im_windowed, base, ext, key="ch3_dl_img")

    with tab_restore:
        if not do_restore:
            st.info("Enable **Image Restoration** in the sidebar to use this tab.")
        else:
            if not run:
                st.info("Press ▶ Run Filter to start restoration.")
            else:
                try:
                    from skimage import restoration as skr
                    skimage_ok = True
                except ImportError:
                    skimage_ok = False
                    st.warning("scikit-image not found. Install: pip install scikit-image")

                with st.spinner("Running degradation & restoration…"):
                    Fl2 = np.sqrt(M * M + N_im * N_im)
                    fh_mtf = _gaussian_mtf(Fl2)
                    FH_mtf = _from1d_to_2d(im_norm, fh_mtf)
                    im_blur = _blur_image(im_norm, FH_mtf)

                    im_noisy = im_blur.copy().astype(float)
                    noise = pct_noise * im_noisy * np.random.rand(*im_noisy.shape)
                    im_noisy = im_noisy + noise
                    im_noisy = imNormalize(im_noisy, 256).astype(float)

                    PSF = np.fft.fftshift(np.abs(np.fft.ifft2(FH_mtf)))
                    PSF = PSF / max(np.max(PSF), 1e-9)

                    diag_orig = np.diag(im_norm[:min(M, N_im), :min(M, N_im)])
                    diag_deg  = np.diag(im_noisy[:min(M, N_im), :min(M, N_im)])
                    SE_d = float(np.sqrt(np.sum((diag_orig - diag_deg) ** 2) / max(len(diag_orig), 1)))

                    wiener_w = unsup_w = rl_w = None
                    SE1 = SE2 = SE3 = None

                    if skimage_ok:
                        try:
                            wiener_im = skr.wiener(im_noisy, PSF, wiener_c, is_real=True, clip=False)
                            wiener_im = imNormalize(wiener_im, 256)
                            wiener_w  = _simple_window(wiener_im, wc, ww, 255, 256)
                            SE1 = float(np.sqrt(np.sum((np.diag(im_norm[:min(M,N_im),:min(M,N_im)]) -
                                                         np.diag(wiener_im[:min(M,N_im),:min(M,N_im)]))**2) /
                                                max(min(M,N_im), 1)))

                            unsup_im, _ = skr.unsupervised_wiener(im_noisy, PSF, clip=False)
                            unsup_im = imNormalize(unsup_im, 256)
                            unsup_w  = _simple_window(unsup_im, wc, ww, 255, 256)
                            SE2 = float(np.sqrt(np.sum((np.diag(im_norm[:min(M,N_im),:min(M,N_im)]) -
                                                         np.diag(unsup_im[:min(M,N_im),:min(M,N_im)]))**2) /
                                                max(min(M,N_im), 1)))

                            rl_im = skr.richardson_lucy(im_noisy, PSF, num_iter=rl_iters, clip=False)
                            rl_im = imNormalize(rl_im, 256)
                            rl_w  = _simple_window(rl_im, wc, ww, 255, 256)
                            SE3 = float(np.sqrt(np.sum((np.diag(im_norm[:min(M,N_im),:min(M,N_im)]) -
                                                         np.diag(rl_im[:min(M,N_im),:min(M,N_im)]))**2) /
                                                max(min(M,N_im), 1)))
                        except Exception as e:
                            st.warning(f"Restoration error: {e}")
                            skimage_ok = False

                rc1, rc2, rc3, rc4 = st.columns(4)
                for col, lbl, val in [
                    (rc1, "Std. Error · Degraded", f"{SE_d:.2f}"),
                    (rc2, "Std. Error · Wiener",   f"{SE1:.2f}" if SE1 else "—"),
                    (rc3, "Std. Error · Unsup-W",  f"{SE2:.2f}" if SE2 else "—"),
                    (rc4, "Std. Error · RL",        f"{SE3:.2f}" if SE3 else "—"),
                ]:
                    with col:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                                    f'<div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

                fig_r1 = Figure(figsize=(16, 6), facecolor="#0d1526")
                ar1 = fig_r1.subplots(1, 3)
                ar1[0].imshow(np.clip(im_norm, 0, 255), cmap="gray", vmin=0, vmax=255)
                ar1[0].set_title("Original"); ar1[0].axis("off")
                ar1[1].imshow(np.clip(im_noisy, 0, 255), cmap="gray", vmin=0, vmax=255)
                ar1[1].set_title(f"Degraded  SE={SE_d:.2f}"); ar1[1].axis("off")
                ar1[2].plot(fh_mtf[:len(fh_mtf)//2], color="#6cb6ff", lw=1.5)
                ar1[2].set_title("Gaussian MTF"); ar1[2].grid(True, alpha=0.3)
                fig_r1.tight_layout(pad=0.8)
                st.pyplot(fig_r1, width='stretch')
                plt.close(fig_r1)

                if skimage_ok and wiener_w is not None:
                    fig_r2 = Figure(figsize=(16, 6), facecolor="#0d1526")
                    ar2 = fig_r2.subplots(1, 3)
                    for ax_r, arr_r, ttl_r in [
                        (ar2[0], wiener_w,  f"Wiener  SE={SE1:.2f}"),
                        (ar2[1], unsup_w,   f"Unsup. Wiener  SE={SE2:.2f}"),
                        (ar2[2], rl_w,       f"R-Lucy ({rl_iters}it)  SE={SE3:.2f}"),
                    ]:
                        ax_r.imshow(np.clip(arr_r, 0, 255), cmap="gray", vmin=0, vmax=255)
                        ax_r.set_title(ttl_r); ax_r.axis("off")
                    fig_r2.tight_layout(pad=0.8)
                    st.pyplot(fig_r2, width='stretch')
                    plt.close(fig_r2)

                    st.markdown("---")
                    # Build a single combined figure with all 6 panels
                    fig_all = Figure(figsize=(20, 10), facecolor="#0d1526")
                    gs_all = gridspec.GridSpec(2, 3, figure=fig_all, wspace=0.08, hspace=0.28)

                    panels_top = [
                        (np.clip(im_norm,   0, 255), "Original",                       {"cmap": "gray", "vmin": 0, "vmax": 255}),
                        (np.clip(im_noisy,  0, 255), f"Degraded  SE={SE_d:.2f}",       {"cmap": "gray", "vmin": 0, "vmax": 255}),
                        (fh_mtf[:len(fh_mtf)//2],    "Gaussian MTF",                   None),
                    ]
                    panels_bot = [
                        (np.clip(wiener_w, 0, 255), f"Wiener  SE={SE1:.2f}",           {"cmap": "gray", "vmin": 0, "vmax": 255}),
                        (np.clip(unsup_w,  0, 255), f"Unsup. Wiener  SE={SE2:.2f}",    {"cmap": "gray", "vmin": 0, "vmax": 255}),
                        (np.clip(rl_w,     0, 255), f"R-Lucy ({rl_iters}it)  SE={SE3:.2f}", {"cmap": "gray", "vmin": 0, "vmax": 255}),
                    ]

                    for col_idx, (arr_p, ttl_p, kw_p) in enumerate(panels_top):
                        ax_p = fig_all.add_subplot(gs_all[0, col_idx])
                        if kw_p is not None:
                            ax_p.imshow(arr_p, **kw_p); ax_p.axis("off")
                        else:
                            ax_p.plot(arr_p, color="#6cb6ff", lw=1.5)
                            ax_p.grid(True, alpha=0.3)
                            ax_p.set_xlabel("Spatial frequency", color="#768390", fontsize=8)
                            ax_p.tick_params(colors="#768390", labelsize=7)
                        ax_p.set_title(ttl_p, color="#6cb6ff", fontsize=9, pad=5)

                    for col_idx, (arr_p, ttl_p, kw_p) in enumerate(panels_bot):
                        ax_p = fig_all.add_subplot(gs_all[1, col_idx])
                        ax_p.imshow(arr_p, **kw_p); ax_p.axis("off")
                        ax_p.set_title(ttl_p, color="#6cb6ff", fontsize=9, pad=5)

                    fig_all.suptitle(
                        f"Image Restoration — wc={wc}, ww={ww}  |  noise={pct_noise:.2f}  |  RL iters={rl_iters}",
                        color="#adbac7", fontsize=10, y=1.01
                    )
                    fig_all.tight_layout(pad=0.6)

                    st.download_button(
                        "⬇ Download all restored images (PNG)",
                        data=fig_to_bytes(fig_all),
                        file_name=f"{base}_restoration_all.png",
                        mime="image/png",
                        key="ch3_r_all",
                    )
                    plt.close(fig_all)

                    st.markdown("**Download individually:**")
                    ri1, ri2, ri3, ri4 = st.columns(4)
                    with ri1:
                        save_download_button("⬇ Degraded", im_noisy, base + "_degraded", ext, key="ch3_r_ind1")
                    with ri2:
                        save_download_button("⬇ Wiener", wiener_w, base + "_wiener", ext, key="ch3_r_ind2")
                    with ri3:
                        save_download_button("⬇ Unsup. Wiener", unsup_w, base + "_unsup_wiener", ext, key="ch3_r_ind3")
                    with ri4:
                        save_download_button("⬇ Richardson-Lucy", rl_w, base + "_rl", ext, key="ch3_r_ind4")


# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER 4 — TOMOGRAPHIC RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def ch4_reconstruction():
    st.markdown("""
    <div class="chapter-header">
      <h2>Chapter 4 · Tomographic Image Reconstruction</h2>
      <p>Filtered Back-Projection (FBP) and Algebraic Reconstruction Technique (ART / SART).</p>
    </div>""", unsafe_allow_html=True)

    if not SKIMAGE_AVAILABLE:
        st.error("scikit-image is required for this chapter. Run: pip install scikit-image")
        return

    im, wc, ww, filename, ext, is_dcm, dcm_meta = image_loader_widget("ch4")

    if im is None:
        st.markdown('<div class="info-box">👈 Load an image from the sidebar to get started.</div>',
                    unsafe_allow_html=True)
        return

    st.sidebar.markdown('<div class="sidebar-chapter">🔭 Projection</div>', unsafe_allow_html=True)
    angle_opt = st.sidebar.radio("View angles", ["180°", "360°", "Custom"], horizontal=True, key="ch4_angles",
                                 help="More angles improve reconstruction quality. 180° is standard for most CT-like scenarios; 360° captures full rotational data.")
    if angle_opt == "180°":
        n_angles = 180
    elif angle_opt == "360°":
        n_angles = 360
    else:
        n_angles = st.sidebar.number_input("Number of angles", 10, 720, 180, 10, key="ch4_custom_ang")

    st.sidebar.markdown('<div class="sidebar-chapter">⚙️ Method</div>', unsafe_allow_html=True)
    method = st.sidebar.radio("Algorithm", ["FBP", "ART"], horizontal=True, key="ch4_method",
                               help= "FBP is fast and works well with many projection angles. ART is iterative and can give better results with fewer angles, but is slower.")

    fbp_filter = None
    art_iters = None
    if method == "FBP":
        fbp_filter = st.sidebar.selectbox("FBP Filter",
            ["ramp", "shepp-logan", "cosine", "hamming", "hann"], key="ch4_fbp_filter",
            help="The filter applied to projections before back-projection. Ramp is the standard choice. Shepp-Logan reduces ringing. Hamming and Hann suppresses noise but reduces sharpness. Cosine is a mild compromise.")
    else:
        art_iters = st.sidebar.slider("SART iterations", 1, 30, 5, key="ch4_art_iters",
                                      help="Each iteration refines the reconstruction by comparing forward-projected estimates against the sinogram. More iterations sharpen detail but increase computation time. 3–10 iterations are generally sufficient, while more than 20 is usually impractical")

    run = st.sidebar.button("▶  Run Reconstruction", key="ch4_run")

    if is_dcm and dcm_meta:
        with st.expander("🩻 DICOM Metadata", expanded=False):
            for k, v in dcm_meta.items():
                st.write(f"**{k}:** {v}")

    M_im, N_im_c = im.shape

    mc1, mc2, mc3 = st.columns(3)
    for col, (lbl, val) in zip([mc1, mc2, mc3], [
        ("Image", filename or "—"),
        ("Angles", str(n_angles)),
        ("Method", method),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                        f'<div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

    col_orig, col_recon = st.columns(2)
    im_windowed_ch4 = np.clip(_simple_window(im, wc, ww), 0, 255)
    with col_orig:
        st.subheader("Original Image")
        fig_o, ax_o = plt.subplots(figsize=(7, 7))
        ax_o.imshow(im_windowed_ch4, cmap="gray", vmin=0, vmax=255); ax_o.axis("off")
        st.pyplot(fig_o, width='stretch')
        plt.close(fig_o)

    if not run:
        with col_recon:
            st.markdown('<div class="info-box" style="margin-top:2.5rem;">Configure parameters and press Run Reconstruction.</div>',
                        unsafe_allow_html=True)
        return

    # Sinogram
    with st.spinner(f"Computing sinogram ({n_angles} angles)…"):
        t0 = time.time()
        end = 180.0 if n_angles <= 180 else 360.0
        theta = np.linspace(0.0, end, n_angles, endpoint=False)
        sinogram = radon(im_windowed_ch4, theta=theta, circle=False)
        t_sino = time.time() - t0

    # Reconstruction
    with st.spinner(f"Running {method}…"):
        t0 = time.time()
        if method == "FBP":
            reconstruction = iradon(sinogram, theta=theta,
                                    filter_name=fbp_filter, output_size=len(im_windowed_ch4))
            method_label = f"FBP · {fbp_filter}"
        else:
            reconstruction = iradon_sart(sinogram, theta=theta)
            for _ in range(art_iters - 1):
                reconstruction = iradon_sart(sinogram, theta=theta, image=reconstruction)
            method_label = f"ART · {art_iters} iter."
        t_recon = time.time() - t0

    # Normalize reconstruction to 0-255 then apply the global window sliders
    mn_r, mx_r = reconstruction.min(), reconstruction.max()
    recon_norm = (reconstruction - mn_r) / max(mx_r - mn_r, 1e-9) * 255.0
    recon_windowed = np.clip(_simple_window(recon_norm, wc, ww, image_depth=255, tones=256), 0, 255)

    # Show reconstructed image next to original
    with col_recon:
        st.subheader(f"Reconstructed  ({method_label})  (wc={wc}, ww={ww})")
        fig_r, ax_r = plt.subplots(figsize=(7, 7))
        ax_r.imshow(recon_windowed, cmap="gray", vmin=0, vmax=255); ax_r.axis("off")
        st.pyplot(fig_r, width='stretch')
        plt.close(fig_r)

    # Timing metrics
    timing_c1, timing_c2 = st.columns(2)
    with timing_c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Sinogram time</div>'
                    f'<div class="metric-value">{t_sino:.2f}s</div></div>', unsafe_allow_html=True)
    with timing_c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Reconstruction time</div>'
                    f'<div class="metric-value">{t_recon:.2f}s</div></div>', unsafe_allow_html=True)

    # Sinogram — full width below images
    st.subheader("Sinogram")
    fig_s, ax_s = plt.subplots(figsize=(14, 4))
    ax_s.imshow(sinogram, cmap="gray", aspect="auto",
                extent=(theta[0], theta[-1], 0, sinogram.shape[0]))
    ax_s.set_xlabel("Projection angle (°)", color="#768390")
    ax_s.set_ylabel("Detector position", color="#768390")
    ax_s.tick_params(colors="#768390")
    fig_s.patch.set_facecolor("#0d1526")
    ax_s.set_facecolor("#111f35")
    st.pyplot(fig_s, width='stretch')
    plt.close(fig_s)

    st.markdown("---")
    st.subheader("Full Report")

    # Big figure — full report
    fig_full = plt.figure(figsize=(18, 12), facecolor="#0d1526")
    gs_f = gridspec.GridSpec(2, 3, figure=fig_full, wspace=0.25, hspace=0.35,
                             height_ratios=[1.4, 1])

    def _ax_style(ax, title):
        ax.set_facecolor("#111f35")
        ax.set_title(title, color="#6cb6ff", fontsize=10, pad=6)

    ax_f0 = fig_full.add_subplot(gs_f[0, 0])
    ax_f0.imshow(im_windowed_ch4, cmap="gray", vmin=0, vmax=255); ax_f0.axis("off"); _ax_style(ax_f0, "Original")

    ax_f1 = fig_full.add_subplot(gs_f[0, 1:])   # sinogram spans 2 columns
    ax_f1.imshow(sinogram, cmap="gray", aspect="auto",
                 extent=(theta[0], theta[-1], 0, sinogram.shape[0]))
    ax_f1.set_xlabel("Angle (°)", color="#768390", fontsize=8)
    ax_f1.set_ylabel("Detector position", color="#768390", fontsize=8)
    ax_f1.tick_params(colors="#768390", labelsize=7); _ax_style(ax_f1, "Sinogram")

    # ax_f2 = fig_full.add_subplot(gs_f[0, 0])   # unnesecary
    # fig_full.clear()

    gs_f = gridspec.GridSpec(2, 3, figure=fig_full, wspace=0.25, hspace=0.38,
                             height_ratios=[1.5, 1])

    ax_orig  = fig_full.add_subplot(gs_f[0, 0])
    ax_recon = fig_full.add_subplot(gs_f[0, 1])
    ax_sino  = fig_full.add_subplot(gs_f[0, 2])

    ax_orig.imshow(im_windowed_ch4, cmap="gray", vmin=0, vmax=255)
    ax_orig.axis("off"); _ax_style(ax_orig, "Original")

    ax_recon.imshow(recon_windowed, cmap="gray", vmin=0, vmax=255)
    ax_recon.axis("off"); _ax_style(ax_recon, f"Reconstructed ({method_label})  wc={wc}, ww={ww}")

    ax_sino.imshow(sinogram, cmap="gray", aspect="auto",
                   extent=(theta[0], theta[-1], 0, sinogram.shape[0]))
    ax_sino.set_xlabel("Angle (°)", color="#768390", fontsize=8)
    ax_sino.tick_params(colors="#768390", labelsize=7); _ax_style(ax_sino, "Sinogram")

    ax_f3 = fig_full.add_subplot(gs_f[1, 0])
    mid = sinogram.shape[1] // 2
    ax_f3.plot(sinogram[:, 0],   color="#6cb6ff", lw=1.2, label=f"{theta[0]:.0f}°")
    ax_f3.plot(sinogram[:, mid], color="#f78166", lw=1.2, label=f"{theta[mid]:.0f}°")
    ax_f3.plot(sinogram[:, -1],  color="#3ead4d", lw=1.2, label=f"{theta[-1]:.0f}°")
    ax_f3.legend(fontsize=7); ax_f3.grid(True, alpha=0.3)
    ax_f3.set_xlabel("Detector pos", color="#768390", fontsize=8)
    ax_f3.tick_params(colors="#768390", labelsize=7); _ax_style(ax_f3, "Sinogram Profiles")

    ax_f4 = fig_full.add_subplot(gs_f[1, 1])
    if method == "FBP":
        freq = np.linspace(0, 1, 256)
        filt_curve = {
            "ramp": freq,
            "shepp-logan": freq * np.sinc(freq),
            "cosine": freq * np.cos(np.pi * freq / 2),
            "hamming": freq * (0.54 + 0.46 * np.cos(np.pi * freq)),
            "hann": freq * 0.5 * (1 + np.cos(np.pi * freq)),
        }.get(fbp_filter, freq)
        ax_f4.plot(filt_curve, color="#6cb6ff", lw=1.5)
        ax_f4.grid(True, alpha=0.3); ax_f4.set_xlabel("Spatial freq", color="#768390", fontsize=8)
        ax_f4.tick_params(colors="#768390", labelsize=7); _ax_style(ax_f4, f"FBP Filter: {fbp_filter}")
    else:
        ax_f4.text(0.5, 0.5, "ART\n(algebraic)", ha="center", va="center",
                   color="#768390", fontsize=12, transform=ax_f4.transAxes); ax_f4.axis("off")
        _ax_style(ax_f4, "Method")

    ax_f5 = fig_full.add_subplot(gs_f[1, 2])
    s = min(im_windowed_ch4.shape)
    diag_orig  = np.diag(im_windowed_ch4[:s, :s])
    rs = min(recon_windowed.shape)
    diag_recon = np.diag(recon_windowed[:rs, :rs])
    mn_d, mx_d = diag_orig.min(), diag_orig.max()
    if mx_d > mn_d: diag_orig = (diag_orig - mn_d) / (mx_d - mn_d)
    rng = np.ptp(diag_recon)
    if rng > 0: diag_recon = (diag_recon - diag_recon.min()) / rng
    ax_f5.plot(diag_orig, color="#6cb6ff", lw=1.4, label="Original")
    ax_f5.plot(diag_recon, color="#f78166", lw=1.4, ls="--", label="Reconstructed")
    ax_f5.legend(fontsize=7); ax_f5.grid(True, alpha=0.3)
    ax_f5.tick_params(colors="#768390", labelsize=7)
    _ax_style(ax_f5, "Diagonal Intensity Profile")

    fig_full.tight_layout(pad=0.8)
    st.pyplot(fig_full, width='stretch')

    # Downloads
    st.markdown("---")
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    ddcol1, ddcol2, ddcol3 = st.columns(3)
    with ddcol1:
        st.download_button("⬇ Full report (PNG)", data=fig_to_bytes(fig_full),
                           file_name=f"{base}_{method.lower()}_report.png",
                           mime="image/png", key="ch4_dl_report")
    with ddcol2:
        data_rec, _ = arr_to_bytes(recon_windowed, "png")
        st.download_button("⬇ Reconstruction (PNG)", data=data_rec,
                           file_name=f"{base}_{method.lower()}_reconstruction.png",
                           mime="image/png", key="ch4_dl_rec")
    with ddcol3:
        mn_s, mx_s = sinogram.min(), sinogram.max()
        sino_norm = ((sinogram - mn_s) / max(mx_s - mn_s, 1e-9) * 255)
        data_sino, _ = arr_to_bytes(sino_norm, "png")
        st.download_button("⬇ Sinogram (PNG)", data=data_sino,
                           file_name=f"{base}_sinogram.png",
                           mime="image/png", key="ch4_dl_sino")
    plt.close(fig_full)


# ─────────────────────────────────────────────────────────────────────────────
# PURE PROCESSING HELPERS (no I/O, no Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def _simple_window(im, wc, ww, image_depth=255, tones=256):
    im1 = np.asarray(im, dtype=float)
    Vb = min((2.0 * wc + ww) / 2.0, image_depth)
    Va = max(Vb - ww, 0)
    out = (tones - 1) * (im1 - Va) / max(Vb - Va, 1e-9)
    return np.clip(out, 0, tones - 1)

def _broken_window(im, gray_val, im_val, image_depth=255, tones=256):
    im = np.asarray(im, dtype=float)
    im1 = np.where(
        im <= im_val,
        (gray_val / max(im_val, 1e-9)) * im,
        (((tones - 1) - (gray_val + 1)) / max(image_depth - (im_val + 1), 1e-9)) *
        (im - (im_val + 1)) + (gray_val + 1)
    )
    return np.round(im1)

def _double_window(im, ww1, wl1, ww2, wl2, image_depth=255, tones=256):
    im = np.asarray(im, dtype=float)
    im1 = np.zeros_like(im)
    half = (tones / 2) - 1
    ve1 = round((2.0 * wl1 + ww1) / 2.0); vs1 = max(ve1 - ww1, 0)
    ve2 = round((2.0 * wl2 + ww2) / 2.0); vs2 = max(ve2 - ww2, 0)      # changed from: vs2 = ve2 - ww2
    if vs2 < ve1:
        new_point = round((vs2 + ve1) / 2.0); ve1 = vs2 = new_point
    ve2 = min(ve2, image_depth)
    conditions = [
        im < vs1,
        (im >= vs1) & (im <= ve1),
        (im > ve1) & (im < vs2),
        (im >= vs2) & (im <= ve2),
    ]
    choices = [
        0,
        np.round((half / max(ve1 - vs1, 1e-9)) * (im - vs1)),
        half + 1,
        np.round(((tones - 1 - (half + 1)) / max(ve2 - vs2, 1e-9)) * (im - vs2) + (half + 1)),
    ]
    im1 = np.select(conditions, choices, default=tones - 1)
    return im1

def _windowing_function(choice, **kwargs):
    x = np.arange(0, IMAGE_DEPTH)
    y = np.zeros(IMAGE_DEPTH)
    y_initial = (TONES - 1) / (IMAGE_DEPTH - 1) * x
    if choice in (0, 1):  # Optimal / Simple
        wc = kwargs.get("wc", 128); ww = kwargs.get("ww", 200)
        Vb = min((2.0 * wc + ww) / 2.0, IMAGE_DEPTH); Va = max(Vb - ww, 0)
        for i, v in enumerate(x):
            if v < Va:    y[i] = 0
            elif v > Vb:  y[i] = TONES - 1
            else:         y[i] = (TONES - 1) * (v - Va) / max(Vb - Va, 1e-9)
    elif choice == 2:
        gray_val = kwargs["gray_val"]; im_val = kwargs["im_val"]
        for i, v in enumerate(x):
            if v <= im_val: y[i] = (gray_val / max(im_val, 1e-9)) * v
            else:           y[i] = (((TONES-1)-(gray_val+1)) / max(IMAGE_DEPTH-(im_val+1),1e-9)) * (v-(im_val+1)) + (gray_val+1)
    elif choice == 3:
        ww1=kwargs["ww1"]; wl1=kwargs["wl1"]; ww2=kwargs["ww2"]; wl2=kwargs["wl2"]
        half = (TONES/2)-1
        ve1=round((2.0*wl1+ww1)/2.0); vs1=max(ve1-ww1,0)
        ve2=round((2.0*wl2+ww2)/2.0); vs2=ve2-ww2
        if vs2 < ve1:
            np_pt=round((vs2+ve1)/2.0); ve1=vs2=np_pt
        ve2=min(ve2, IMAGE_DEPTH)
        for i, v in enumerate(x):
            if v < vs1:    y[i] = 0
            elif v <= ve1: y[i] = (half / max(ve1-vs1,1e-9)) * (v-vs1)
            elif v < vs2:  y[i] = half + 1
            elif v <= ve2: y[i] = ((TONES-1-(half+1)) / max(ve2-vs2,1e-9)) * (v-vs2) + (half+1)
            else:          y[i] = TONES - 1
    mn_y, mx_y = min(y), max(y)
    if mx_y > mn_y:
        y = (TONES-1) * (y - mn_y) / (mx_y - mn_y)
    return y_initial, y

def _nonlinear_process(im, choice):
    im = np.asarray(im, dtype=float)
    tones = TONES
    x = np.arange(0, IMAGE_DEPTH)
    y_initial = (tones-1) / (IMAGE_DEPTH-1) * x
    w = np.zeros(tones)
    labels = ["inverse","logarithmic","inverse logarithmic","power","sine-window","exp-window"]
    for i in range(tones):
        if choice == 0:   w[i] = tones - i - 1
        elif choice == 1: w[i] = math.log(1 + 0.05 * i)
        elif choice == 2: w[i] = np.exp(i) ** (1/128) - 1
        elif choice == 3: w[i] = i ** 0.55
        elif choice == 4: w[i] = np.sin(2 * np.pi * i / (4 * (tones-1)))
        elif choice == 5: w[i] = 1 - np.exp(-i / 90)
    w = (tones-1) * ((w - np.min(w)) / max(np.max(w) - np.min(w), 1e-9))
    mn, mx = np.min(im), np.max(im)
    im1 = np.round((tones-1) * (im - mn) / max(mx-mn, 1e-9))
    im_out = np.vectorize(lambda v: w[int(np.clip(v, 0, tones-1))])(im1)
    return im_out, y_initial, w

def _histogram(A):
    h = np.zeros(TONES, dtype=float)
    B = np.round(np.clip(A, 0, TONES-1)).astype(int)
    for val in B.ravel():
        h[val] += 1
    return h

def _hist_cumsum(im):
    hist, _ = np.histogram(im.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_norm = cdf * hist.max() / max(cdf.max(), 1e-9)
    return cdf_norm

def _hist_equalization(im):
    tones = TONES; image_depth = 256
    B = np.round((tones-1) * (np.clip(im, 0, image_depth) / image_depth))
    M, N = B.shape
    Bval = B.ravel()
    p = np.argsort(Bval)
    neq = int((M*N) / tones + 0.5)
    az = int((M*N) // neq)
    zRem = int((M*N) % neq)
    D = np.zeros(M*N)
    k = -1
    for i in range(0, neq*az, neq):
        k += 1
        for j in range(neq):
            if i+j < len(D): D[i+j] = k
    if zRem > 0:
        for i in range(neq*az, neq*az+zRem):
            if i < len(D): D[i] = tones - 1
    L = np.zeros(M*N)
    k = -1
    for i in range(M):
        for j in range(N):
            k += 1
            if k < len(p) and p[k] < len(D): L[p[k]] = D[k]
    Z = L.reshape(B.shape)
    return imNormalize(Z, tones)

def _cdf_equalization(im):
    tones = TONES
    B = np.round((tones-1) * (im / max(np.max(im), 1e-9)))
    M, N = im.shape
    h = _histogram(im)
    tone_values = (M*N) / tones
    CDFh = np.cumsum(h)
    result = CDFh[np.clip(B.astype(int), 0, tones-1).ravel()] / max(tone_values, 1e-9)
    return np.round(result).reshape(B.shape)

# ── 1D Filter builders ────────────────────────────────────────────────────────

def _make_ideal(N, fco, TYPE, enh, trans, w):
    N = int(N)
    fh = np.zeros(N, dtype=float)
    L = int(np.round(N/2 + (1 if N%2==0 else 0.5)))
    M = int(np.round(N/2 + (2 if N%2==0 else 1.5)))
    if TYPE == 1:
        fh = np.ones(N, dtype=float)
        fh[int(fco):L] = enh; sText = "Ideal LP"
    elif TYPE == 2:
        fh = np.zeros(N, dtype=float) + enh
        fh[int(fco):L] = 1; sText = "Ideal HP"
    elif TYPE == 3:
        fh = np.ones(N, dtype=float)
        for k in range(int(trans - w/2 + 0.5), int(trans + w/2 + 0.5)):
            if 0 <= k < N: fh[k] = enh
        sText = "Ideal BR"
    elif TYPE == 4:
        fh = np.zeros(N, dtype=float) + enh
        for k in range(int(trans - w/2 + 0.5), int(trans + w/2 + 0.5)):
            if 0 <= k < N: fh[k] = 1
        sText = "Ideal BP"
    else:
        return np.ones(N), "None"
    for k in range(M-1, N):
        mirror = N - k
        if 0 <= mirror < N:
            fh[k] = fh[mirror]
    mx = np.max(np.abs(fh))
    return fh / (mx if mx > 0 else 1), sText

def _make_butterworth(N, ndegree, fco, TYPE, trans):
    N = int(N); fh = np.zeros(N, dtype=float)
    L = int(np.round(N/2 + (1 if N%2==0 else 0.5)))
    M = int(np.round(N/2 + (2 if N%2==0 else 1.5)))
    if TYPE == 1:
        for k in range(L): fh[k] = 1.0 / (1.0 + 0.414 * (k/max(fco,1e-9))**(2*ndegree))
        sText = "Butterworth LP"
    elif TYPE == 2:
        for k in range(L): fh[k] = 1.0 / (1.0 + 0.414 * (max(fco,1)/(k+0.001))**(2*ndegree))
        fh_copy = fh.copy()
        for k in range(L):
            #fh[k] = fh_copy[k+int(trans)] if k < int(N/2-trans) else fh_copy[int(N/2)] 
            fh[k] = fh_copy[min(k + int(trans), len(fh_copy) - 1)] if k < int(N/2 - trans) else fh_copy[min(int(N/2), len(fh_copy) - 1)]
        sText = "Butterworth HP"
    elif TYPE == 3:
        for k in range(L): fh[k] = 1.0 / (1.0 + 0.414 * (fco/(k-trans+0.001))**(2*ndegree))
        sText = "Butterworth BR"
    elif TYPE == 4:
        fh = np.zeros(N, dtype=float) + 0.001
        for k in range(L): fh[k] = 1.0 / (1.0 + 0.414 * ((k-trans)/max(fco,1e-9))**(2*ndegree))
        sText = "Butterworth BP"
    else:
        return np.ones(N), "None"
    for k in range(M-1, N):
        mirror = N - k
        if 0 <= mirror < N:
            fh[k] = fh[mirror]
    mx = np.max(np.abs(fh))
    return fh / (mx if mx > 0 else 1), sText

def _make_exponential(N, ndegree, fco, TYPE, trans):
    N = int(N); fh = np.zeros(N, dtype=float)
    L = int(np.round(N/2 + (1 if N%2==0 else 0.5)))
    M = int(np.round(N/2 + (2 if N%2==0 else 1.5)))
    if TYPE == 1:
        for k in range(L): fh[k] = np.exp(-np.log(2) * (k/max(fco,1e-9))**ndegree)
        sText = "Exponential LP"
    elif TYPE == 2:
        for k in range(L): fh[k] = np.exp(-np.log(2) * (fco/(k+0.0001))**ndegree)
        fh_copy = fh.copy()
        for k in range(L):
            fh[k] = fh_copy[k+int(trans)] if k < int(N/2-trans) else fh_copy[int(N/2)]
        sText = "Exponential HP"
    elif TYPE == 3:
        for k in range(L): fh[k] = np.exp(-np.log(2) * (fco/(k-trans+0.00001))**ndegree)
        sText = "Exponential BR"
    elif TYPE == 4:
        fh = np.zeros(N, dtype=float) + 0.001
        for k in range(L): fh[k] = np.exp(-np.log(2) * ((k-trans)/max(fco,1e-9))**ndegree)
        sText = "Exponential BP"
    else:
        return np.ones(N), "None"
    for k in range(M-1, N):
        mirror = N - k
        if 0 <= mirror < N:
            fh[k] = fh[mirror]
    mx = np.max(np.abs(fh))
    return fh / (mx if mx > 0 else 1), sText

def _make_gaussian(N, ndegree, fco, TYPE, trans):
    N = int(N); fh = np.zeros(N, dtype=float)
    L = int(np.round(N/2 + (1 if N%2==0 else 0.5)))
    M = int(np.round(N/2 + (2 if N%2==0 else 1.5)))
    if TYPE == 1:
        for k in range(L): fh[k] = np.exp(-(k**2/(2*max(fco,1e-9)**2))**ndegree)
        sText = "Gaussian LP"
    elif TYPE == 2:
        for k in range(L): fh[k] = np.exp(-(2*fco**2/(k+0.0001)**2)**ndegree)
        fh_copy = fh.copy()
        for k in range(L):
            fh[k] = fh_copy[k+int(trans)] if k < int(N/2-trans) else fh_copy[int(N/2)]
        sText = "Gaussian HP"
    elif TYPE == 3:
        for k in range(L): fh[k] = np.exp(-(2*fco**2/(k-trans+0.00001)**2)**ndegree)
        sText = "Gaussian BR"
    elif TYPE == 4:
        fh = np.zeros(N, dtype=float) + 0.001
        for k in range(L): fh[k] = np.exp(-((k-trans)**2/(2*max(fco,1e-9)**2))**ndegree)
        sText = "Gaussian BP"
    else:
        return np.ones(N), "None"
    for k in range(M-1, N):
        mirror = N - k
        if 0 <= mirror < N:
            fh[k] = fh[mirror]
    mx = np.max(np.abs(fh))
    return fh / (mx if mx > 0 else 1), sText

def _design_2d_filter(im, fh):
    y, x = im.shape
    FH = np.zeros(im.shape, dtype=float)
    for k in range(y):
        for m in range(x):
            K = y/2 - k + 1; M_f = x/2 - m + 1
            ir = int(np.sqrt(K*K + M_f*M_f) + 0.5)
            ir = min(ir, len(fh) - 1)
            FH[k][m] = fh[ir]
    return np.fft.fftshift(FH)

def _filter_image(im, FH):
    return np.real(np.fft.ifft2(np.fft.fft2(im) * FH))

def _ampl_fft2(im):
    return np.round(10.0 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(im))) + 1))

def _gaussian_mtf(N):
    N = int(N); fh = np.zeros(N, dtype=float)
    L = int(np.round(N/2 + (1 if N%2==0 else 0.5)))
    M = int(np.round(N/2 + (2 if N%2==0 else 1.5)))
    sigma = L/2 - 1
    for k in range(L): 
        fh[k] = np.exp(-k**2 / (2 * sigma**2))
    #for k in range(M-1, N): 
    # fh[k] = fh[int(N-k)]
    for k in range(M-1, N):
         mirror = N - k
         if 0 <= mirror < N:
            fh[k] = fh[mirror]
    return fh

def _from1d_to_2d(im, fh):
    y, x = im.shape
    FH = np.zeros(im.shape, dtype=float)
    for k in range(y):
        for m in range(x):
            K = y/2 - k + 1; M_f = x/2 - m + 1
            ir = int(np.sqrt(K*K + M_f*M_f) + 0.5)
            ir = min(ir, len(fh) - 1)
            FH[k][m] = fh[ir]
    return FH / max(np.amax(FH), 1e-9)

def _blur_image(im, FH):
    return np.real(np.fft.ifft2(np.fft.fft2(im) * np.fft.fftshift(FH)))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

CHAPTERS = {
    "1 · Display Modification":       ch1_display_methods,
    "2 · Space Domain Filters":       ch2_spatial_filters,
    "3 · Frequency Domain Filters":   ch3_frequency_filters,
    "4 · Tomographic Reconstruction": ch4_reconstruction,
}

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
                color:#6cb6ff;padding:12px 0 6px 0;border-bottom:1px solid #1c2d40;
                margin-bottom:12px;letter-spacing:-0.2px;">
      🔬 Medical Imaging Processing Web-App
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-chapter">📖 Chapter</div>', unsafe_allow_html=True)
    chapter = st.radio(
        "Select chapter",
        list(CHAPTERS.keys()),
        label_visibility="collapsed",
        key="chapter_nav",
    )

    st.markdown("---")

    _prev_share = st.session_state.get("share_image", False)

    _ch_prefix = {
        "1 · Display Modification":       "ch1",
        "2 · Space Domain Filters":       "ch2",
        "3 · Frequency Domain Filters":   "ch3",
        "4 · Tomographic Reconstruction": "ch4",
    }.get(chapter, "ch1")

    st.markdown('<div class="sidebar-chapter">🖼 Shared Image</div>', unsafe_allow_html=True)
    share_image = st.checkbox(
        "Use same image across all chapters",
        value=_prev_share,
        key="share_image",
    )

    if share_image:
        # On the FIRST tick of enabling the checkbox, seed the shared loader
        # with the exact image the current chapter already had selected,
        # so the user's chosen image is preserved — not reset to default.
        if not _prev_share:
            _ch_source = st.session_state.get(f"{_ch_prefix}_source", "Folder ./images/")
            _ch_fn     = st.session_state.get(f"{_ch_prefix}_pick", None)
            _ch_wc     = st.session_state.get(f"{_ch_prefix}_wc", 128)
            _ch_ww     = st.session_state.get(f"{_ch_prefix}_ww", 255)

            st.session_state["shared_source"] = _ch_source
            st.session_state["shared_wc"]     = _ch_wc
            st.session_state["shared_ww"]     = _ch_ww

            if _ch_fn is not None and _ch_source == "Folder ./images/":
                _img_dir = "./images"
                if os.path.isdir(_img_dir):
                    _valid_ext = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
                    if DICOM_AVAILABLE:
                        _valid_ext += tuple(DICOM_EXTS)
                    _files = sorted([f for f in os.listdir(_img_dir)
                                     if f.lower().endswith(_valid_ext) and
                                     os.path.isfile(os.path.join(_img_dir, f))])
                    if _ch_fn in _files:
                        st.session_state["shared_pick"] = _files.index(_ch_fn)

        shared_im, shared_wc, shared_ww, shared_fn, shared_ext, shared_dcm, shared_meta = \
            image_loader_widget("shared")
        st.session_state["_shared_im"]    = shared_im
        st.session_state["_shared_wc"]    = shared_wc
        st.session_state["_shared_ww"]    = shared_ww
        st.session_state["_shared_fn"]    = shared_fn
        st.session_state["_shared_ext"]   = shared_ext
        st.session_state["_shared_dcm"]   = shared_dcm
        st.session_state["_shared_meta"]  = shared_meta
        if shared_im is not None:
            st.markdown(f'<span style="font-size:1rem; color:#8297ab;">📎 {shared_fn}</span>', unsafe_allow_html=True)
        else:
            st.info("Select an image above — it will be used in all chapters.")
    else:
        st.session_state["_shared_im"] = None

    st.markdown("---")
    if not DICOM_AVAILABLE:
        st.info("💡 Install pydicom for DICOM support:\n`pip install pydicom`")
    if not SKIMAGE_AVAILABLE:
        st.info("💡 Install scikit-image for reconstruction:\n`pip install scikit-image`")

# ── Patch image_loader_widget so chapters transparently use shared image ─────
_orig_loader = image_loader_widget

def image_loader_widget(key_prefix="main"):      #### noqa: F811  (intentional override)
    if st.session_state.get("share_image") and \
       st.session_state.get("_shared_im") is not None:
        return (
            st.session_state["_shared_im"],
            st.session_state["_shared_wc"],
            st.session_state["_shared_ww"],
            st.session_state["_shared_fn"],
            st.session_state["_shared_ext"],
            st.session_state["_shared_dcm"],
            st.session_state["_shared_meta"],
        )
    return _orig_loader(key_prefix)

CHAPTERS[chapter]()