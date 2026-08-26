"""
Streamlit interface for Bollywood Poster Style Transfer.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
import torch
from PIL import Image

from style_transfer import load_image, run_style_transfer, IMG_SIZE
from torchvision.utils import save_image


st.set_page_config(page_title="Vintage Poster Maker", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #2b1114 0%, #4a1c1c 40%, #6b2a1f 75%, #8c4a1e 100%);
    }
    h1 {
        font-family: 'Georgia', serif;
        letter-spacing: 1px;
        color: #f4e4c1;
    }
    h2, h3 {
        color: #f4e4c1;
    }
    .stButton > button {
        background-color: #c9832f;
        color: #2b1114;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #e0a04a;
        color: #2b1114;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Vintage Poster Maker")
st.write(
    "Turn a photo into retro Bollywood poster art using neural style transfer "
    "(VGG19, Gatys et al. method)."
)

STYLES_DIR = "styles"


@st.cache_data(show_spinner=False)
def list_styles():
    if not os.path.isdir(STYLES_DIR):
        return []
    return sorted(
        f for f in os.listdir(STYLES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload your photo")
    content_file = st.file_uploader("Content image", type=["jpg", "jpeg", "png"])
    if content_file:
        st.image(content_file, caption="Your photo", use_container_width=True)

with col2:
    st.subheader("2. Pick a poster style")
    available_styles = list_styles()
    style_choice = None
    style_upload = st.file_uploader("Or upload your own style image", type=["jpg", "jpeg", "png"])

    if style_upload:
        st.image(style_upload, caption="Style reference", use_container_width=True)
    elif available_styles:
        style_choice = st.selectbox("Choose a preset poster", available_styles)
        st.image(
            os.path.join(STYLES_DIR, style_choice),
            caption=style_choice,
            use_container_width=True,
        )
    else:
        st.warning("No preset posters found in styles/. Upload one instead.")

st.subheader("3. Settings")
with st.expander("Advanced settings (optional)"):
    steps = st.slider("Optimization steps", 200, 2000, 1000, step=100)
    style_weight = st.select_slider(
        "Style strength",
        options=[1e4, 3e4, 1e5, 3e5, 1e6],
        value=3e5,
        format_func=lambda x: f"{x:.0e}",
    )
    lr = st.select_slider("Learning rate", options=[0.005, 0.01, 0.02, 0.05], value=0.02)

st.caption(
    "Tip: clean portrait photos with hands away from the face give the best results — "
    "the model struggles with fine detail like fingers."
)

run = st.button("✨ Generate poster", type="primary", use_container_width=True)

if run:
    if not content_file:
        st.error("Please upload a content photo first.")
    elif not style_choice and not style_upload:
        st.error("Please choose or upload a style image first.")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            content_path = os.path.join(tmp, "content.jpg")
            Image.open(content_file).convert("RGB").save(content_path)

            if style_upload:
                style_path = os.path.join(tmp, "style.jpg")
                Image.open(style_upload).convert("RGB").save(style_path)
            else:
                style_path = os.path.join(STYLES_DIR, style_choice)

            with st.spinner(f"Optimizing on {'GPU' if torch.cuda.is_available() else 'CPU'}... this may take a minute."):
                content_img = load_image(content_path)
                style_img = load_image(style_path)

                output = run_style_transfer(
                    content_img, style_img,
                    num_steps=steps,
                    style_weight=style_weight,
                    lr=lr,
                )

                output_path = os.path.join(tmp, "result.jpg")
                save_image(output, output_path)

                result_image = Image.open(output_path).convert("RGB")

            st.success("Done!")
            st.image(result_image, caption="Your Bollywood poster", use_container_width=True)

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download image",
                    data=f,
                    file_name="bollywood_poster.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
