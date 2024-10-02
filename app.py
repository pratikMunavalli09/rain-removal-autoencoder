import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load the trained model
model = torch.load('autoencoder_model.pth', map_location='cpu')
model.eval()

# Preprocess function to convert uploaded image to the model's input format
def preprocess(image):
    image = image.resize((256, 256))  # Resize the image to the appropriate size
    image = np.array(image).astype(np.float32) / 255.0  # Normalize the image
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor
    return image

# Deprocess function to convert model output back to a displayable image format
def deprocess(tensor):
    image = tensor.squeeze().permute(1, 2, 0).detach().numpy()  # Reverse the preprocessing steps
    image = (image * 255).astype(np.uint8)  # De-normalize and convert to uint8 format
    return Image.fromarray(image)

# Streamlit App Interface
st.title("Rain Removal from Images")

uploaded_file = st.file_uploader("Upload a rainy image", type=["jpg", "png"])

if uploaded_file is not None:
    # Preprocess and run inference
    image = Image.open(uploaded_file)
    rain_img = preprocess(image)
    
    with torch.no_grad():
        clean_img = model(rain_img).cpu()
    
    # Convert tensor to image and display
    clean_img = deprocess(clean_img)
    st.image([image, clean_img], caption=["Rainy Image", "Clean Image"], width=300)
