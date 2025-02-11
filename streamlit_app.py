import torch
import streamlit as st
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from classifier import NeuralNetwork  # Ensure this is properly structured
import torch.nn.functional as F  # For softmax


model = torch.load("mnist.pth", map_location=torch.device("cpu"))
model.eval()

# Streamlit UI
st.title("MNIST Digit Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Get model prediction
    with torch.no_grad():
        logits = model(image_tensor)  # Get raw output
        probabilities = F.softmax(logits, dim=1).numpy()  # Convert to probabilities
        predicted_label = np.argmax(probabilities)  # Get highest confidence label

    # Display results
    st.write(f"**Predicted Digit:** {predicted_label}")
    st.bar_chart(probabilities[0])  # Confidence as a bar chart
