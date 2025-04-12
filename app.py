import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

# Load class names
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("car_class.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🚗 Car Brand Classifier")
st.write("Upload a car image to identify its brand.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            prediction = class_names[predicted.item()]

        st.success(f"Predicted Car Brand: **{prediction}**")
