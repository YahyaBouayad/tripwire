import streamlit as st
from torchvision import models
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load the model
def load_model(model_path):
    # Load a ResNet18 model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: cable present or absent
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model('cable_detector.pth')

# Required transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize
])

# Function to make predictions
def predict_image(image_path, model, class_names):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')  # Load in color mode
    transformed_image = data_transforms(image).unsqueeze(0)  # Add batch dimension
    
    # Make the prediction
    with torch.no_grad():
        outputs = model(transformed_image)
        _, predicted = torch.max(outputs, 1)
    
    # Return the predicted class
    return class_names[predicted.item()]

# Function to display the prediction
def show_prediction(image_path, model, class_names):
    prediction = predict_image(image_path, model, class_names)
    
    # Display the image
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Prediction: {prediction}")
    plt.show()

# Class names
class_names = ['Cable Absent', 'Cable Present']

# Streamlit app title
st.title("Cable Detection")

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make a prediction
    prediction = predict_image(uploaded_file, model, class_names)
    
    # Change the background color based on the result
    if prediction == "Cable Present":
        st.markdown(
            """
            <style>
            body {
                background-color: #FFCCCB;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.write("### 🚨 **Result: Cable Present!** 🚨")
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #D4EDDA;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.write("### ✅ **Result: No Cable Detected!** ✅")

    # Add an explanatory message
    st.markdown("You can upload another image to test again.")
