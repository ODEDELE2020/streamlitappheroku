import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np
import cv2
import easyocr

# Load the pretrained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 56)
model.load_state_dict(torch.load('/content/my_model.pt', map_location=torch.device('cpu')))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels (replace with your actual labels for 50 states and 6 districts)
class_labels = [
    'ALABAMA', 'ALASKA', 'AMERICAN SAMOA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'CNMI', 'COLORADO', 'CONNECTICUT',
    'DELAWARE', 'FLORIDA', 'GEORGIA', 'GUAM', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY',
    'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA',
    'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA',
    'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'PUERTO RICO', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA',
    'TENNESSEE', 'TEXAS', 'U S VIRGIN ISLANDS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON', 'WASHINGTON DC',
    'WEST VIRGINIA', 'WISCONSIN', 'WYOMING'
]

# Function to make predictions
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

# Function to recognize license number
def recognize_license_number(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to read the image file")

        resized_image = cv2.resize(image, (205, 58))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        reader = easyocr.Reader(['en'])
        result = reader.readtext(threshold_image)
        license_number = ''.join([detection[1] for detection in result if detection[2] > 0.5])
        return license_number
    except Exception as e:
        st.write(f"Error: {str(e)}")
        return ""

# Streamlit UI
st.title("License Plate Classification")
st.write("Upload an image of a license plate to classify the state or district.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Classify'):
        try:
            image_path = "/content/" + uploaded_file.name  # Use the uploaded image path
            image.save(image_path)  # Savethe uploaded image
            prediction = predict(image)  # Call the predict function
            predicted_label = class_labels[prediction]  # Now predicted is defined
            st.write(f" The Predicted State/District: {predicted_label}")

            # Recognize license number
            license_number = recognize_license_number(image_path)
            st.write(f"License Number: {license_number}")
        except Exception as e:
            st.write(f"Error: {str(e)}")
