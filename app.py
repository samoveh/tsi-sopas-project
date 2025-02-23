import streamlit as st
import numpy as np
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
import supervision as sv  # Import the supervision library

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="F2Qjw6ol597K3W4eu89s"  # Replace with your actual API key
)

# Function to make predictions using Roboflow API
def predict(image):
    # Send image to Roboflow API
    result = CLIENT.infer(image, model_id="sopas/10")  # Replace with your model ID
    return result

# Streamlit app
st.title("SOPAS-PROJECT: YOLOv12 Object Detection with Roboflow and Streamlit")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","bmp"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Make prediction
    st.write("Predicting...")
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    predictions = predict(image_np)

    # Display predictions
    st.write("Predictions:")
    st.json(predictions)

    # Load the results into the supervision Detections API
    detections = sv.Detections.from_inference(predictions)

    # Create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)

    # Annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image_np, detections=detections)

    # Add labels to the annotated image
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    # Convert back to RGB for Streamlit
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the annotated image
    st.image(annotated_image, caption='Predicted Image with Labels.', use_container_width=True)
