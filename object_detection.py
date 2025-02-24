import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# Page configuration
st.set_page_config(layout="wide")

# Load object detection model
@st.cache_resource
def load_model():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
    model.eval()
    return model, weights.transforms()

# Course database
COURSE_DATABASE = {
    "pen": [
        {"title": "Creative Writing Fundamentals", "level": "Beginner", "duration": "4 weeks"},
        {"title": "Technical Drawing Basics", "level": "Beginner", "duration": "6 weeks"},
    ],
    "pencil": [
        {"title": "Sketching for Beginners", "level": "Beginner", "duration": "4 weeks"},
        {"title": "Advanced Drawing Techniques", "level": "Advanced", "duration": "10 weeks"},
    ],
    "laptop": [
        {"title": "Introduction to Computing", "level": "Beginner", "duration": "8 weeks"},
        {"title": "Computer Maintenance", "level": "Intermediate", "duration": "4 weeks"},
    ],
    "cell phone": [
        {"title": "Mobile App Development", "level": "Intermediate", "duration": "12 weeks"},
        {"title": "Mobile Photography", "level": "Beginner", "duration": "4 weeks"},
    ],
    "book": [
        {"title": "Speed Reading Techniques", "level": "Beginner", "duration": "4 weeks"},
        {"title": "Literature Analysis", "level": "Intermediate", "duration": "8 weeks"},
    ],
    "keyboard": [
        {"title": "Touch Typing Mastery", "level": "Beginner", "duration": "4 weeks"},
        {"title": "Music Production Basics", "level": "Beginner", "duration": "8 weeks"},
    ]
}

def process_image(image, model, transforms):
    """Process an uploaded image for object detection"""
    img_tensor = transforms(image.convert("RGB"))
    with torch.no_grad():
        prediction = model([img_tensor])
    return prediction[0]

def draw_detections(image, prediction, score_threshold=0.7):
    """Draw bounding boxes and labels on the image"""
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    detected_objects = set()
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    class_names = weights.meta["categories"]
    
    image_np = np.array(image)
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            box = box.astype(int)
            class_name = class_names[label]
            cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(image_np, label_text, (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detected_objects.add(class_name.lower())
    return Image.fromarray(image_np), detected_objects

def get_course_recommendations(detected_objects):
    """Get course recommendations based on detected objects"""
    recommendations = []
    seen_courses = set()
    for obj in detected_objects:
        for key in COURSE_DATABASE:
            if key in obj and key not in seen_courses:
                recommendations.extend(COURSE_DATABASE[key])
                seen_courses.add(key)
    return recommendations

def main():
    st.title("Course Finder from Images")
    st.write("Upload an image to detect objects and get relevant course recommendations.")
    
    # Load model
    model, transforms = load_model()
    
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process image
        prediction = process_image(image, model, transforms)
        annotated_image, detected_objects = draw_detections(image, prediction)
        
        # Show annotated image
        st.image(annotated_image, caption="Detected Objects", use_container_width=True)
        
        # Get and display course recommendations
        recommendations = get_course_recommendations(detected_objects)
        if recommendations:
            st.markdown("### Recommended Courses")
            for course in recommendations:
                st.markdown(f"**{course['title']}**  ")
                st.markdown(f"Level: {course['level']}  ")
                st.markdown(f"Duration: {course['duration']}\n")
        else:
            st.write("No relevant courses found for detected objects.")

if __name__ == "__main__":
    main()
