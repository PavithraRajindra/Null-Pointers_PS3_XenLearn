import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import calendar
import time
from PIL import Image
import json
import random
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import google.generativeai as genai
from firebase_config import auth

# Initialize Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Create a users directory if it doesn't exist
if not os.path.exists('users'):
    os.makedirs('users')

# Load object detection model
@st.cache_resource
def load_model():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
    model.eval()
    return model, weights.transforms()

# Course database for camera detection
CAMERA_COURSE_DATABASE = {
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

def save_user_data(email, data):
    """Save user data to a JSON file"""
    filename = f"users/{email.replace('@', '_at_')}.json"
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_user_data(email):
    """Load user data from JSON file"""
    filename = f"users/{email.replace('@', '_at_')}.json"
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'
    if 'user_interests' not in st.session_state:
        st.session_state.user_interests = []
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'navigation_history' not in st.session_state:
        st.session_state.navigation_history = []
    if 'detected_objects' not in st.session_state:
        st.session_state.detected_objects = set()
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_token' not in st.session_state:
        st.session_state.user_token = None

def navigate_to(page):
    if st.session_state.current_page != page:
        st.session_state.navigation_history.append(st.session_state.current_page)
        st.session_state.current_page = page
        st.rerun()

def handle_login(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.user_email = email
        st.session_state.user_token = user['idToken']
        st.session_state.logged_in = True

        # Load user data if exists
        user_data = load_user_data(email)
        if user_data:
            st.session_state.user_interests = user_data.get('interests', [])
            st.session_state.user_profile = user_data
            st.session_state.current_page = 'dashboard'
        else:
            st.session_state.current_page = 'interests'
        return True
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False

def handle_signup(email, password, full_name):
    try:
        # Create user in Firebase
        user = auth.create_user_with_email_and_password(email, password)

        # Initialize user data
        user_data = {
            'email': email,
            'full_name': full_name,
            'interests': [],
            'joined_date': datetime.now().strftime("%Y-%m-%d"),
            'courses': []
        }

        # Save user data
        save_user_data(email, user_data)

        # Set session state
        st.session_state.user_email = email
        st.session_state.user_token = user['idToken']
        st.session_state.user_profile = user_data
        st.session_state.logged_in = True
        st.session_state.current_page = 'interests'
        return True
    except Exception as e:
        st.error(f"Sign up failed: {str(e)}")
        return False
def logout():
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.session_state.user_token = None
    st.session_state.user_profile = None
    st.session_state.current_page = 'login'
    st.rerun()

def go_back():
    if st.session_state.navigation_history:
        st.session_state.current_page = st.session_state.navigation_history.pop()
        st.rerun()

def navigation_buttons():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back") and st.session_state.navigation_history:
            go_back()
    with col2:
        if st.button("Forward →") and st.session_state.current_page != 'dashboard':
            navigate_to('dashboard')

def login_page():
    st.markdown("""
        <style>
        .login-container { padding: 2rem; }
        .stAlert { color: black !important; }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("# XenLearn")
        st.markdown("### Transforming physical resources into digital learning experiences")
    
    with col2:
        st.markdown("### Welcome!")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if email and password:
                    if handle_login(email, password):
                        st.success("Login successful!")
                        st.rerun()
                else:
                    st.warning("Please enter both email and password.")
                
        with tab2:
            full_name = st.text_input("Full Name", key="signup_name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            
            if st.button("Sign Up"):
                if not all([full_name, email, password, confirm_password]):
                    st.warning("Please fill in all fields.")
                elif password != confirm_password:
                    st.warning("Passwords do not match.")
                else:
                    if handle_signup(email, password, full_name):
                        st.success("Sign up successful! Please proceed to set your interests.")
                        st.rerun()


def interests_page():
    st.title("Select Your Interests")
    
    subjects = [
        "Mathematics", "Physics", "Chemistry", "Biology",
        "Computer Science", "Environmental Science", "Art",
        "Music", "Literature", "History", "Geography",
        "Economics", "Psychology", "Engineering"
    ]
    
    st.markdown("### Choose subjects that interest you:")
    
    cols = st.columns(3)
    selected_interests = []
    
    for i, subject in enumerate(subjects):
        with cols[i % 3]:
            if st.checkbox(subject):
                selected_interests.append(subject)
    
    if st.button("Continue to Dashboard"):
        st.session_state.user_interests = selected_interests
        st.session_state.current_page = 'dashboard'
        st.rerun()

def profile_page():
    st.title("Profile Settings")

    if not st.session_state.user_profile:
        st.warning("Please log in to view profile settings.")
        return

    with st.form("profile_form"):
        new_full_name = st.text_input("Full Name", value=st.session_state.user_profile['full_name'])
        new_email = st.text_input("Email", value=st.session_state.user_profile['email'])
        new_password = st.text_input("New Password", type="password", placeholder="Leave blank to keep current")
        
        if st.form_submit_button("Save Changes"):
            
            try:
                # Update Firebase auth if email/password changed
                if new_password:
                    auth.update_user_password(st.session_state.user_token, new_password)
                if new_email != st.session_state.user_profile['email']:
                    auth.update_user_email(st.session_state.user_token, new_email)

                # Update local profile
                st.session_state.user_profile.update({
                    'email': new_email,
                    'full_name': new_full_name
                })
                save_user_data(new_email, st.session_state.user_profile)
                st.success("Profile updated successfully!")
            except Exception as e:
                st.error(f"Failed to update profile: {str(e)}")

def collaboration_page():
    st.title("Collaboration Hub")
    
    # Simulated online users
    online_users = [
        {"name": "Alice Smith", "expertise": "Physics"},
        {"name": "Bob Johnson", "expertise": "Mathematics"},
        {"name": "Carol Davis", "expertise": "Chemistry"}
    ]
    
    # Display online users
    st.subheader("Online Users")
    selected_user = None
    
    for user in online_users:
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            st.write(user["name"])
        with col2:
            st.write(user["expertise"])
        with col3:
            if st.button("Connect", key=user["name"]):
                selected_user = user
    
    # Chat interface
    if selected_user:
        st.subheader(f"Chat with {selected_user['name']}")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_messages:
                st.write(f"{message['sender']}: {message['text']}")
        
        # Message input
        message = st.text_input("Type your message")
        if st.button("Send"):
            if message:
                st.session_state.chat_messages.append({
                    "sender": "You",
                    "text": message
                })
                # Simulate response
                st.session_state.chat_messages.append({
                    "sender": selected_user["name"],
                    "text": f"Thanks for your message about {message}"
                })
                st.rerun()

def get_recommended_courses(interests):
    # Dictionary mapping interests to courses
    course_recommendations = {
        "Mathematics": ["Advanced Calculus", "Linear Algebra", "Statistics"],
        "Physics": ["Quantum Mechanics", "Classical Mechanics", "Thermodynamics"],
        "Computer Science": ["Python Programming", "Data Structures", "Machine Learning"],
        "Chemistry": ["Organic Chemistry", "Inorganic Chemistry", "Biochemistry"],
        "Biology": ["Molecular Biology", "Genetics", "Ecology"],
        "Environmental Science": ["Climate Science", "Environmental Policy", "Sustainability"],
        "Art": ["Digital Art", "Art History", "Painting Techniques"],
        "Music": ["Music Theory", "Digital Music Production", "Sound Design"]
    }
    
    recommended = []
    for interest in interests:
        if interest in course_recommendations:
            recommended.extend(course_recommendations[interest])
    
    # Randomly select 3 courses if we have more
    if len(recommended) > 3:
        recommended = random.sample(recommended, 3)
    return recommended

# Object detection related functions from first code
def process_frame(frame, model, transforms):
    """Process a single frame for object detection"""
    frame = cv2.resize(frame, (640, 480))
    img_tensor = transforms(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    
    with torch.no_grad():
        prediction = model([img_tensor])
    
    return prediction[0]

def draw_detections(frame, prediction, score_threshold=0.7):
    """Draw bounding boxes and labels on the frame"""
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    detected_objects = set()
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    class_names = weights.meta["categories"]
    
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            box = box.astype(int)
            class_name = class_names[label]
            
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label_text, (box[0], box[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detected_objects.add(class_name.lower())
    
    return frame, detected_objects

def get_camera_course_recommendations(detected_objects):
    """Get course recommendations based on detected objects"""
    recommendations = []
    seen_courses = set()
    
    for obj in detected_objects:
        for key in CAMERA_COURSE_DATABASE:
            if key in obj and key not in seen_courses:
                recommendations.extend(CAMERA_COURSE_DATABASE[key])
                seen_courses.add(key)
    return recommendations

def get_project_suggestions(detected_objects):
    """Get creative project suggestions based on detected objects using Gemini API"""
    if not detected_objects:
        return []
    
    try:
        objects_list = ", ".join(detected_objects)
        prompt = f"""Given these objects: {objects_list}
        Generate exactly 3 creative, fun, and educational experiments or projects that can be done using these items. These experiments should be of professional level that one can issue certifications for.
        
        Return ONLY a valid JSON array with exactly this structure and no additional text:
        [
            {{
                "title": "Project Title",
                "description": "Project description",
                "difficulty": "easy/medium/hard",
                "time": "30 minutes"
            }},
            {{
                "title": "Second Project",
                "description": "Second description",
                "difficulty": "easy/medium/hard",
                "time": "1 hour"
            }},
            {{
                "title": "Third Project",
                "description": "Third description",
                "difficulty": "easy/medium/hard",
                "time": "45 minutes"
            }}
        ]"""

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Add safety check for response
        if not response.text:
            return []
            
        try:
            # Clean the response text and parse JSON
            cleaned_text = response.text.strip()
            suggestions = json.loads(cleaned_text)
            
            # Validate the structure
            if not isinstance(suggestions, list):
                return []
                
            return suggestions
        except json.JSONDecodeError as je:
            st.error(f"Invalid JSON format in response: {je}")
            return []
            
    except Exception as e:
        st.error(f"Error generating project suggestions: {str(e)}")
        return []

def object_detection():
    st.title("Course Finder - Photo Analysis")
    navigation_buttons()
    
    # Add the styling
    st.markdown("""
        <style>
        .stApp { background-color: #f5f5f5; }
        .nav-button { margin: 5px 0; }
        h4,h5,h6 {color: black !important}
        h1, h2, h3 {color: black !important}
        [data-testid="stMetricValue"] div { color: black !important; }
        [data-testid="stTextInput"] label { color: black !important; }
        [data-testid="stMetricLabel"] {
            color: black !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load model
    model, transforms = load_model()
    
    # Create layout
    left_col, main_col, right_col = st.columns([1,2,1])
    
    # Left Navigation
    with left_col:
        st.markdown("### Navigation")
        if st.button("👤 Profile"):
            navigate_to('profile')
        st.button("📚 Courses")
        if st.button("🏠 Back to Dashboard"):
            navigate_to('dashboard')
        if st.button("👥 Collaborate"):
            navigate_to('collaborate')
        st.button("🎓 Certifications")
        st.button("⚙️ Settings")
    
    # Main content
    with main_col:
        # Initialize session state for captured image
        if 'captured_image' not in st.session_state:
            st.session_state.captured_image = None
        if 'analyzed_results' not in st.session_state:
            st.session_state.analyzed_results = None
        
        # Camera input and capture button
        camera_col, button_col = st.columns([3,1])
        with camera_col:
            img_file_buffer = st.camera_input("Take a picture")
        
        # Process the captured image
        if img_file_buffer is not None:
            # Convert the file buffer to opencv image
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Store the captured image
            st.session_state.captured_image = cv2_img
            
            # Process image and draw detections
            prediction = process_frame(cv2_img, model, transforms)
            annotated_frame, detected_objects = draw_detections(cv2_img, prediction)
            
            # Store detected objects
            st.session_state.detected_objects = detected_objects
            
            # Show annotated image
            st.image(annotated_frame, channels="BGR", caption="Analyzed Image")
            
            # Show detected objects
            if detected_objects:
                st.write("Detected Objects:", ", ".join(detected_objects))
    
    # Right sidebar (recommendations)
    with right_col:
        st.markdown("### Recommended Courses")
        if hasattr(st.session_state, 'detected_objects') and st.session_state.detected_objects:
            recommendations = get_camera_course_recommendations(st.session_state.detected_objects)
            
            if recommendations:
                for course in recommendations:
                    st.markdown(f"""
                        <div style='background-color: white; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                            <h4 style='color: black;'>{course['title']}</h4>
                            <p style='color: black;'>Level: {course['level']}</p>
                            <p style='color: black;'>Duration: {course['duration']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No course recommendations found for detected objects.")
        
        # Add project suggestions section
        st.markdown("### Creative Projects")
        if hasattr(st.session_state, 'detected_objects') and st.session_state.detected_objects:
            projects = get_project_suggestions(st.session_state.detected_objects)
            
            if projects:
                for project in projects:
                    st.markdown(f"""
                        <div style='background-color: white; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                            <h4 style='color: black;'>{project['title']}</h4>
                            <p style='color: black;'>{project['description']}</p>
                            <p style='color: black;'>Difficulty: {project['difficulty']}</p>
                            <p style='color: black;'>Time: {project['time']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No project suggestions available.")
        else:
            st.info("Take a picture to get course recommendations!")

def dashboard():
    st.markdown("""
        <style>
        .stApp { background-color: #f5f5f5; }
        .nav-button { margin: 5px 0; }
        
        h4,h5,h6 {color: black !important}
        h1, h2, h3 {color: black !important}
        [data-testid="stMetricValue"] div { color: black !important; }
        [data-testid="stTextInput"] label { color: black !important; }
        [data-testid="stMetricLabel"] {
            color: black !important;
            font-weight: bold;
        label { 
            color: black !important;
            font-weight: bold;
        }
        }
        
        </style>
    """, unsafe_allow_html=True)
    
    # Top bar
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.text_input("🔍 Search for courses, materials, or topics...", label_visibility='visible')
    with col2:
        st.metric("Points", "1250", "+50 today")
    with col3:
        st.metric("Streak", "7 days", "Personal Best!")
    
    # Main layout
    left_col, main_col, right_col = st.columns([1,2,1])
    
    # Left Navigation
    with left_col:
        st.markdown("### Navigation")
        if st.button("👤 Profile"):
            st.session_state.current_page = 'profile'
            st.rerun()
        st.button("📚 Courses")
        if st.button("📷 Search with Camera"):
            st.session_state.current_page = 'camera'
            st.rerun()
        if st.button("👥 Collaborate"):
            st.session_state.current_page = 'collaborate'
            st.rerun()
        st.button("🎓 Certifications")
        st.button("⚙️ Settings")
    
    # Main Content
    with main_col:
        # User's current badge
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: black;'>🏆 Your Current Badge</h3>
                <p style='color: black;'>🥇 Gold Badge - Master Learner</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Recommended Courses based on interests
        st.markdown("### 📚 Recommended Courses")
        recommended_courses = get_recommended_courses(st.session_state.user_interests)
        for course in recommended_courses:
            st.markdown(f"""
                <div style='background-color: white; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <h4 style='color: black;'>{course}</h4>
                    <p style='color: black;'>Level: Beginner | Duration: 4 weeks | Rating: ⭐⭐⭐⭐⭐</p>
                    <div style='background-color: #f0f0f0; border-radius: 5px; padding: 5px;'>
                        <small style='color: black;'>Progress: 65%</small>
                        <div style='background-color: #2196F3; width: 65%; height: 5px; border-radius: 5px;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Calendar
        st.markdown("### 📅 Calendar")
        current_date = datetime.now()
        month = st.selectbox("Select Month", list(calendar.month_name)[1:])
        
        # Display calendar grid
        month_num = list(calendar.month_name).index(month)
        cal = calendar.monthcalendar(current_date.year, month_num)
        
        # Create calendar grid
        cols = st.columns(7)
        for i, day in enumerate("Mon Tue Wed Thu Fri Sat Sun".split()):
            cols[i].write(day)
        
        for week in cal:
            cols = st.columns(7)
            for i, day in enumerate(week):
                if day != 0:
                    cols[i].write(day)
                else:
                    cols[i].write("")
    
    # Right Sidebar - Current Courses
    with right_col:
        st.markdown("### Current Courses")
        current_courses = ["Environmental Studies", "Digital Learning"]
        for course in current_courses:
            st.markdown(f"""
                <div style='background-color: white; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <h4 style='color: black;'>{course}</h4>
                    <p style='color: black;'>Next lesson: Tomorrow</p>
                    <div style='background-color: #f0f0f0; border-radius: 5px; padding: 5px;'>
                        <small style='color: black;'>Progress: 65%</small>
                        <div style='background-color: #2196F3; width: 65%; height: 5px; border-radius: 5px;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
def main():
    initialize_session_state()
    st.set_page_config(page_title="XenLearn", layout="wide")
    
    if not st.session_state.logged_in:
        login_page()
    elif st.session_state.current_page == 'interests':
        interests_page()
    elif st.session_state.current_page == 'dashboard':
        dashboard()
    elif st.session_state.current_page == 'camera':
        object_detection()
    elif st.session_state.current_page == 'profile':
        profile_page()
    elif st.session_state.current_page == 'collaborate':
        collaboration_page()

if __name__ == "__main__":
    main()
