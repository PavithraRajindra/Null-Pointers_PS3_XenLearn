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
    if 'navigation_future' not in st.session_state:  # Add this for forward navigation
        st.session_state.navigation_future = []
    if 'detected_objects' not in st.session_state:
        st.session_state.detected_objects = set()
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_token' not in st.session_state:
        st.session_state.user_token = None


def navigate_to(page):
    """Navigate to a new page and update history"""
    if st.session_state.current_page != page:
        # Add current page to history before changing
        st.session_state.navigation_history.append(st.session_state.current_page)
        # Clear forward history when navigating to a new page
        st.session_state.navigation_future = []
        
        # Special handling for DoubtAI page
        if page == 'doubtai':
            # If we're navigating to DoubtAI page, save current conversation to history if not empty
            if 'current_conversation' in st.session_state and st.session_state.current_conversation:
                if 'career_chat_history' not in st.session_state:
                    st.session_state.career_chat_history = []
                
                # Save current conversation with timestamp
                conversation_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.session_state.career_chat_history.append({
                    "timestamp": conversation_time,
                    "messages": st.session_state.current_conversation.copy()
                })
                
            # Start a fresh conversation
            st.session_state.current_conversation = []
        
        st.session_state.current_page = page
        st.rerun()


def go_back():
    if st.session_state.navigation_history:
        # Get the previous page
        previous_page = st.session_state.navigation_history.pop()
        
        # Store current page in forward history
        st.session_state.navigation_future.append(st.session_state.current_page)
        
        # If we're currently on the DoubtAI page, save the current conversation
        if st.session_state.current_page == 'doubtai' and 'current_conversation' in st.session_state and st.session_state.current_conversation:
            if 'career_chat_history' not in st.session_state:
                st.session_state.career_chat_history = []
            
            # Save current conversation with timestamp
            conversation_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.career_chat_history.append({
                "timestamp": conversation_time,
                "messages": st.session_state.current_conversation.copy()
            })
            
            # Start a fresh conversation
            st.session_state.current_conversation = []
        
        # Go to previous page
        st.session_state.current_page = previous_page
        st.rerun()

def go_forward():
    if st.session_state.navigation_future:
        # Get the next page
        next_page = st.session_state.navigation_future.pop()
        
        # Store current page in back history
        st.session_state.navigation_history.append(st.session_state.current_page)
        
        # If we're currently on the DoubtAI page, save the current conversation
        if st.session_state.current_page == 'doubtai' and 'current_conversation' in st.session_state and st.session_state.current_conversation:
            if 'career_chat_history' not in st.session_state:
                st.session_state.career_chat_history = []
            
            # Save current conversation with timestamp
            conversation_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.career_chat_history.append({
                "timestamp": conversation_time,
                "messages": st.session_state.current_conversation.copy()
            })
            
            # Clear for when we might return to this page
            st.session_state.current_conversation = []
        
        # If we're navigating to DoubtAI, prepare a fresh conversation
        if next_page == 'doubtai':
            st.session_state.current_conversation = []
        
        # Go to next page
        st.session_state.current_page = next_page
        st.rerun()

def navigation_buttons():
    """Create navigation buttons in a container at the top of the page"""
    with st.container():
        col1, col2, col3, col4 = st.columns([1, 2, 3, 1])
        with col1:
            if st.button("← Back", key="back_btn", disabled=len(st.session_state.navigation_history) == 0):
                go_back()
        with col3:
            if st.button("Forward →", key="forward_btn", disabled=len(st.session_state.navigation_future) == 0):
                go_forward()
        with col4:
            if st.button("Logout", key="logout_btn", type="primary"):
                logout()

def logout():
    # Clear all session state
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.session_state.user_token = None
    st.session_state.user_profile = None
    st.session_state.user_interests = []
    st.session_state.navigation_history = []
    st.session_state.navigation_future = []
    st.session_state.current_page = 'login'
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

def interests_page():
    navigation_buttons()  # Add this as first line
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
    navigation_buttons()  # Add this as first line
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

def doubtai_page():
    navigation_buttons()  # Add navigation buttons at the top
    st.title("DoubtAI - All Your Doubts Cleared!")
    
    # Initialize chat history in session state if it doesn't exist
    if 'career_chat_history' not in st.session_state:
        st.session_state.career_chat_history = []
    
    # Initialize current conversation in session state if it doesn't exist
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = []
    
    # Add a button to start a new conversation
    if st.button("New Conversation"):
        # Save current conversation to history before clearing
        if st.session_state.current_conversation:
            conversation_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.career_chat_history.append({
                "timestamp": conversation_time,
                "messages": st.session_state.current_conversation.copy()
            })
            # Clear current conversation
            st.session_state.current_conversation = []
            st.rerun()
    
    # Add dropdown to view past conversations if there are any
    if st.session_state.career_chat_history:
        with st.expander("View Past Conversations"):
            for i, conv in enumerate(st.session_state.career_chat_history):
                st.markdown(f"**Conversation {i+1}** - {conv['timestamp']}")
                for msg in conv['messages']:
                    sender = msg.split(": ")[0]
                    content = ": ".join(msg.split(": ")[1:])
                    
                    if sender == "You":
                        st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin: 5px 0; color: black'><b>{sender}:</b> {content}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background-color: black; padding: 10px; border-radius: 10px; margin: 5px 0; color: white'><b>{sender}:</b> {content}</div>", unsafe_allow_html=True)
                st.markdown("---")
    
    # Display only the current conversation
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.current_conversation:
            sender = message.split(": ")[0]
            content = ": ".join(message.split(": ")[1:])
            
            if sender == "You":
                st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin: 5px 0; color: black'><b>{sender}:</b> {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: black; padding: 10px; border-radius: 10px; margin: 5px 0; color: white'><b>{sender}:</b> {content}</div>", unsafe_allow_html=True)
    
    # Add the system prompt to provide context for the assistant
    system_prompt = "You are DoubtAI, a helpful educational assistant designed to help students understand concepts and solve problems across various academic subjects including Mathematics, Physics, Chemistry, Biology, Computer Science, Literature, History, and more."
    
    # Input for new messages
    with st.form(key="chat_form"):
        user_input = st.text_area("Type your question here:", height=100)
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message to current conversation
            st.session_state.current_conversation.append(f"You: {user_input}")
            
            try:
                # Generate response
                prompt = f"{system_prompt}\n\n{user_input}"
                response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
                bot_response = response.text
                
                # Add bot response to current conversation
                st.session_state.current_conversation.append(f"DoubtAI: {bot_response}")
                
                # Rerun to display the updated chat
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")


def collaboration_page():
    navigation_buttons()  # Add this as first line
    st.title("Collaboration Hub")
    
    # Google Meet style page
    st.markdown("""
        <style>
        .meet-container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .meet-input {
            width: 100%;
            padding: 12px;
            margin: 20px 0;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 16px;
        }
        .meet-link {
            background-color: #1a73e8;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            margin-top: 10px;
            cursor: pointer;
        }
        .meet-description {
            margin: 20px 0;
            color: #5f6368;
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Generate a unique meeting ID if not already generated
    if 'meeting_id' not in st.session_state:
        # Generate a random meeting ID similar to Google Meet
        characters = "abcdefghijkmnopqrstuvwxyz0123456789"
        meeting_id = "".join(random.choice(characters) for _ in range(3))
        meeting_id += "-"
        meeting_id += "".join(random.choice(characters) for _ in range(4))
        meeting_id += "-"
        meeting_id += "".join(random.choice(characters) for _ in range(3))
        st.session_state.meeting_id = meeting_id
    
    # Display the Google Meet style interface
    st.markdown(f"""
        <div class="meet-container">
            <h2 style=color:black>Start a group study session</h2>
            <p class="meet-description">Share this link with your friends and learn together. XenLearn makes collaborative learning easy.</p>
            <input type="text" class="meet-input" value="meet.xenlearn.com/{st.session_state.meeting_id}" readonly />
            <p class="meet-description">Clicking the button below will copy the link and open a new session.</p>
            <div class="meet-description">
                <p>Your meeting is ready</p>
                <p>Share this link with others you want in the meeting</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Join meeting", key="join_meeting"):
        st.success(f"Meeting link copied! Share it with your friends: meet.xenlearn.com/{st.session_state.meeting_id}")
        st.balloons()
    
    # Reset meeting button
    if st.button("Create new meeting", key="new_meeting"):
        # Generate a new random meeting ID
        characters = "abcdefghijkmnopqrstuvwxyz0123456789"
        meeting_id = "".join(random.choice(characters) for _ in range(3))
        meeting_id += "-"
        meeting_id += "".join(random.choice(characters) for _ in range(4))
        meeting_id += "-"
        meeting_id += "".join(random.choice(characters) for _ in range(3))
        st.session_state.meeting_id = meeting_id
        st.rerun()

def certificates_page():
    navigation_buttons()
    st.title("Certificates")
    st.write("Certificates obtained:")
    
    # Sample certificates (can be replaced with actual data or images)
    certificates = [
        "Certificate of Completion - Course AI & ML", 
        "Certificate of Excellence - Course Web Development"
    ]
    
    for cert in certificates:
        st.write(cert)
    
    # # Optionally, you can display images of the certificates or provide links to PDF files
    # st.image("https://example.com/sample_certificate.png", caption="Sample Certificate", use_column_width=True)

# Load object detection model
@st.cache_resource
def load_model():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
    model.eval()
    return model, weights.transforms()

def process_frame(frame, model, transforms):
    """Process a single frame for object detection"""
    frame = cv2.resize(frame, (640, 480))
    img_tensor = transforms(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    
    with torch.no_grad():
        prediction = model([img_tensor])
    
    return prediction[0]

def detect_objects():
    """Opens the camera, detects objects, and returns detected labels"""
    model, transforms = load_model()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        st.error("Unable to access camera. Please check your camera permissions.")
        return None
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("Failed to capture image from camera.")
        return None
    
    prediction = process_frame(frame, model, transforms)
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    class_names = weights.meta["categories"]
    
    detected_labels = {class_names[label].lower() for label, score in zip(labels, scores) if score > 0.7}
    return detected_labels

def show_ar_page():
    st.title("Learn with AR")
    
    st.write("Click below to start AR object detection.")
    if st.button("Start AR Detection"):
        detected_objects = detect_objects()
        if detected_objects:
            st.write("Detected objects:", detected_objects)
            
            if "cell phone" in detected_objects:
                st.write("Displaying 3D Model of Cell Phone")
                st.components.v1.iframe("https://tinyglb.com/viewer/06d63d40c76c4e4a8c7a6360401b0fb5", height=500)  # Updated link to the 3D model
            else:
                st.write("No AR-supported object detected.")


def reward_shop_page():
    navigation_buttons()
    
    # Apply styling consistent with other pages
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
        .reward-card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            transition: transform 0.3s;
        }
        .reward-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .coin-display {
            background-color: #FFD700;
            color: black;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        .redeem-button {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        /* Tab styling to ensure black text and orange on hover for all tabs (Electronics, Courses, Merchandise) */
        .stTabs [role="tab"] {
            color: black !important;
            transition: color 0.3s;
        }
        .stTabs [role="tab"]:hover {
            color: orange !important;
        }
        /* Make sure the selected tab indicator is orange too */
        .stTabs [role="tab"][aria-selected="true"] {
            color: black !important;
            border-bottom-color: orange !important;
        }
        /* Style for coin prices */
        .coin-price {
            font-weight: bold;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main layout
    left_col, main_col, right_col = st.columns([1,2,1])
    
    # Left Navigation (same as in dashboard)
    with left_col:
        st.markdown("### Navigation")
        if st.button("👤 Profile", key="nav_profile"):
            navigate_to('profile')
        if st.button("📷 Search with Camera", key="nav_camera"):
            navigate_to('camera')
        if st.button("🧠 Learn with AR", key="nav_ar"):  # Add this line
            navigate_to('ar')  # Navigate to AR page when clicked
        if st.button("👥 Collaborate", key="nav_collaborate"):
            navigate_to('collaborate')
        if st.button("🏪 Reward Shop", key="nav_reward"):  # Update this line
            navigate_to('reward_shop')
        if st.button("🤖 DoubtAI", key="nav_doubtai"):
            navigate_to('doubtai')
        if st.button("🏅 Certificates", key="nav_certificates"):
            navigate_to('certificates')
        if st.button("⚙️ Settings", key="nav_settings"):
            navigate_to('settings')
    
    # Main Content
    with main_col:
        # Display user points at the top-left
        points_container = st.container()
        with points_container:
            st.markdown("""
                <div class="coin-display">
                    🪙 1250 XenCoins
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("## 🏪 Reward Shop")
        st.markdown("Redeem your XenCoins for exciting rewards!")
        
        # Create tabs for different categories
        tab1, tab2, tab3 = st.tabs(["Electronics", "Courses", "Merchandise"])
        
        # Electronics tab
        with tab1:
            electronics = [
                {"name": "Premium Headphones", "price": 5000, "description": "High-quality over-ear headphones with noise cancellation", "image": "headphones.jpg"},
                {"name": "Wireless Earbuds", "price": 3000, "description": "Lightweight earbuds with 24-hour battery life", "image": "earbuds.jpg"},
                {"name": "Portable Bluetooth Speaker", "price": 2500, "description": "Compact, water-resistant speaker for on-the-go learning", "image": "speaker.jpg"},
                {"name": "Digital Stylus Pen", "price": 1800, "description": "Precision stylus for tablets and touchscreens", "image": "stylus.jpg"}
            ]
            
            # Create 2 columns for the items
            cols = st.columns(2)
            for i, item in enumerate(electronics):
                with cols[i % 2]:
                    st.markdown(f"""
                        <div class="reward-card">
                            <h4>{item['name']}</h4>
                            <p style="color: gray;">{item['description']}</p>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="coin-price">🪙 {item['price']}</span>
                                <button class="redeem-button">Redeem</button>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Courses tab
        with tab2:
            courses = [
                {"name": "Advanced Data Science", "price": 2000, "description": "Master the art of data analysis and visualization", "level": "Advanced", "duration": "8 weeks"},
                {"name": "AI & Machine Learning", "price": 2500, "description": "Learn the fundamentals of AI and ML algorithms", "level": "Intermediate", "duration": "10 weeks"},
                {"name": "Web Development Bootcamp", "price": 1800, "description": "Comprehensive course on full-stack web development", "level": "Beginner", "duration": "12 weeks"},
                {"name": "Digital Marketing Masterclass", "price": 1500, "description": "Learn SEO, social media marketing, and more", "level": "Intermediate", "duration": "6 weeks"},
                {"name": "3D Modeling & Animation", "price": 2200, "description": "Create professional 3D models and animations", "level": "Intermediate", "duration": "8 weeks"}
            ]
            
            # Create 2 columns for the items
            cols = st.columns(2)
            for i, course in enumerate(courses):
                with cols[i % 2]:
                    st.markdown(f"""
                        <div class="reward-card">
                            <h4>{course['name']}</h4>
                            <p style="color: gray;">{course['description']}</p>
                            <p>Level: {course['level']} | Duration: {course['duration']}</p>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="coin-price">🪙 {course['price']}</span>
                                <button class="redeem-button">Unlock Course</button>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Merchandise tab
        with tab3:
            merchandise = [
                {"name": "XenLearn T-Shirt", "price": 800, "description": "Comfortable cotton t-shirt with XenLearn logo"},
                {"name": "Study Planner Notebook", "price": 500, "description": "Premium notebook with study planning templates"},
                {"name": "XenLearn Backpack", "price": 1200, "description": "Durable backpack with laptop compartment"},
                {"name": "Water Bottle", "price": 400, "description": "Eco-friendly insulated water bottle"}
            ]
            
            # Create 2 columns for the items
            cols = st.columns(2)
            for i, item in enumerate(merchandise):
                with cols[i % 2]:
                    st.markdown(f"""
                        <div class="reward-card">
                            <h4>{item['name']}</h4>
                            <p style="color: gray;">{item['description']}</p>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="coin-price">🪙 {item['price']}</span>
                                <button class="redeem-button">Redeem</button>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
    
    # Right Sidebar - Personalized Recommendations
    with right_col:
        st.markdown("### Recommended for You")
        recommended_items = [
            {"name": "Wireless Earbuds", "price": 3000},
            {"name": "AI & Machine Learning", "price": 2500},
            {"name": "XenLearn Backpack", "price": 1200}
        ]
        
        for item in recommended_items:
            st.markdown(f"""
                <div style='background-color: white; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <h4 style='color: black;'>{item['name']}</h4>
                    <p class="coin-price">🪙 {item['price']}</p>
                    <button class="redeem-button" style="width: 100%;">Redeem</button>
                </div>
            """, unsafe_allow_html=True)
        
        # Display recent transactions
        st.markdown("### Recent Transactions")
        transactions = [
            {"item": "Data Science Course", "date": "Feb 20, 2025", "coins": "-2000"},
            {"item": "Daily Login Bonus", "date": "Feb 24, 2025", "coins": "+50"},
            {"item": "Course Completion", "date": "Feb 15, 2025", "coins": "+500"}
        ]
        
        for transaction in transactions:
            coin_color = "green" if "+" in transaction["coins"] else "red"
            st.markdown(f"""
                <div style='background-color: white; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='color: black;'>{transaction['item']}</span>
                        <span style='color: {coin_color};'>{transaction['coins']}</span>
                    </div>
                    <small style='color: gray;'>{transaction['date']}</small>
                </div>
            """, unsafe_allow_html=True)

def settings_page():
    navigation_buttons()
    st.title("Settings")
    st.write("Adjust your preferences here.")
    
    # Theme Change
    theme = st.selectbox("Select Theme", ["Light", "Dark"])
    st.session_state.theme = theme
    st.write(f"Selected theme: {theme}")
    
    # Notification Preferences
    notifications = st.checkbox("Enable Notifications", value=True)
    st.session_state.notifications = notifications
    st.write(f"Notifications: {'Enabled' if notifications else 'Disabled'}")
    
    # Adjust user settings (example: preferred language)
    language = st.selectbox("Preferred Language", ["English", "Spanish", "French"])
    st.session_state.language = language
    st.write(f"Selected language: {language}")


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
        prompt = f'''You are an expert educational content creator specializing in project-based learning for high school and college students. Given these objects detected in the learning environment: {objects_list}, generate 3 sophisticated educational projects or experiments.

        Requirements:
        1. Projects must directly utilize the detected objects in meaningful ways
        2. Focus on STEM, research, or analytical skills
        3. Projects should be challenging but achievable
        4. Each project must have clear educational value and practical applications
        5. No basic or elementary-level projects
        6. No projects that could be done without the listed objects
        7. Include specific technical details and measurements where relevant
        8. Materials used should strictly only be among {objects_list} or be something 

        For each project, provide:
        1. **Title**: Professional and descriptive title
        2. **Description**: Detailed overview including scientific/technical principles involved
        3. **Difficulty Level**: Specify ONLY as: "easy", "intermediate", or "hard"
        4. **Estimated Time**: Realistic time frame including setup and execution
        5. **Materials Needed**: Comprehensive list with specific requirements/specifications
        6. **Step-by-Step Instructions**: VERY detailed technical steps with exact measurements, timings, and specific procedures. No vague instructions.
        7. **Learning Outcomes**: Specific technical skills and knowledge gained
        8. **Safety Precautions**: Include ONLY if there are genuine safety concerns, otherwise leave empty

        IMPORTANT: Return ONLY a valid JSON array with exactly this structure and no additional text or markdown:
        [
            {{
                "title": "Project Title",
                "description": "Technical description with principles",
                "difficulty": "easy/intermediate/hard",
                "time": "estimated time",
                "materials": ["item1 with specifications", "item2 with specifications"],
                "steps": [
                    "Step 1: Mix exactly 100ml of solution A with 50ml of solution B",
                    "Step 2: Heat the mixture to precisely 75°C for 10 minutes"
                ],
                "learning_outcomes": [
                    "Specific technical skill 1",
                    "Specific knowledge gained 2"
                ],
                "safety_precautions": []
            }}
        ]'''

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Add safety check for response
        if not response.text:
            return []
        
        # Clean the response text
        cleaned_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in cleaned_text:
            cleaned_text = cleaned_text.split("```json")[1].split("```")[0]
        elif "```" in cleaned_text:
            cleaned_text = cleaned_text.split("```")[1]
            
        cleaned_text = cleaned_text.strip()
        
        # Debug output
        print("Cleaned text:", cleaned_text)
        
        try:
            suggestions = json.loads(cleaned_text)
            if isinstance(suggestions, list):
                return suggestions
            return []
        except json.JSONDecodeError as je:
            print(f"JSON Error: {je}")
            print("Failed text:", cleaned_text)
            return []
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def object_detection():
    navigation_buttons()  # Add this as first line
    st.title("Course Finder - Photo Analysis")
    
    # Add custom CSS for dark text on light background
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
        .project-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .project-container * {
            color: white !important;
        }
        .stMarkdown {
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: black !important;
        }
        p {
            color: black !important;
        }
        li {
            color: black !important;
        }
        .stButton button {
            color: white !important;
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
        if st.button("👤 Profile", key="nav_profile"):
            navigate_to('profile')
        
        if st.button("📷 Search with Camera", key="nav_camera"):
            navigate_to('camera')
        if st.button("👥 Collaborate", key="nav_collaborate"):
            navigate_to('collaborate')
        st.button("🏪 Reward Shop")
        if st.button("🤖 DoubtAI", key="nav_doubtai"):
            navigate_to('chatbot')  # Point the DoubtAI to the chatbot page
        if st.button("🎓 Certifications"):
            st.subheader("Certifications")
            
            # Add tabs for different certification categories
            cert_tabs = st.tabs(["Technical", "Professional", "Academic"])
            
            with cert_tabs[0]:  # Technical certifications
                st.markdown("#### Technical Certifications")
                tech_certs = [
                    {"name": "AWS Certified Solutions Architect", "issued": "Jan 2024", "expires": "Jan 2027"},
                    {"name": "Microsoft Azure Developer Associate", "issued": "Mar 2023", "expires": "Mar 2026"},
                    {"name": "Google Cloud Professional Data Engineer", "issued": "Jul 2024", "expires": "Jul 2027"}
                ]
                
                for cert in tech_certs:
                    with st.expander(cert["name"]):
                        st.write(f"**Issued:** {cert['issued']}")
                        st.write(f"**Expires:** {cert['expires']}")
                        st.download_button("Download Certificate", "certificate_data", f"{cert['name']}.pdf")
            
            with cert_tabs[1]:  # Professional certifications
                st.markdown("#### Professional Certifications")
                prof_certs = [
                    {"name": "Project Management Professional (PMP)", "issued": "Sep 2022", "expires": "Sep 2025"},
                    {"name": "Scrum Master Certification", "issued": "Nov 2023", "expires": "Nov 2025"}
                ]
                
                for cert in prof_certs:
                    with st.expander(cert["name"]):
                        st.write(f"**Issued:** {cert['issued']}")
                        st.write(f"**Expires:** {cert['expires']}")
                        st.download_button("Download Certificate", "certificate_data", f"{cert['name']}.pdf")
            
            with cert_tabs[2]:  # Academic certifications
                st.markdown("#### Academic Certificates")
                acad_certs = [
                    {"name": "Data Science Specialization", "institution": "Stanford University", "completed": "May 2023"},
                    {"name": "AI and Machine Learning", "institution": "MIT", "completed": "Dec 2022"}
                ]
                
                for cert in acad_certs:
                    with st.expander(cert["name"]):
                        st.write(f"**Institution:** {cert['institution']}")
                        st.write(f"**Completed:** {cert['completed']}")
                        st.download_button("Download Certificate", "certificate_data", f"{cert['name']}.pdf")
            
            # Add button to upload new certificates
            st.markdown("#### Upload New Certificate")
            cert_file = st.file_uploader("Choose a certificate file", type=["pdf", "jpg", "png"])
            cert_name = st.text_input("Certificate Name")
            cert_type = st.selectbox("Certificate Type", ["Technical", "Professional", "Academic"])
            
            if st.button("Upload Certificate") and cert_file is not None and cert_name:
                st.success(f"Certificate '{cert_name}' uploaded successfully!")
        if st.button("⚙️ Settings"):
            st.subheader("Settings")
            
            # User preferences section
            st.markdown("#### User Preferences")
            theme = st.selectbox("Theme", ["Light", "Dark", "System Default"])
            notifications = st.toggle("Enable Notifications", value=True)
            language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Chinese"])
            
            # Application settings section
            st.markdown("#### Application Settings")
            auto_save = st.slider("Auto-save Interval (minutes)", 1, 60, 15)
            font_size = st.select_slider("Font Size", options=["Small", "Medium", "Large"])
            
            # Data privacy section
            st.markdown("#### Data Privacy")
            data_sharing = st.toggle("Allow Anonymous Usage Data", value=False)
            
            # Save settings button
            if st.button("Save Settings"):
                st.success("Settings saved successfully!")
    
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
    navigation_buttons()
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
        st.metric("🪙 XenCoins", "50", "+50 Daily Login Bonus")
    with col3:
        st.metric("⚡ Streak", "1 days", "Great Going!")
    
    # Main layout
    left_col, main_col, right_col = st.columns([1,2,1])
    
    # Left Navigation
    # Update dashboard and other relevant pages with this navigation code:
    with left_col:
        st.markdown("### Navigation")
        if st.button("👤 Profile", key="nav_profile"):
            navigate_to('profile')
        if st.button("🧠 Learn with AR", key="nav_ar"):  # Add this line
            navigate_to('ar')  # Navigate to AR page when clicked
        st.button("📚 Courses")
        if st.button("📷 Search with Camera", key="nav_camera"):
            navigate_to('camera')
        if st.button("👥 Collaborate", key="nav_collaborate"):
            navigate_to('collaborate')
            st.rerun()
        if st.button("🏪 Reward Shop", key="nav_reward"):  # Update this line
            navigate_to('reward_shop')
        if st.button("🤖 DoubtAI", key="nav_doubtai"):
            navigate_to('doubtai')  # Point the DoubtAI to the chatbot page
        if st.button("🏅 Certificates", key="nav_certificates"):
            navigate_to('certificates')
            
        if st.button("⚙️ Settings", key="nav_settings"):
            navigate_to('settings')
    
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
            
def login_page():
    """Display the login and signup page"""
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
            
            if st.button("Login", key="login_button"):
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
            
            if st.button("Sign Up", key="signup_button"):
                if not all([full_name, email, password, confirm_password]):
                    st.warning("Please fill in all fields.")
                elif password != confirm_password:
                    st.warning("Passwords do not match.")
                else:
                    if handle_signup(email, password, full_name):
                        st.success("Sign up successful! Please proceed to set your interests.")
                        st.rerun()

def main():
    initialize_session_state()
    st.set_page_config(page_title="XenLearn", layout="wide")
    
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = []
    
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
    elif st.session_state.current_page == 'doubtai':  
        doubtai_page()
    elif st.session_state.current_page == 'certificates':
        certificates_page()  # Add this page function
    elif st.session_state.current_page == 'settings':
        settings_page()  # Add this page function
    elif st.session_state.current_page == 'reward_shop':  # Add this new condition
        reward_shop_page()
    elif st.session_state.current_page == 'ar':
        show_ar_page()

if __name__ == "__main__":
    main()
