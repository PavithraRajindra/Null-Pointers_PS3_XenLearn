import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import calendar
import time
from PIL import Image
import socket
import json
import random

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
        st.session_state.user_profile = {
            'email': 'user@example.com',
            'password': '********',
            'interests': []
        }

def login_page():
    st.markdown("""
        <style>
        .login-container { padding: 2rem; }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://via.placeholder.com/150", caption="Logo")
        st.markdown("### EcoLearn")
        st.markdown("*Transforming physical resources into digital learning experiences*")
    
    with col2:
        st.markdown("### Welcome!")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                st.session_state.current_page = 'interests'
                st.session_state.logged_in = True
                st.rerun()
                
        with tab2:
            st.text_input("Full Name", key="signup_name")
            st.text_input("Email", key="signup_email")
            st.text_input("Password", type="password", key="signup_password")
            st.text_input("Confirm Password", type="password", key="signup_confirm")
            if st.button("Sign Up"):
                st.session_state.current_page = 'interests'
                st.session_state.logged_in = True
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

def object_detection():
    st.title("Object Detection")
    
    # Start camera
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button = st.button("Stop Camera")
    
    # Load pre-trained model for pen detection (using basic color detection for demo)
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for blue pen color (adjust these values based on your needs)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask and detect pen
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert colors from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        # If pen detected
        if np.sum(mask) > 5000:  # Adjust threshold as needed
            st.success("Pen detected!")
            break
        
        time.sleep(0.1)
    
    cap.release()

def profile_page():
    st.title("Profile Settings")
    
    with st.form("profile_form"):
        new_email = st.text_input("Email", value=st.session_state.user_profile['email'])
        new_password = st.text_input("New Password", type="password", placeholder="Leave blank to keep current")
        
        if st.form_submit_button("Save Changes"):
            st.session_state.user_profile['email'] = new_email
            if new_password:
                st.session_state.user_profile['password'] = new_password
            st.success("Profile updated successfully!")

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
    st.set_page_config(page_title="EcoLearn", layout="wide")
    
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
