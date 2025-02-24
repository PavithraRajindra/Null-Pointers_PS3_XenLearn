# import pyrebase

# firebase_config = {
#     apiKey: "AIzaSyAKo7utty3s6t-9Xp1A81NIAIounqy9p94",
#   authDomain: "xenlearn-f58d3.firebaseapp.com",
#   projectId: "xenlearn-f58d3",
#   storageBucket: "xenlearn-f58d3.firebasestorage.app",
#   messagingSenderId: "621588213526",
#   appId: "1:621588213526:web:67b9b8ce7ed46d4f376a37"
# }

# firebase = pyrebase.initialize_app(firebase_config)
# auth = firebase.auth()

import pyrebase

firebase_config = {
    "apiKey": "AIzaSyAKo7utty3s6t-9Xp1A81NIAIounqy9p94",
    "authDomain": "xenlearn-f58d3.firebaseapp.com",
    "projectId": "xenlearn-f58d3",
    "storageBucket": "xenlearn-f58d3.appspot.com",
    "messagingSenderId": "621588213526",
    "appId": "1:621588213526:web:67b9b8ce7ed46d4f376a37",
    "databaseURL": "https://console.firebase.google.com/project/xenlearn-f58d3/database/xenlearn-f58d3-default-rtdb/data/~2F"  # Add this if you're using Firebase Database
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
