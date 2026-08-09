import os
import logging
from flask import Flask

# Suppress TensorFlow logging in production to keep terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder for temporary image storage
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Import routes at the bottom to avoid circular dependencies
from app import routes
