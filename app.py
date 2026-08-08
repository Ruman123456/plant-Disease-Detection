"""
app.py
Main web routing file for the LeafScan AI application.
Built with Flask.
"""

import os
import logging
from flask import Flask, render_template, request

# Import our modularized code
from utils import disease_info
from model_handler import predict_disease, get_model

# Suppress TensorFlow logging in production to keep terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize Flask app
app = Flask(__name__, template_folder='Templates')

# Pre-load the AI model during application startup
# This ensures the server is ready to handle requests instantly
get_model()

# Configure upload folder for temporary image storage
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for security
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the uploaded file has a valid image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image uploads, run AI prediction, and return results."""
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Process only if the file is an allowed image
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Pass the image path to our model handler to get the diagnosis
            result = predict_disease(filepath)
            
            # Render the result template with the diagnosis data
            return render_template(
                'result.html',
                prediction=result['prediction'],
                cause=result['cause'],
                treatment=result['treatment'],
                image_path=f'/{filepath}'
            )
        except Exception as e:
            return f"Error processing the image: {str(e)}"
        
        # Note: In a true production environment, you might want to delete 
        # the file here using os.remove(filepath) after rendering, to save disk space.
        # However, since the result.html template needs to display the image via an <img> tag, 
        # we have to keep it on disk for the browser to fetch it.
    else:
        return "Invalid file type. Only PNG, JPG, and JPEG files are allowed."

@app.route('/diseases')
def diseases():
    """Render the disease library, passing in the static disease data."""
    return render_template('diseases.html', disease_info=disease_info)

@app.route('/feedback')
def feedback():
    """Render the feedback form."""
    return render_template('feedback.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Handle feedback submission."""
    feedback_text = request.form.get('feedback', '')
    category = request.form.get('category', '')
    rating = request.form.get('rating', '')
    
    # In a real app, this would be saved to a database (e.g., SQLite or PostgreSQL)
    print(f"Feedback Received | Rating: {rating}/5 | Category: {category} | Note: {feedback_text}")
    
    return "Thank you for your feedback!"

# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == '__main__':
    # Hugging Face Spaces requires the app to bind to 0.0.0.0 on port 7860
    app.run(host="0.0.0.0", port=7860, debug=False)
