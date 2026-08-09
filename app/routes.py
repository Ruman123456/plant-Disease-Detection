import os
from flask import render_template, request
from app import app
from app.core.utils import disease_info
from app.core.model_handler import predict_disease

# Allowed file extensions for security
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the uploaded file has a valid image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            # Use a relative URL path that Flask can serve via its static route
            image_url = f'/static/uploads/{file.filename}'
            
            return render_template(
                'result.html',
                prediction=result['prediction'],
                cause=result['cause'],
                treatment=result['treatment'],
                image_path=image_url
            )
        except Exception as e:
            return f"Error processing the image: {str(e)}"
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
