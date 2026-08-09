"""
app.py
Entry point for the LeafScan AI application.
"""

from app import app
from app.core.model_handler import get_model

# Pre-load the AI model during application startup
# This ensures the server is ready to handle requests instantly
get_model()

if __name__ == '__main__':
    # Hugging Face Spaces requires the app to bind to 0.0.0.0 on port 7860
    app.run(host="0.0.0.0", port=7860, debug=False)
