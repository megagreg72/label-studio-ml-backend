import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from label_studio_ml.api import init_app
from model import AWSRekognitionObjectDetector

# Initialize the Flask app
app = init_app(AWSRekognitionObjectDetector)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 9090))
    app.run(host='0.0.0.0', port=port, debug=False)
