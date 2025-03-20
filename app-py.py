from flask import Flask, request, render_template, jsonify, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import logging
from datetime import datetime
import time
from functools import lru_cache
import json
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define class labels with severity scores for better outputs
class_info = {
    "No_DR": {"index": 2, "severity": 0, "description": "No diabetic retinopathy detected"},
    "Mild": {"index": 0, "severity": 1, "description": "Mild non-proliferative diabetic retinopathy"},
    "Moderate": {"index": 1, "severity": 2, "description": "Moderate non-proliferative diabetic retinopathy"},
    "Severe": {"index": 4, "severity": 3, "description": "Severe non-proliferative diabetic retinopathy"},
    "Proliferate_DR": {"index": 3, "severity": 4, "description": "Proliferative diabetic retinopathy"}
}

# Reverse mapping from index to class name
index_to_class = {info["index"]: class_name for class_name, info in class_info.items()}

# Model configurations
model_paths = {
    "ResNet18": "models/resnet18.pth",
    "EfficientNet": "models/efficientnet.pth",
    "DenseNet121": "models/densenet121.pth",
    "Xception": "models/xception.pth",
    "MobileNetV2": "models/mobilenetv2.pth"
}

# Model weights for ensemble (can be adjusted based on individual model performance)
model_weights = {
    "ResNet18": 1.0,
    "EfficientNet": 1.2,
    "DenseNet121": 1.1,
    "Xception": 1.0,
    "MobileNetV2": 0.9
}


# Cache for model loading to improve performance
@lru_cache(maxsize=None)
def load_model(model_name, model_path):
    """Load model with caching for better performance"""
    logger.info(f"Loading model: {model_name}")
    start_time = time.time()
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        logger.info(f"Model {model_name} loaded in {time.time() - start_time:.2f} seconds")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        return None


# Lazy loading of models when first needed
models = {}

def get_models():
    """Lazy load models when needed"""
    if not models:
        for name, path in model_paths.items():
            model = load_model(name, path)
            if model is not None:
                models[name] = model
        if not models:
            raise ValueError("No models could be loaded")
    return models


# Enhanced image transformations with multiple preprocessing options
standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Alternative transform for images that might need more preprocessing
enhanced_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image, use_enhanced=False):
    """Preprocess image before feeding it to the model"""
    try:
        # Convert grayscale to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply appropriate transform
        if use_enhanced:
            image_tensor = enhanced_transform(image).unsqueeze(0)
        else:
            image_tensor = standard_transform(image).unsqueeze(0)
            
        return image_tensor.to(device)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise


def predict_with_confidence(image, use_enhanced=False):
    """Get predictions from all models with confidence scores"""
    try:
        # Get available models
        available_models = get_models()
        if not available_models:
            raise ValueError("No models available for prediction")
        
        # Process image
        processed_image = preprocess_image(image, use_enhanced)
        
        # Collect outputs from each model
        model_outputs = {}
        for name, model in available_models.items():
            start_time = time.time()
            with torch.no_grad():
                output = model(processed_image)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                model_outputs[name] = {
                    "raw_output": output.cpu().numpy(),
                    "probabilities": probabilities.cpu().numpy(),
                    "time_taken": time.time() - start_time
                }
        
        # Weighted ensemble voting
        weighted_sum = np.zeros((1, len(class_info)))
        for name, output_data in model_outputs.items():
            weight = model_weights.get(name, 1.0)
            weighted_sum += weight * output_data["raw_output"]
        
        # Convert to probabilities
        ensemble_probabilities = torch.nn.functional.softmax(torch.from_numpy(weighted_sum), dim=1)[0].numpy()
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(ensemble_probabilities)
        predicted_class = index_to_class[predicted_idx]
        confidence = ensemble_probabilities[predicted_idx] * 100
        
        # Get individual model predictions
        model_predictions = {}
        for name, output_data in model_outputs.items():
            probs = output_data["probabilities"]
            model_pred_idx = np.argmax(probs)
            model_predictions[name] = {
                "prediction": index_to_class[model_pred_idx],
                "confidence": float(probs[model_pred_idx] * 100),
                "time_taken": output_data["time_taken"]
            }
        
        # Create result object
        result = {
            "prediction": predicted_class,
            "confidence": float(confidence),
            "severity": class_info[predicted_class]["severity"],
            "description": class_info[predicted_class]["description"],
            "class_probabilities": {index_to_class[i]: float(prob * 100) for i, prob in enumerate(ensemble_probabilities)},
            "model_predictions": model_predictions
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}\n{traceback.format_exc()}")
        raise


def save_upload_file(file):
    """Save uploaded file with timestamp to prevent overwriting"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath


@app.route("/", methods=["GET"])
def index():
    """Render the main page"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """Endpoint for image prediction"""
    start_time = time.time()
    
    try:
        # Check if file exists in request
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files["file"]
        
        # Check if filename is valid
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
            
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Please upload a {', '.join(app.config['ALLOWED_EXTENSIONS'])} image"}), 400
        
        # Get preprocessing preference
        use_enhanced = request.form.get('enhance', 'false').lower() == 'true'
        
        # Save file
        filepath = save_upload_file(file)
        
        # Open image and make prediction
        with Image.open(filepath) as image:
            result = predict_with_confidence(image, use_enhanced)
            
        # Add performance metrics
        result["processing_time"] = time.time() - start_time
        
        # Log successful prediction
        logger.info(f"Successful prediction for {file.filename}: {result['prediction']} with {result['confidence']:.2f}% confidence")
        
        return jsonify(result)
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Prediction error: {error_message}\n{traceback.format_exc()}")
        return jsonify({"error": error_message}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint for monitoring system health"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "models_loaded": list(models.keys()) if models else [],
        "models_available": list(model_paths.keys())
    }
    return jsonify(health_data)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle large file uploads"""
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413


@app.errorhandler(500)
def server_error(error):
    """Handle server errors"""
    logger.error(f"Server error: {str(error)}")
    return jsonify({"error": "Internal server error occurred"}), 500


if __name__ == "__main__":
    # Initialize models before starting server
    try:
        get_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models at startup: {str(e)}")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=False)
