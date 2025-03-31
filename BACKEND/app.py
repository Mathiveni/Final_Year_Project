import os
import tempfile
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import base64
import cv2
import logging

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = '/Users/vinoth/Desktop/Final_yr_project/BACKEND/Models/best_cnn_model.h5'
model = None

idx_to_class = {
    0: "CNV (Choroidal Neovascularization)",
    1: "DME (Diabetic Macular Edema)",
    2: "Drusen",
    3: "Normal"
}


def load_model_on_startup():
    global model
    try:
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


# Helper function to convert NumPy types to native Python types
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


# Calculate a more comprehensive severity assessment.
def calculate_advanced_severity(heatmap, predictions, predicted_class_index):
    # 1. Heatmap-based severity
    heatmap_severity = float(np.mean(heatmap) * 100)

    # 2. Confidence-based severity contribution
    confidence_severity = float(predictions[0][predicted_class_index] * 100)

    # 3. Spatial intensity analysis
    spatial_intensity = float(np.std(heatmap) * 100)  # Higher std indicates more localized abnormalities

    # 4. Top abnormal regions analysis
    top_percentile = float(np.percentile(heatmap, 90) * 100)  # Focus on top 10% intense regions

    # Weighted severity calculation
    severity_components = {
        'Heatmap Severity': round(heatmap_severity * 0.4, 2),
        'Confidence Severity': round(confidence_severity * 0.3, 2),
        'Spatial Intensity': round(spatial_intensity * 0.2, 2),
        'Top Regions Intensity': round(top_percentile * 0.1, 2)
    }

    # Combined severity score
    total_severity = sum(severity_components.values())

    # Severity level determination with more granular classification
    if total_severity < 25:
        severity_level = "Minimal"
    elif total_severity < 40:
        severity_level = "Mild"
    elif total_severity < 60:
        severity_level = "Moderate"
    elif total_severity < 80:
        severity_level = "Severe"
    else:
        severity_level = "Critical"

    return {
        'score': round(total_severity, 2),
        'level': severity_level,
        'components': severity_components
    }


def grad_cam(model, img_array, layer_name, class_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            predicted_class = tf.argmax(predictions[0])
        else:
            predicted_class = class_index
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradient computation failed")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0) / (np.max(heatmap.numpy()) + 1e-10)

    return heatmap, int(predicted_class.numpy())


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))  # Resize to match original image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlayed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)  # Overlay heatmap on original image
    return overlayed_img


def image_to_base64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', img_rgb)
    return base64.b64encode(buffer).decode('utf-8')


def get_extended_explanation(predicted_class_name, severity_info):
    """Provide extended explanation for the predicted class."""
    extended_explanations = {
        "CNV (Choroidal Neovascularization)": (
            "The AI highlighted abnormal vascular regions, often associated with excessive blood vessel growth. "
            "These regions may indicate leakage or neovascularization, commonly seen in wet AMD. "
            "The heatmap shows the AI's focus on irregular patterns in the retina, which aligns with CNV characteristics."
            "\n\nNext Step: Confirm with Fluorescein Angiography or OCT Angiography to assess neovascularization."
        ),
        "DME (Diabetic Macular Edema)": (
            "The AI detected fluid accumulation in the macula, emphasizing regions with potential swelling. "
            "The highlighted areas suggest changes in retinal thickness, which are key indicators of macular edema. "
            "The intensity of the heatmap in the central macular zone supports this diagnosis."
            "\n\nNext Step: Confirm with Fundus Photography or Additional OCT scans to evaluate macular thickness."
        ),
        "Drusen": (
            "The AI focused on bright, distinct deposits beneath the retina, which are characteristic of Drusen. "
            "These deposits, often found near the macula, can contribute to vision impairment if they grow larger. "
            "The heatmap highlights these abnormal deposits, reinforcing the likelihood of this condition."
            "\n\nNext Step: Regular OCT scans are advised to monitor Drusen size and density."
        ),
        "Normal": (
            "The AI did not find significant abnormalities in the retinal structure, leading to a normal classification. "
            "The absence of heatmap intensity in critical regions suggests no concerning signs of disease. "
            "A well-defined and evenly structured retina supports this assessment."
            "\n\nNext Step: Routine eye exams are still recommended for continued eye health."
        )
    }

    heatmap_explanation = (
        "Red areas indicate the most critical regions influencing the AI's decision, suggesting high abnormality. "
        "Orange and yellow areas represent moderate attention, possibly indicating early signs of disease. "
        "Blue and green areas contribute the least to the decision, implying normal or less concerning regions."
    )

    return {
        "explanation": extended_explanations.get(predicted_class_name, "No additional details available."),
        "heatmap_interpretation": heatmap_explanation
    }


def get_severity_explanation(predicted_class_name, severity_info):
    """Provide extended explanation for the predicted class with severity context."""
    extended_explanations = {
        "CNV (Choroidal Neovascularization)": {
            "Minimal": "Early stage with minimal vascular irregularities. Close monitoring recommended.",
            "Mild": "Moderate vascular changes detected. Further investigation advised.",
            "Moderate": "Significant vascular abnormalities present. Immediate medical consultation recommended.",
            "Severe": "Advanced vascular growth detected. Urgent medical intervention required.",
            "Critical": "Extensive vascular complications. Immediate specialized treatment necessary."
        },
        "DME (Diabetic Macular Edema)": {
            "Minimal": "Initial signs of fluid accumulation. Regular check-ups suggested.",
            "Mild": "Slight macular edema detected. More frequent monitoring needed.",
            "Moderate": "Notable fluid buildup in macula. Prompt medical assessment required.",
            "Severe": "Significant macular swelling. Urgent treatment recommended.",
            "Critical": "Extensive macular edema. Immediate medical intervention critical."
        },
        "Drusen": {
            "Minimal": "Few small drusen deposits. Standard monitoring advised.",
            "Mild": "Moderate drusen presence. More frequent eye exams recommended.",
            "Moderate": "Significant drusen accumulation. Potential risk of progression.",
            "Severe": "Extensive drusen formation. High risk of further complications.",
            "Critical": "Dense drusen network. Immediate comprehensive evaluation needed."
        },
        "Normal": {
            "Minimal": "Completely normal retinal structure. Routine eye health maintenance.",
            "Mild": "Near-normal condition. No significant concerns detected.",
            "Moderate": "Generally healthy, with minor variations within normal range.",
            "Severe": "Unexpected complexity in seemingly normal scan. Further investigation recommended.",
            "Critical": "Anomalous findings require comprehensive additional screening."
        }
    }

    severity_level = severity_info['level']
    explanation = extended_explanations[predicted_class_name].get(severity_level, "No specific details available.")

    return {
        "explanation": explanation,
        "severity_components": severity_info['components']
    }


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    file.save(temp_file.name)

    try:
        img = image.load_img(temp_file.name, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array_with_batch = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array_with_batch)
        logger.info(f"Raw predictions shape: {predictions.shape}")

        predicted_class_index = np.argmax(predictions[0])
        predicted_class = idx_to_class[predicted_class_index]
        logger.info(f"Predicted class: {predicted_class}")

        confidence = float(predictions[0][predicted_class_index] * 100)
        logger.info(f"Confidence: {confidence}%")

        # Format all probabilities with proper rounding
        all_probabilities = {idx_to_class[i]: round(float(predictions[0][i] * 100), 2) for i in
                             range(len(idx_to_class))}
        logger.info(f"All probabilities: {all_probabilities}")

        last_conv_layer_name = find_last_conv_layer(model)
        logger.info(f"Last conv layer: {last_conv_layer_name}")

        heatmap, _ = grad_cam(model, img_array_with_batch, last_conv_layer_name)

        # Advanced severity calculation
        severity_info = calculate_advanced_severity(heatmap, predictions, predicted_class_index)
        logger.info(f"Severity info: {convert_to_serializable(severity_info)}")

        original_img = cv2.imread(temp_file.name)
        overlayed_img = overlay_heatmap(original_img, heatmap)

        # Convert images to base64
        original_img_base64 = image_to_base64(original_img)
        overlayed_img_base64 = image_to_base64(overlayed_img)

        # Get extended explanation with severity context
        extended_explanation = get_extended_explanation(predicted_class, severity_info)
        severity_explanation = get_severity_explanation(predicted_class, severity_info)

        # Create response data
        response_data = {
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'severity_score': severity_info['score'],
            'severity_level': severity_info['level'],
            'severity_components': severity_info['components'],
            'severity_explanation': severity_explanation['explanation'],
            'severity_details': severity_explanation['severity_components'],
            'all_probabilities': all_probabilities,
            'original_image': original_img_base64,
            'heatmap_image': overlayed_img_base64,
            'explanation': extended_explanation['explanation'],
            'heatmap_interpretation': extended_explanation['heatmap_interpretation']
        }

        # Convert any NumPy types to native Python types for JSON serialization
        serializable_data = convert_to_serializable(response_data)

        # Log the full response data structure (except images which are too large)
        log_data = serializable_data.copy()
        log_data['original_image'] = '[BASE64 IMAGE DATA]'
        log_data['heatmap_image'] = '[BASE64 IMAGE DATA]'
        logger.info(f"Response data: {log_data}")

        return jsonify(serializable_data)

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

    finally:
        os.unlink(temp_file.name)


if __name__ == '__main__':
    load_model_on_startup()
    app.run(host='0.0.0.0', port=8080, debug=True)