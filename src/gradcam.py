import os

import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input


# ============================================================
# MODEL
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "resnet50v2.keras"
)

model = load_model(MODEL_PATH)

LAST_CONV_LAYER = "post_relu"

print("ResNet50V2 loaded successfully")
print("Model path:", MODEL_PATH)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index=None
):

    # Create a model that returns:
    # 1. Feature maps from the last convolutional layer
    # 2. Final predictions

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        # Pass input as a list because the loaded Keras model
        # expects its input structure in this form.

        conv_outputs, predictions = grad_model(
            [img_array],
            training=False
        )

        if pred_index is None:
            pred_index = tf.argmax(
                predictions[0]
            )

        class_channel = predictions[:, pred_index]

    # --------------------------------------------------------
    # Gradient of predicted class
    # with respect to convolutional feature maps
    # --------------------------------------------------------

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # --------------------------------------------------------
    # Average gradients over height and width
    # --------------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # --------------------------------------------------------
    # Weight feature maps using gradients
    # --------------------------------------------------------

    heatmap = conv_outputs @ pooled_grads[0][..., tf.newaxis]

    heatmap = tf.squeeze(
        heatmap
    )

    # Keep only positive influence
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # Normalize heatmap
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


# ============================================================
# PREDICTION + GRAD-CAM
# ============================================================

def predict_with_gradcam(image_path):

    # ========================================================
    # 1. LOAD IMAGE
    # ========================================================

    original = cv2.imread(
        image_path
    )

    if original is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # Convert BGR → RGB
    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    # Image used for visualization
    original_display = cv2.resize(
        original,
        (224, 224)
    )

    # ========================================================
    # 2. PREPARE IMAGE FOR RESNET50V2
    # ========================================================

    img = cv2.resize(
        original,
        (224, 224)
    )

    img_array = img.astype(
        np.float32
    )

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ResNet50V2 preprocessing
    img_array = preprocess_input(
        img_array
    )

    # ========================================================
    # 3. MODEL PREDICTION
    # ========================================================

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_class = np.argmax(
        predictions
    )

    confidence = predictions[
        predicted_class
    ]

    # ========================================================
    # 4. CLASS NAME
    # ========================================================

    class_names = {
        0: "NORMAL",
        1: "PNEUMONIA"
    }

    predicted_label = class_names[
        predicted_class
    ]

    # ========================================================
    # 5. GENERATE GRAD-CAM
    # ========================================================

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        LAST_CONV_LAYER,
        pred_index=predicted_class
    )

    # ========================================================
    # 6. RESIZE HEATMAP
    # ========================================================

    heatmap_resized = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    # ========================================================
    # 7. COLOR HEATMAP
    # ========================================================

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # Convert BGR → RGB
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # 8. CREATE OVERLAY
    # ========================================================

    overlay = cv2.addWeighted(
        original_display,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    # ========================================================
    # 9. RETURN RESULTS
    # ========================================================

    return {
        "prediction": predicted_label,
        "confidence": float(confidence),

        # IMPORTANT:
        # Return the COLOR heatmap, not the grayscale heatmap.
        "heatmap": heatmap_color,

        "overlay": overlay
    }