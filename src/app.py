import streamlit as st
import tempfile
import os

from gradcam import predict_with_gradcam


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🫁 Pneumonia Detection")
st.write(
    "Upload a chest X-ray to classify it as NORMAL or PNEUMONIA "
    "using ResNet50V2 with Grad-CAM explainability."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        image_path = temp_file.name


    # Display uploaded image
    st.subheader("Uploaded X-ray")

    st.image(
        uploaded_file,
        width=400
    )


    # ========================================================
    # RUN MODEL
    # ========================================================

    if st.button(
        "🔍 Analyze X-ray",
        type="primary"
    ):

        with st.spinner(
            "Analyzing X-ray..."
        ):

            try:

                result = predict_with_gradcam(
                    image_path
                )


                # ====================================================
                # RESULTS
                # ====================================================

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.divider()

                st.subheader(
                    "Prediction Result"
                )


                # ====================================================
                # PREDICTION
                # ====================================================

                if prediction == "PNEUMONIA":

                    st.error(
                        f"Prediction: {prediction}"
                    )

                else:

                    st.success(
                        f"Prediction: {prediction}"
                    )


                # ====================================================
                # CONFIDENCE
                # ====================================================

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )


                # ====================================================
                # GRAD-CAM RESULTS
                # ====================================================

                st.divider()

                st.subheader(
                    "Grad-CAM Explainability"
                )

                col1, col2, col3 = st.columns(3)


                # Original image
                with col1:

                    st.markdown(
                        "**Original X-ray**"
                    )

                    st.image(
                        uploaded_file,
                        use_container_width=True
                    )


                # Heatmap
                with col2:
                      st.markdown("**Grad-CAM Heatmap**")
                      st.image(
                          result["heatmap"],
                          clamp=True,
                          use_container_width=True
                      )


                # Overlay
                with col3:

                    st.markdown(
                        "**Grad-CAM Overlay**"
                    )

                    st.image(
                        result["overlay"],
                        use_container_width=True
                    )


                # ====================================================
                # EXPLANATION
                # ====================================================

                st.divider()

                st.subheader(
                    "How to interpret the Grad-CAM"
                )

                st.write(
                    "The Grad-CAM highlights image regions that "
                    "contributed more strongly to the model's "
                    "prediction. Warmer regions indicate stronger "
                    "activation for the predicted class."
                )


                # ====================================================
                # DISCLAIMER
                # ====================================================

                st.warning(
                    "This system is a machine-learning prototype "
                    "for educational and research purposes and "
                    "should not be used as a substitute for "
                    "professional medical diagnosis."
                )


            except Exception as e:

                st.error(
                    f"Error while analyzing image: {e}"
                )


            finally:

                # Remove temporary file
                if os.path.exists(image_path):

                    os.remove(
                        image_path
                    )
                    