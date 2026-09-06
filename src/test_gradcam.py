from gradcam import predict_with_gradcam


IMAGE_PATH = r"D:\DS_genAI\Projects\Pneumonia_Detection\examples\gradcam\correct_pneumonia\case_3.png"


result = predict_with_gradcam(IMAGE_PATH)


print("\n==============================")
print("PREDICTION RESULT")
print("==============================")
print("Prediction :", result["prediction"])
print("Confidence :", f'{result["confidence"] * 100:.2f}%')