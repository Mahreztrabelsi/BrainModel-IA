import streamlit as st
from PIL import Image
import io
import torch
from torchvision import transforms
from src.model import load_model
from src.utils import predict
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import base64

# Use the non-interactive Agg backend for matplotlib
matplotlib.use('Agg')

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/model_38"
model = load_model(model_path, device)

# Define the image transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]
)

label_dict = {
    0: "No Tumor",
    1: "Pituitary",
    2: "Glioma",
    3: "Meningioma",
    4: "Other",
}

stage_info = {
    0: {"stage": "Stage 0",  "info": "The tumor is localized and small, with no invasion of nearby tissue.", "recommendation": "Observation or surgical removal is usually recommended."},
    1: {"stage": "Stage 1", "info": "The tumor is slightly larger, but still confined to the region of origin.", "recommendation": "Treatment may include surgery or localized radiation."},
    2: {"stage": "Stage 2", "info": "The tumor is larger or may have started spreading to nearby tissues.", "recommendation": "Radiation and/or chemotherapy may be necessary along with surgery."},
    3: {"stage": "Stage 3", "info": "The tumor has spread to nearby lymph nodes or other tissues.", "recommendation": "Aggressive treatment including surgery, radiation, and chemotherapy is often required."},
    4: {"stage": "Stage 4", "info": "The tumor has metastasized to distant parts of the body.", "recommendation": "Palliative care, systemic treatments such as chemotherapy or immunotherapy, may be recommended."}
}

# Streamlit UI
st.title("Tumor Type Prediction and Staging")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = transform(image).unsqueeze(0).to(device)

    # Get predicted probabilities
    predicted_probs = predict(model, preprocessed_image, device, return_probs=True)

    # Get the predicted class and label
    predicted_class = torch.argmax(predicted_probs).item()
    label = label_dict.get(predicted_class, "Unknown")

    # Convert probabilities to a list
    probs_list = predicted_probs.squeeze().cpu().numpy().tolist()

    # Stage prediction logic (replace with actual logic)
    predicted_stage = np.random.randint(0, 5)  # Replace with actual stage prediction
    stage_details = stage_info.get(predicted_stage, {
        "stage": "Unknown",
        "info": "Stage information not available.",
        "recommendation": "No recommendation available."
    })


    # Generate enhanced scatter plot for tumor type probabilities
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    # Scatter plot for each tumor type probability
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']  # Define distinct colors for each class
    for i, (tumor_type, prob) in enumerate(zip(label_dict.values(), probs_list)):
        if i == predicted_class:
            # Highlight the predicted class with a different color and marker style
            ax1.scatter(tumor_type, prob, color='red', s=150, marker='o', label=f"Predicted: {tumor_type} ({prob:.2f})", zorder=5)
        else:
            # Regular points for other classes
            ax1.scatter(tumor_type, prob, color=colors[i], s=100, marker='x', label=f"{tumor_type}: {prob:.2f}")

    # Draw a horizontal line for the predicted class probability
    predicted_probability = probs_list[predicted_class]
    ax1.axhline(predicted_probability, color='red', linestyle='--', linewidth=1, label=f"Prediction Line ({label}: {predicted_probability:.2f})")

    # Add labels and title
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Tumor Type')
    ax1.set_title('Tumor Type Probability Scatter Plot with Prediction Line')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move the legend outside of the plot for clarity
    plt.tight_layout()

    # Generate scatter plot for tumor type probabilities
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(label_dict.values(), probs_list, color='blue', label="Probability Points")
    ax.set_ylabel('Probability')
    ax.set_xlabel('Tumor Type')
    ax.set_title('Tumor Type Probability Scatter Plot')
    ax.legend()
    plt.tight_layout()

    



    # Display results
    st.write("### Prediction Result")
    st.write("Tumor Type:", label)
    if label == "No Tumor":
        st.write("Stage Information:", "There is no Stage for tumor type: No Tumor")
        st.write("Treatment Recommendation:", "There is no Recommendations for tumor type: No Tumor")
    else:
        st.write("Stage Prediction:", stage_details['stage'])
        st.write("Stage Information:", stage_details['info'])
        st.write("Treatment Recommendation:", stage_details['recommendation'])
        st.pyplot(fig)  # Display scatter plot directly
        # Generate scatter plot for tumor type probabilities

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(label_dict.values(), probs_list, color='blue', label="Probability Points")

        # Draw a horizontal line for the predicted class probability
        predicted_probability = probs_list[predicted_class]
        ax.axhline(predicted_probability, color='red', linestyle='--', label=f"Prediction Line: {label} ({predicted_probability:.2f})")

        # Highlight the predicted class point
        ax.scatter(label_dict[predicted_class], predicted_probability, color='red', s=100, label="Predicted Class", zorder=5)

        # Labels and title
        ax.set_ylabel('Probability')
        ax.set_xlabel('Tumor Type')
        ax.set_title('Tumor Type Probability Scatter Plot with Prediction Line')
        ax.legend()
        plt.tight_layout()
        # Display scatter plot with prediction line in Streamlit
        st.pyplot(fig)
        
        st.pyplot(fig1)  # Display scatter plot directly
