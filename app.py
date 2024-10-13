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
    0: {"stage": "Stage 0", "info": "The tumor is localized and small, with no invasion of nearby tissue.", "recommendation": "Observation or surgical removal is usually recommended."},
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

    # Generate histogram for tumor type
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(label_dict.values(), probs_list)
    ax.set_ylabel('Probability')
    ax.set_xlabel('Tumor Type')
    ax.set_title('Tumor Type Probabilities')
    plt.tight_layout()

    # Save histogram to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Encode the image in base64 to display in Streamlit
    histogram_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Display results
    st.write("### Prediction Result")
    st.write("Tumor Type:", label)
    if (label == "No Tumor"):
            st.write("Stage Information:", "There is no Stage for  tumor type: No Tumor ")
            st.write("Treatment Recommendation:", "There is no Recommendations for tumor type: No Tumor ")
            
    st.write("Stage Prediction:", stage_details['stage'])
    st.write("Stage Information:", stage_details['info'])
    st.write("Treatment Recommendation:", stage_details['recommendation'])
    st.image("data:image/png;base64," + histogram_base64, caption='Tumor Type Probabilities Histogram', use_column_width=True)

