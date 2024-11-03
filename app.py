import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
from torchvision import models
import torch.nn as nn
import cupy as cp
from cupyx.scipy.ndimage import zoom as cpzoom, gaussian_filter as cpgaussian
from cupyx.scipy.ndimage import binary_dilation as cpdilation
import cv2
import numpy as np
import os

def get_pretrained_convnext():
    model = models.convnext.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    # Modify the classifier for binary classification
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)  # Binary classification output
    return model


class ConvNeXtWithAdditionalFeatures(nn.Module):
    def __init__(self, num_additional_features):
        super(ConvNeXtWithAdditionalFeatures, self).__init__()
        # Load the pre-trained ConvNeXt model
        self.convnext = models.convnext.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        # Replace the last classification layer with an identity layer to extract features
        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Identity()
        # Define a new classifier that takes concatenated features
        self.classifier = nn.Linear(num_features + num_additional_features, 10)  # Multilabel classification

    def forward(self, x, additional_features):
        # Pass input through ConvNeXt
        x = self.convnext(x)
        # Ensure additional_features has the correct shape
        if len(additional_features.shape) == 1:
            additional_features = additional_features.unsqueeze(1)
        # Concatenate ConvNeXt features with additional features
        x = torch.cat((x, additional_features), dim=1)
        # Pass concatenated features through the classifier
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

# Example usage:
def get_pretrained_convnext_with_additional_features(num_additional_features):
    model = ConvNeXtWithAdditionalFeatures(num_additional_features)
    return model
def process_image_cupy(image):
    # Convert image to CuPy array
    image_cp = cp.asarray(image)
    originalImage_cp = image_cp.copy()
    originalShape = image.shape

    # Resize image to 512x512
    image_cp = cpzoom(image_cp, (512 / image.shape[0], 512 / image.shape[1], 1))
    mask_cp = cp.zeros_like(image_cp)
    mask_cp[50:-50, 20:-20] = True
    image_cp = cp.where(mask_cp, image_cp, 0)

    blackMask_cp = cp.all(image_cp < 5, axis=-1)

    # Convert to grayscale (requires CPU computation)
    image_np = cp.asnumpy(image_cp)
    image_gray_np = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    image_gray_cp = cp.asarray(image_gray_np)

    for _ in range(30):
        blackMask_cp = cpdilation(blackMask_cp)
    image_cp = cpgaussian(image_gray_cp, sigma=3)
    grad = cp.array(cp.gradient(image_cp))
    grad_cp = cp.linalg.norm(grad, axis=0)
    image_cp = cp.where(blackMask_cp, 0, image_cp)
    grad_cp = cp.where(blackMask_cp, 0, grad_cp)

    # Find min and max locations (requires CPU computation)
    grad_image_np = cp.asnumpy(grad_cp * image_cp)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grad_image_np)
    scaled_loc_x = max_loc[0] * (originalShape[1] / 512)
    scaled_loc_y = max_loc[1] * (originalShape[0] / 512)
    newBound_height = 0.125 * originalShape[0]
    newBound_width = 0.125 * originalShape[1]

    xmin = max(0, int(scaled_loc_x - newBound_width))
    xmax = min(originalShape[1] - 1, int(scaled_loc_x + newBound_width))
    ymin = max(0, int(scaled_loc_y - newBound_height))
    ymax = min(originalShape[0] - 1, int(scaled_loc_y + newBound_height))
    crop_cp = originalImage_cp[ymin:ymax, xmin:xmax]

    # Convert to grayscale (requires CPU computation)
    crop_np = cp.asnumpy(crop_cp)
    crop_gray_np = cv2.cvtColor(crop_np.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    distribution = np.sum(crop_gray_np, axis=0) / 1000
    diff = np.sum(distribution[:len(distribution) // 2]) - np.sum(distribution[len(distribution) // 2:])
    #print(diff, np.sum(distribution[:len(distribution) // 2]),
    #      diff / np.sum(distribution[:len(distribution) // 2]))
    if (np.abs(diff) / np.sum(distribution[:len(distribution) // 2])) > 0.3:
        if diff > 0:
            scaled_loc_x -= 0.2 * len(distribution)
        else:
            scaled_loc_x += 0.2 * len(distribution)
    xmin = max(0, int(scaled_loc_x - newBound_width))
    xmax = min(originalShape[1] - 1, int(scaled_loc_x + newBound_width))
    ymin = max(0, int(scaled_loc_y - newBound_height))
    ymax = min(originalShape[0] - 1, int(scaled_loc_y + newBound_height))
    crop_cp = originalImage_cp[ymin:ymax, xmin:xmax]
    final_cp = cpzoom(crop_cp, (512 / crop_cp.shape[0], 512 / crop_cp.shape[1], 1))
    final_np = cp.asnumpy(final_cp)
    # Apply CLAHE
    # Convert to LAB color space
    lab = cv2.cvtColor(final_np, cv2.COLOR_BGR2LAB)
    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)
    # Merge the CLAHE enhanced L-channel back with a and b channels
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    # Convert back to BGR color space
    final_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return final_clahe


# Load models
glaucoma_model = get_pretrained_convnext().to('cuda:0')
glaucoma_model.load_state_dict({k.replace('_orig_mod.','') :v for k,v in torch.load('./outputs/best_model_binary.pth',weights_only=True).items()})
feature_model = get_pretrained_convnext_with_additional_features(5).to('cuda:0')
feature_model.load_state_dict(torch.load('./outputs/best_model_multi.pth',weights_only=True))

# Set models to evaluation mode
glaucoma_model.eval()
feature_model.eval()

# Define image transformations
transform = transforms.Compose([
    # transforms.Resize((512, 512)),
    # transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

feature_labels = [
    'Appearance neuroretinal rim superiorly (ANRS)', 'Appearance neuroretinal rim inferiorly (ANRI)',\
    'Retinal nerve fiber layer defect superiorly (RNFLDS)', 'Retinal nerve fiber layer defect inferiorly (RNFLDI)',\
    'Baring circumlinear vessel superiorly (BCLVS)', 'Baring circumlinear vessel inferiorly (BCLVI)', \
    'Nasalisation of vessel trunk (NVT)', 'Disc hemorrhages (DH)', 'Laminar dots (LD)', 'Large cup (LC)'
]

#st.title("InsEYEte")
st.markdown("<h1 style='text-align: center; color: white;'>InsEYEte</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])
import pickle
d = pickle.load(open('cup.pkl','rb'))

if uploaded_file is not None:
    print(uploaded_file)
    image = Image.open(uploaded_file)
    original = image.convert('RGB')
    image = process_image_cupy(np.array(image))
    st.image(original, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    input_tensor = transform(torch.tensor(np.transpose(image,(2,0,1))).float()).unsqueeze(0).to('cuda:0')

    # Glaucoma classification
    with torch.no_grad():
        glaucoma_output = glaucoma_model(input_tensor)
        glaucoma_prob = torch.sigmoid(glaucoma_output).item()

    if glaucoma_prob >= 0.5:
        # Prediction results
        st.subheader("Prediction Results")
        st.write(f"**Prediction:** Glaucoma Detected")
        st.write(f"**Confidence:** {glaucoma_prob:.2f}")

        with torch.no_grad():
            feature_output = feature_model(input_tensor,torch.tensor(d[uploaded_file.name.split('.')[0]]).unsqueeze(0).to('cuda:0'))
            feature_prob = torch.sigmoid(feature_output).squeeze()


        # Feature justifications
        with st.expander("Model's Feature Justifications"):
            for i in range(len(feature_labels)):
                st.write(f"- {feature_labels[i]}: {feature_prob[i].item():.2f}")

        # Editable justifications
        st.subheader("Edit Justifications")
        edited_features = {}
        for i,label in enumerate(feature_labels):
            edited_features[label] = st.checkbox(label, value=feature_prob[i] >= 0.02)

        # Final justifications
        st.subheader("Final Justifications")
        for label, selected in edited_features.items():
            if selected:
                st.write(f"- {label}")

    else:
        st.write("**Prediction:** No glaucoma detected.")
        st.write(f"**Confidence:** {1 - glaucoma_prob:.2f}")