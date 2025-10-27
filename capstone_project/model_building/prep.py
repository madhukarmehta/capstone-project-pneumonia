# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
import cv2, pydicom
from tqdm import tqdm
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download 

# Define constants for the dataset and output paths
repo_id = "madhukarmehta/capstone-project-pneumonia"
api = HfApi(token=os.getenv("HF_TOKEN"))


# Defines input and output directories. Creates the output directory if it doesn't exist.
DICOM_DIR = "hf://datasets/madhukarmehta/capstone-project-pneumonia/train_dicom/"
PNG_DIR   = "hf://datasets/madhukarmehta/capstone-project-pneumonia/train_png"
os.makedirs(PNG_DIR, exist_ok=True)

# Defines a function to convert a single DICOM file to a resized PNG image.
def dicom_to_png(dicom_path, png_path, size=(224,224)):

  # Reads the DICOM file and extracts the pixel array as a float32 image.
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)

    # Checks if the image is inverted (MONOCHROME1) and corrects it by flipping pixel intensities.
    if str(getattr(ds, 'PhotometricInterpretation','')).upper() == 'MONOCHROME1':
        img = img.max() - img

    # Normalize to [0,255]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    # Resize for consistency
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # Save as PNG (grayscale, 1 channel)
    cv2.imwrite(png_path, img)

# Iterates through all .dcm files in the input directory, converts each to PNG, and saves it in the output directory with the same base name.

# Download dataset locally (returns a local path)
local_dir = snapshot_download(
    repo_id="madhukarmehta/capstone-project-pneumonia",
    repo_type="dataset"
)

# Construct the real local path
DICOM_DIR = os.path.join(local_dir, "train_dicom")

if not os.path.exists(DICOM_DIR):
    raise FileNotFoundError(f"DICOM directory not found at {DICOM_DIR}")

for fname in tqdm(os.listdir(DICOM_DIR)):
    if fname.endswith(".dcm"):
        dicom_path = os.path.join(DICOM_DIR, fname)
        png_path   = os.path.join(PNG_DIR, fname.replace(".dcm",".png"))
        dicom_to_png(dicom_path, png_path)

# Prints a completion message and lists a few sample PNG filenames for verification.
print("Conversion complete. PNGs saved in:", PNG_DIR)
print("Sample PNG files:", os.listdir(PNG_DIR)[:5])


#DATASET_PATH_NPY = "hf://datasets/madhukarmehta/capstone-project-pneumonia/pneu_image_cut.npy"
#DATASET_PATH_LABEL = "hf://datasets/madhukarmehta/capstone-project-pneumonia/train_label.csv"
#lable_df = pd.read_csv(DATASET_PATH_LABEL)

#local_npy_path = hf_hub_download(
#    repo_id=repo_id,
#    filename="pneu_image_cut.npy",
#    repo_type="dataset",          # <-- crucial line
#    token=os.getenv("HF_TOKEN")   # optional if private
#)

#gray_images = np.load(local_npy_path)

#print("Dataset loaded successfully.")

#X_train, X_test, y_train, y_test = train_test_split(gray_images,lable_df['Target'],test_size=0.2, random_state=42,stratify=lable_df['Target'])


#np.save("X_train.npy", X_train)
#np.save("X_test.npy", X_test)
#np.save("y_train.npy", y_train)
#np.save("y_test.npy", y_test)

#files = ["X_train.npy","X_test.npy","y_train.npy","y_test.npy"]

#for file_path in files:
#    api.upload_file(
#        path_or_fileobj=file_path,
#        path_in_repo=file_path.split("/")[-1],  # just the filename
#        repo_id="madhukarmehta/capstone-project-pneumonia",
#       repo_type="dataset",
#    )
