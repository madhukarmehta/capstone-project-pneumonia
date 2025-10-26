# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from huggingface_hub import hf_hub_download

# Define constants for the dataset and output paths
repo_id = "madhukarmehta/capstone-project-pneumonia"
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH_NPY = "hf://datasets/madhukarmehta/capstone-project-pneumonia/pneu_image_cut.npy"
DATASET_PATH_LABEL = "hf://datasets/madhukarmehta/capstone-project-pneumonia/train_label.csv"
lable_df = pd.read_csv(DATASET_PATH_LABEL)

local_npy_path = hf_hub_download(
    repo_id=repo_id,
    filename="pneu_image_cut.npy",
    repo_type="dataset",          # <-- crucial line
    token=os.getenv("HF_TOKEN")   # optional if private
)

gray_images = np.load(local_npy_path)

print("Dataset loaded successfully.")

X_train, X_test, y_train, y_test = train_test_split(gray_images,lable_df['Target'],test_size=0.2, random_state=42,stratify=lable_df['Target'])


np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

files = ["X_train.npy","X_test.npy","y_train.npy","y_test.npy"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="madhukarmehta/capstone-project-pneumonia",
        repo_type="dataset",
    )
