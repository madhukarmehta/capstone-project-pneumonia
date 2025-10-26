# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH_NPY = "hf://datasets/madhukarmehta/capstone-project-pneumonia/pneu_image_cut.npy"
DATASET_PATH_LABEL = "hf://datasets/madhukarmehta/capstone-project-pneumonia/train_label.csv"
gray_images = pd.read_csv(DATASET_PATH_NPY)
lable_df = pd.read_csv(DATASET_PATH_LABEL)
print("Dataset loaded successfully.")

X_train, X_val, y_train, y_val = train_test_split(gray_images,lable_df['Target'],test_size=0.2, random_state=42,stratify=lable_df['Target'])



# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train.to_csv("Xtrain.csv",index=False)
X_val.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_val.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="madhukarmehta/capstone-project-pneumonia",
        repo_type="dataset",
    )
