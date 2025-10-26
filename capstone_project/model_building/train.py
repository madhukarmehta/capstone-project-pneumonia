# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:6000")
mlflow.set_experiment("capstone-experiment")

api = HfApi()

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

Xtrain_path = "hf://datasets/madhukarmehta/capstone-project-pneumonia/X_train.npy"
Xtest_path = "hf://datasets/madhukarmehta/capstone-project-pneumonia/X_test.npy"
ytrain_path = "hf://datasets/madhukarmehta/capstone-project-pneumonia/y_train.npy"
ytest_path = "hf://datasets/madhukarmehta/capstone-project-pneumonia/y_test.npy"

Xtrain = np.load(Xtrain_path)
Xtest = np.load(Xtest_path)
ytrain = np.load(ytrain_path)
ytest = np.load(ytest_path)


from sklearn.preprocessing import LabelBinarizer
enc = LabelBinarizer()
y_train_encoded = enc.fit_transform(ytrain)
y_test_encoded=enc.transform(ytest)

# Intializing a sequential model
model = Sequential()

# Adding first conv layer with 64 filters and kernel size 3x3 , padding 'same' provides the output size same as the input size
# Input_shape denotes input image dimension of images
model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(224, 224, 1)))

# Adding max pooling to reduce the size of output of first conv layer
model.add(MaxPooling2D((2, 2), padding = 'same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D((2, 2), padding = 'same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D((2, 2), padding = 'same'))

# flattening the output of the conv layer after max pooling to make it ready for creating dense connections
model.add(Flatten())

# Adding a fully connected dense layer with 100 neurons    
model.add(Dense(100, activation='relu'))

# Adding the output layer with 1 neurons and activation functions as sigmoid since this is a binary classification problem  
model.add(Dense(1, activation='sigmoid'))

# Using Adam Optimizer
opt = Adam()

# Compile model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Generating the summary of the model
model.summary()


# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    history_1 = model.fit(
            Xtrain, y_train_encoded,
            epochs=15,
            validation_split=0.1,
            shuffle=True,
            batch_size=64,
            verbose=2   
    )

    # Save the model locally
    model_path = "best_pneumonia_prediction_model_v1.joblib"
    joblib.dump(model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "madhukarmehta/capstone-project-pneumonia"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_pneumonia_prediction_model_v1.joblib",
        path_in_repo="best_pneumonia_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
