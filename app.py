import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import tempfile
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo
# import requests
from streamlit_gsheets import GSheetsConnection

import gc

# transfer learning

# option 1

# ==== ADD THESE NEW IMPORTS ====
# from tensorflow.keras.applications import (
#     efficientnet,
#     efficientnet_v2,
#     mobilenet,
#     mobilenet_v2,
#     resnet,
#     resnet50,
#     resnet_v2,
#     vgg16,
#     vgg19,
#     xception,
#     inception_v3,
#     densenet,
#     nasnet
# )

# # Create a dictionary of these modules to help load_model find them
# CUSTOM_OBJECTS = {
#     # Module names
#     "efficientnet": efficientnet,
#     "efficientnet_v2": efficientnet_v2,
#     "mobilenet": mobilenet,
#     "mobilenet_v2": mobilenet_v2,
#     "resnet": resnet,
#     "resnet50": resnet50,
#     "resnet_v2": resnet_v2,
#     "vgg16": vgg16,
#     "vgg19": vgg19,
#     "xception": xception,
#     "inception_v3": inception_v3,
#     "densenet": densenet,
#     "nasnet": nasnet}

#option 2
# Create a mapping of "UI Name" to "Keras Module Name"
# This list covers the most common transfer learning models
MODEL_OPTIONS = {
    "No Transfer Learning (Custom Model)": None,
    "EfficientNet (B0-B7)": "efficientnet",
    "EfficientNet V2": "efficientnet_v2",
    "MobileNet": "mobilenet",
    "MobileNet V2": "mobilenet_v2",
    "ResNet50": "resnet50",
    "ResNet (Other)": "resnet",
    "ResNet V2": "resnet_v2",
    "VGG16": "vgg16",
    "VGG19": "vgg19",
    "Xception": "xception",
    "Inception V3": "inception_v3",
    "DenseNet": "densenet",
    "NASNet": "nasnet"
}

st.write("### Model Configuration")
selected_model_label = st.selectbox("Which base model did you use?", options=list(MODEL_OPTIONS.keys()))
selected_module_name = MODEL_OPTIONS[selected_model_label]

# ==== CONFIGURATION ====
TEST_IMAGE_DIR = "test_images"
LEADERBOARD_FILE = "leaderboard.csv"
CLASS_NAMES = ["A", "B", "C"]

# jsonbin.io
# BIN_ID = "YOUR_BIN_ID_HERE"
# API_KEY = "YOUR_MASTER_KEY_HERE"
# BASE_URL = f"https://api.jsonbin.io/v3/b/{BIN_ID}"
# HEADERS = {
#     "Content-Type": "application/json",
#     "X-Master-Key": API_KEY
# }

# ==== LOAD ALL TEST IMAGES (do NOT resize here!) ====
# @st.cache_data(show_spinner="Loading hidden test set…")
# def load_raw_test_images():
#     images, labels = [], []
#     for idx, cls in enumerate(CLASS_NAMES):
#         folder = os.path.join(TEST_IMAGE_DIR, cls)
#         for fname in sorted(os.listdir(folder)):
#             fpath = os.path.join(folder, fname)
#             img = Image.open(fpath).convert("RGB")
#             images.append(img)
#             labels.append(idx)
#     return images, np.array(labels)

# alternatively
# Stop Caching Images (Cache Paths Instead)
# Replace your load_raw_test_images function with this. We only store file paths (strings),
# which take almost zero RAM.
@st.cache_data(show_spinner="Loading hidden test set…")
def get_test_image_paths():
    """Load paths only. Keeps memory footprint tiny."""
    paths, labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(TEST_IMAGE_DIR, cls)
        # Ensure folder exists to avoid crashes
        if os.path.isdir(folder):
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(folder, fname))
                    labels.append(idx)
    return paths, np.array(labels)

# gsheets
def load_leaderboard():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        # ttl=0 prevents caching old data
        df = conn.read(worksheet="Sheet1", ttl=0)
        if df.empty or "Username" not in df.columns:
             return pd.DataFrame(columns=["Username", "Accuracy", "Timestamp"])
        return df
    except:
        return pd.DataFrame(columns=["Username", "Accuracy", "Timestamp"])

# gsheets
def save_leaderboard(df):
    conn = st.connection("gsheets", type=GSheetsConnection)
    st.write('conneciton established')
    conn.update(worksheet="Sheet1", data=df)

# def load_leaderboard():
#     if os.path.exists(LEADERBOARD_FILE):
#         return pd.read_csv(LEADERBOARD_FILE)
#     else:
#         return pd.DataFrame(columns=["Username", "Accuracy", "Timestamp"])

# def save_leaderboard(df):
#     df.to_csv(LEADERBOARD_FILE, index=False)

# jsonbin.io
# def load_leaderboard():
#     try:
#         response = requests.get(BASE_URL, headers=HEADERS)
#         if response.status_code == 200:
#             data = response.json().get("record", [])
#             if not data: # If empty list
#                 return pd.DataFrame(columns=["Username", "Accuracy", "Timestamp"])
#             return pd.DataFrame(data)
#         else:
#             st.error(f"Error loading leaderboard: {response.status_code}")
#             return pd.DataFrame(columns=["Username", "Accuracy", "Timestamp"])
#     except Exception as e:
#         st.error(f"Connection error: {e}")
#         return pd.DataFrame(columns=["Username", "Accuracy", "Timestamp"])

# jsonbin.io
# def save_leaderboard(df):
#     # Convert dataframe to a list of dicts (JSON format)
#     json_data = df.to_dict(orient="records")

#     response = requests.put(BASE_URL, json=json_data, headers=HEADERS)

#     if response.status_code != 200:
#         st.error("Failed to save to leaderboard! Please contact admin.")

# def evaluate_model(model, pil_images, y, input_size):
#     resized_imgs = [np.array(img.resize(input_size)) / 255.0 for img in pil_images]
#     x = np.stack(resized_imgs)
#     preds = model.predict(x)
#     y_pred = np.argmax(preds, axis=1)
#     acc = (y_pred == y).mean()
#     return acc

# alternatively
# Memory-Safe Evaluation Loop: processes images in chunks of 32. It never holds the whole dataset in RAM.

# 1 predict_on_batch vs predict: predict tries to build the whole result array in memory.
# predict_on_batch runs one step and returns.

# 2 tf.keras.backend.clear_session():
# This is the magic command. Without it, TensorFlow keeps the computational graph of every model ever
# uploaded in the background until the app restarts. This clears the slate.

# 3 del + gc.collect(): Python doesn't always delete variables immediately when they go out of scope.
# In a constrained environment (Streamlit Free Tier), forcing gc.collect() inside the loop ensures the memory from Batch 1 is actually freed before Batch 2 starts.




        # Load model freshly
        # model = tf.keras.models.load_model(model_path)

        # Transfer Learning

        # option 1
        # when fine-tuning the model:
        # ✅ DO THIS:
        # from tensorflow.keras.applications import efficientnet
        # ...
        # x = tf.keras.layers.Lambda(efficientnet.preprocess_input)(x)
        # ❌ DON'T DO THIS (The app won't know where 'preprocess_input' comes from):
        # from tensorflow.keras.applications.efficientnet import preprocess_input
        # ...
        # x = tf.keras.layers.Lambda(preprocess_input)(x)

        # By passing custom_objects, you are manually injecting these modules into the scope where the model is being loaded.
        # Even if the student's local environment path was slightly different, mapping "efficientnet": efficientnet bridges the gap.

def evaluate_model_efficiently(model_path, image_paths, y_true, custom_objects=None, preprocessor=None):
    # 1. Clear session
    tf.keras.backend.clear_session()
    gc.collect()

    try:
        # Load model (compile=False is safer for inference)
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )

        input_shape = model.input_shape
        # Check shape (assuming standard NHWC format)
        if len(input_shape) != 4:
             raise ValueError("Model must have 4 dimensions (None, H, W, C).")

        h, w = input_shape[1], input_shape[2]

        # 2. Batch Processing
        BATCH_SIZE = 32
        total_correct = 0
        num_samples = len(image_paths)

        progress_bar = st.progress(0, text="Processing test images...")

        for start_idx in range(0, num_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_samples)
            batch_paths = image_paths[start_idx:end_idx]
            batch_y = y_true[start_idx:end_idx]

            batch_images = []
            for p in batch_paths:
                with Image.open(p) as img:
                    img_resized = img.convert("RGB").resize((w, h))
                    batch_images.append(np.array(img_resized))

            # Convert to float32 numpy array
            x_batch = np.array(batch_images, dtype=np.float32)

            # ✅ CRITICAL FIX: Apply the specific preprocessor here!
            if preprocessor:
                x_batch = preprocessor(x_batch)
            else:
                # If "No Transfer Learning" selected, assume standard 0-1 scaling
                # (Most custom student models are trained on 0-1 data)
                x_batch = x_batch / 255.0

            preds = model.predict_on_batch(x_batch)
            y_pred = np.argmax(preds, axis=1)
            total_correct += np.sum(y_pred == batch_y)

            progress_bar.progress(end_idx / num_samples)

            del x_batch, batch_images, preds
            gc.collect()

        accuracy = total_correct / num_samples
        return accuracy

    except Exception as e:
        raise e
    finally:
        if 'model' in locals():
            del model
        tf.keras.backend.clear_session()
        gc.collect()


st.title("Sign Language Model Showdown!")
st.markdown("")

st.markdown(
    "Can your Keras model tell the difference between A, B, and C? ✊🖐️🤏 Time to put it to the test!"
)
st.markdown("")

st.markdown(
    "Upload your trained `.keras` model, and we’ll run it on our secret set of sign language photos. Once your model's evaluated, your score pops up on the leaderboard. Top the table, and those bragging rights are all yours! 🏆\n\n"
    "You can use almost any image size: 64×64, 128×128, 224×224, 256×256.Just make sure your model expects standard 3-channel (RGB) colour images."
)
st.markdown("")

col1, col2, col3 = st.columns([1, 12, 1])
with col2:
    st.markdown("👇 **Fill in your username, upload your model, and join the leaderboard. Good luck!** 👇")


username = st.text_input("Enter your username:")
uploaded_file = st.file_uploader(
    "Upload your Keras model (.keras only)", type=["keras"], accept_multiple_files=False
)
st.markdown("")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit = st.button("Submit model for evaluation", type="primary")

leaderboard = load_leaderboard()
# raw_images, y_test = load_raw_test_images()
# here
test_paths, y_test = get_test_image_paths()

if submit and uploaded_file and username.strip():
    with st.spinner("Evaluating your model..."):

        # 1. Determine which preprocessor to use
        selected_preprocessor = None
        custom_objects = {}

        if selected_module_name:
            try:
                # Import module dynamically (e.g. tensorflow.keras.applications.mobilenet)
                app_module = getattr(tf.keras.applications, selected_module_name)

                # Get the function (e.g. mobilenet.preprocess_input)
                selected_preprocessor = app_module.preprocess_input

                # Still strictly necessary to help load the model if they DID use Lambda layers
                # (It doesn't hurt to have it even if they didn't)
                custom_objects["preprocess_input"] = selected_preprocessor
                custom_objects[selected_module_name] = app_module

            except AttributeError:
                st.error(f"Could not find module {selected_module_name}")
                st.stop()


        with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmpf:
            tmpf.write(uploaded_file.read())
            tmpf.flush()
            try:
                # model = tf.keras.models.load_model(tmpf.name)
                model = tf.keras.models.load_model(tmpf.name, custom_objects=custom_objects)

                input_shape = model.input_shape
                # Accept models with shape (None, H, W, 3)
                if len(input_shape) == 4 and input_shape[-1] == 3:
                    input_size = (input_shape[1], input_shape[2])  # (height, width)
                    if None in input_size:
                        st.error("Model input shape must have concrete dimensions, not None. Please specify fixed input size in your model.")
                        model = None
                    if model is not None:
                        try:
                            # acc = evaluate_model(model, raw_images, y_test, input_size)
                            # acc = evaluate_model_efficiently(tmpf.name, test_paths, y_test)

                            # Pass the 'selected_preprocessor' to the function
                            acc = evaluate_model_efficiently(
                                tmpf.name,
                                test_paths,
                                y_test,
                                custom_objects=custom_objects,
                                preprocessor=selected_preprocessor)  # <--- NEW ARGUMENT)

                            timestamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S")
                            new_row = {
                                "Username": username,
                                "Accuracy": round(acc * 100, 2),
                                "Timestamp": timestamp,
                            }
                            leaderboard = pd.concat(
                                [leaderboard, pd.DataFrame([new_row])], ignore_index=True
                            )
                            save_leaderboard(leaderboard)
                            st.success(
                                f"🎉 All done! Your model scored {acc:.2%} on our test set. Your result has been added to the leaderboard."
                            )
                        except Exception as e:
                            st.error(f"Model could not be run on the test set: {e}")
                else:
                    st.error(
                        f"Model input shape {input_shape} is not supported. "
                        "Model must accept 3-channel (RGB) images."
                    )
            except Exception as e:
                st.error(
                    "Could not load your model file. "
                    "Please ensure it is a valid Keras `.keras` file. "
                    f"Error: {e}"
                )

elif submit and not uploaded_file:
    st.warning("Please upload your Keras `.keras` model file before submitting.")
elif submit and not username.strip():
    st.warning("Please enter a username before submitting.")

st.header("Leaderboard")
if leaderboard.empty:
    st.write("No submissions yet.")
else:
    leaderboard_display = leaderboard.copy()
    leaderboard_display["Accuracy"] = leaderboard_display["Accuracy"].astype(str) + " %"
    st.table(leaderboard_display.sort_values("Accuracy", ascending=False).reset_index(drop=True))

st.divider()

st.markdown(
    "You can submit as many models as you like. Each submission will appear as a new row in the leaderboard. Your uploaded model file is deleted after evaluation. "
)
