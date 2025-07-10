import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random
# Dataset Visualization  here in Thi function Data is Plotted and visuvalized 
def data_visual():
    fig = plt.figure(figsize=(15,15))

    image_folder = Path("/Users/pallavisumakurmala/Downloads/archive")

    for index, label in enumerate(os.listdir(image_folder)):
        print(index, label)
        if label == ".DS_Store":
            continue
        else:
            if label == "Poses.json":
                continue

            image_list = os.listdir(os.path.join(image_folder,label))
            if len(image_list) == 0:
                os.remove(os.path.join(image_folder,label))


            img = random.choice(image_list)
            img = Image.open(os.path.join(image_folder,label,img))

            fig.add_subplot(10,5,index+1)
            plt.imshow(img)
            plt.title(label, fontsize=10)
            plt.axis("off")
    plt.show()

# Thunder Model Loading and Extracting the keyPoints in this function 
def load_model_and_extract_keypoints():
    # Load the MoveNet model
    try:
        movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        model = movenet.signatures["serving_default"]
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Define the path to your dataset
    image_folder = "/Users/pallavisumakurmala/Downloads/archive"

    def extract_keypoints(image_path):
        try:
            # Open image
            image = Image.open(image_path)
            
            # Handle different image modes
            if image.mode == "P":  # Palette mode
                image = image.convert("RGBA")  # Convert to RGBA first
                image = image.convert("RGB")   # Then to RGB for MoveNet
            elif image.mode != "RGB":      # Any other non-RGB mode
                image = image.convert("RGB")
            
            # Convert to numpy array with correct dtype
            image = np.array(image, dtype=np.uint8)
            
            # Convert to tensor and add batch dimension
            image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
            image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, H, W, 3)
            
            # Resize and pad to 192x192 - keep as uint8
            input_image = tf.image.resize_with_pad(image_tensor, 192, 192)  # Shape: (1, 192, 192, 3)
            
            # Ensure the tensor is int32 as expected by MoveNet
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            # Debug: Print input shape
            print(f"Input shape for {image_path}: {input_image.shape}")
            
            # Run inference
            outputs = model(input_image)
            raw_keypoints = outputs["output_0"].numpy()  # Expected shape: (1, 17, 3) or (1, 1, 17, 3)
            print(f"Raw keypoints shape for {image_path}: {raw_keypoints.shape}")
            
            # Remove batch dimensions
            # Handle both (1, 17, 3) and (1, 1, 17, 3) cases
            if len(raw_keypoints.shape) == 4:  # Shape: (1, 1, 17, 3)
                if raw_keypoints.shape[0] != 1 or raw_keypoints.shape[1] != 1:
                    print(f"Unexpected batch dimensions for {image_path}: {raw_keypoints.shape}")
                    return None
                keypoints = raw_keypoints[0, 0]  # Shape: (17, 3)
            elif len(raw_keypoints.shape) == 3:  # Shape: (1, 17, 3)
                if raw_keypoints.shape[0] != 1:
                    print(f"Unexpected batch size for {image_path}: {raw_keypoints.shape[0]}")
                    return None
                keypoints = raw_keypoints[0]  # Shape: (17, 3)
            else:
                print(f"Unexpected raw keypoints shape for {image_path}: {raw_keypoints.shape}")
                return None
            
            # Debug: Print shape after batch removal
            print(f"Keypoints shape after batch removal for {image_path}: {keypoints.shape}")
            
            # Verify keypoint shape
            if keypoints.shape != (17, 3):
                print(f"Unexpected keypoints shape for {image_path}: {keypoints.shape}")
                return None
                
            # Flatten x, y coordinates (17 keypoints * 2 = 34 values)
            flattened_keypoints = keypoints[:, :2].flatten()  # Only take x, y, discard confidence
            print(f"Flattened keypoints length for {image_path}: {len(flattened_keypoints)}")
            if len(flattened_keypoints) != 34:
                print(f"Unexpected number of keypoints for {image_path}: {len(flattened_keypoints)}")
                return None
                
            return flattened_keypoints
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    # List to store all data
    data = []
    
    # Process each asana folder
    for label in os.listdir(image_folder):
        if label in [".DS_Store", "Poses.json"]:
            continue
            
        image_dir = os.path.join(image_folder, label)
        if not os.path.isdir(image_dir):
            continue
            
        print(f"Processing asana: {label}")
        for img in os.listdir(image_dir):
            if img.startswith('.'):
                continue
                
            image_path = os.path.join(image_dir, img)
            keypoints = extract_keypoints(image_path)
            if keypoints is not None:
                row = list(keypoints) + [image_path, label]
                if len(row) != 36:
                    print(f"Skipping {image_path}: row has {len(row)} elements, expected 36")
                    continue
                data.append(row)
                print(f"Processed {image_path}: {len(keypoints)} keypoints")

    if not data:
        print("No valid data extracted! Check your images or model.")
        return None
        
    # Define columns for the DataFrame
    columns = [f"kp_{i}" for i in range(34)] + ["image_path", "label"]
    
    # Create and save DataFrame
    try:
        df = pd.DataFrame(data, columns=columns)
        df.to_csv("yoga_detection_dataset.csv", index=False)
        print(f"✅ Dataset saved successfully with {len(df)} samples!")
    except Exception as e:
        print(f"Error creating/saving DataFrame: {e}")
        return None
    
    return df

#load the dataset and train the Model 

if __name__ == "__main__":
    