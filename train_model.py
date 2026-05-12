import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import shutil
import numpy as np
from sklearn.utils import class_weight

# --- 1. SETUP & AUTO-FIX FOLDERS ---
dataset_root = "dataset"
healthy_folder_current = os.path.join(dataset_root, 'Healthy')
disease_folder = os.path.join(dataset_root, 'Disease')
healthy_folder_target = os.path.join(disease_folder, 'Healthy')

print(f"📂 Checking folder structure in: {dataset_root}...")

if os.path.exists(healthy_folder_current):
    print(f"   Found 'Healthy' folder outside. Moving it into 'Disease' folder...")
    try:
        shutil.move(healthy_folder_current, healthy_folder_target)
        print("✅ Moved successfully! Structure is now correct.")
    except Exception as e:
        print(f"⚠️ Could not move folder automatically: {e}")
        print("   Please manually drag the 'Healthy' folder inside the 'Disease' folder.")
elif os.path.exists(healthy_folder_target):
    print("✅ 'Healthy' folder is already inside 'Disease'. Good to go!")
else:
    print("❌ Error: Could not find 'Healthy' folder. Please check your folder names.")

# --- CONFIGURATION ---
DATASET_DIR = disease_folder 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
SEED = 42  # Added to ensure train/val splits don't mix!

# --- 2. DATA PREPARATION ---
# Training Generator (WITH Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 
)

# Validation Generator (NO Augmentation, clean images only)
val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

print(f"🚀 Loading images from: {DATASET_DIR}")

# Load Training Data (80%)
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

# Load Validation Data (20%)
val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

# --- CAPTURE THE LABELS ---
class_map = train_generator.class_indices
print(f"\n✅ The model will learn these classes: {list(class_map.keys())}")

# --- 3. BUILD THE MODEL ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(class_map), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- CALCULATE CLASS WEIGHTS ---
print("\n⚖️ Calculating class weights to handle imbalance...")
train_classes = train_generator.classes
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_classes),
    y=train_classes
)
class_weight_dict = dict(enumerate(weights))
print(f"Class Weights applied: {class_weight_dict}")

# --- 4. TRAIN ---
print("\n💪 Starting Training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight_dict  # Applied here
)

# --- 5. FINE TUNE (Boost Accuracy) ---
print("\n🔧 Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE/10),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=5,
    class_weight=class_weight_dict  # And applied here
)

# --- 6. SAVE ---
if not os.path.exists('models'):
    os.makedirs('models')
    
save_path = 'models/lemon_leaf_disease_model_accurate.h5'
model.save(save_path)
print(f"\n🎉 Model saved at: {save_path}")
print("----------------------------------------------------")
print("⚠️ IMPORTANT: UPDATE APP.PY WITH THESE LABELS:")
print(list(class_map.keys()))
print("----------------------------------------------------")