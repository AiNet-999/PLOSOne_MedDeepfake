import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_curve, roc_auc_score, confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape,
    LeakyReLU, GlobalAveragePooling2D, Multiply, Layer, MultiHeadAttention,
    LayerNormalization, BatchNormalization, Add, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger

warnings.filterwarnings('ignore')


#https://github.com/kobiso/CBAM-keras
#https://github.com/titu1994/keras-squeeze-excite-network

class VGGBlock(Layer):
    def __init__(self, num_convs, filters, **kwargs):
        super(VGGBlock, self).__init__(**kwargs)
        self.num_convs = num_convs
        self.filters = filters
        self.convs = [Conv2D(filters, kernel_size=3, padding='same', activation='relu') for _ in range(num_convs)]
        self.pool = MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        return self.pool(x) 




class AttentionModule(Layer):
    def __init__(self, num_heads):
        super(AttentionModule, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=256)
        self.layer_norm = LayerNormalization()

    def call(self, patches):
        attn_output = self.attention(patches, patches)
        return self.layer_norm(attn_output + patches)  

class SEBlock(Layer):
    def __init__(self, ratio=16):
        super(SEBlock, self).__init__()
        self.ratio = ratio
        self.global_avg_pool = GlobalAveragePooling2D()
        self.fc1 = None  # Will initialize later
        self.fc2 = None

    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = Dense(channels // self.ratio, activation='relu')
        self.fc2 = Dense(channels, activation='sigmoid')
        self.reshape = Reshape((1, 1, channels))

    def call(self, inputs):
        se = self.global_avg_pool(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        return Multiply()([inputs, se])


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        self.fc1 = Dense(channel_dim // self.reduction_ratio, activation='relu')
        self.fc2 = Dense(channel_dim, activation='sigmoid')

    def call(self, inputs):
        se = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        se = self.fc1(se)
        se = self.fc2(se)

     
        se = Reshape((1, 1, K.int_shape(inputs)[-1]))(se)

      
        return Multiply()([inputs, se])

def load_and_preprocess_images(directory, label, img_rows=512, img_cols=512):
    images = []
    labels = []
    files = os.listdir(directory)
    for f in files:
        img = cv2.imread(os.path.join(directory, f), 1)
        if img is not None:
            img = cv2.resize(img, (img_rows, img_cols))
            images.append(img)
            labels.append(label)
    return images, labels



X_train, Y_train = [], []
X_test, Y_test = [], []
X_data, Y_data = [], []
X_train, Y_train = [], []
X_test, Y_test = [], []

data_dirs = {
    '/kaggle/input/kneemeddataset/KneeMedDataset/deepfake': 0,
    '/kaggle/input/kneemeddataset/KneeMedDataset/real': 1
    #'/kaggle/input/mydataset/Dataset-Lungs/Binary/Test/deepfake': 0,
   # '/kaggle/input/mydataset/Dataset-Lungs/Binary/Test/real': 1
}

for directory, label in data_dirs.items():
    images, labels = load_and_preprocess_images(directory, label, img_rows=224, img_cols=224)
    X_data.extend(images)
    Y_data.extend(labels)


X_data = np.array(X_data)
Y_data = np.array(Y_data)


img_rows, img_cols = 224,224
if tf.keras.backend.image_data_format() == 'channels_first':
    X_data = X_data.reshape(X_data.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_data = X_data.reshape(X_data.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

X_data = X_data.astype('float32') / 255.0

from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=4,stratify=Y_data)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")





class MultiScalePatches(Layer):
    def __init__(self, patch_sizes, projection_dim):
        super(MultiScalePatches, self).__init__()
        self.patch_sizes = patch_sizes
        self.projection_dim = projection_dim
        self.projection_layers = [Dense(projection_dim) for _ in patch_sizes]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches_list = []
        for i, patch_size in enumerate(self.patch_sizes):
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            projected_patches = self.projection_layers[i](patches)  # Project to the same dimension
            patches_list.append(projected_patches)
        return tf.concat(patches_list, axis=1)

def get_hybrid_model(input_shape):
    input_img = Input(input_shape)

    patch_sizes = [16, 32, 48]
    projection_dim = 64
    patches = MultiScalePatches(patch_sizes, projection_dim)(input_img)

    attn_output = AttentionModule(num_heads=8)(patches)

    num_patches = sum([(input_shape[0] // patch_size) * (input_shape[1] // patch_size) for patch_size in patch_sizes])
    patch_depth = projection_dim

    attn_output = Reshape((num_patches, patch_depth))(attn_output)
    x = Reshape((num_patches, patch_depth, 1))(attn_output)
    x = VGGBlock(num_convs=1, filters=32)(x)
 
    x = SEBlock(32)(x)
    x = VGGBlock(num_convs=1, filters=64)(x)  
   
    x = SEBlock(64)(x)  
    x = VGGBlock(num_convs=1, filters=128)(x)
 
    x = SEBlock(128)(x)
    x = VGGBlock(num_convs=1, filters=256)(x) 
    
    x = SEBlock(256)(x) 

    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.05)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_img], outputs=[x])
    return model

 

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fnr[np.nanargmin(np.abs(fnr - fpr))]
    return eer


num_folds = 3
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
metrics = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
    'eer': [], 'mcc': []
}


early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=5,
    restore_best_weights=True
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve
)
for fold, (train_index, test_index) in enumerate(kf.split(X_data)):
    print(f"\nTraining fold {fold + 1}/{num_folds}...")


    X_train_full, X_test = X_data[train_index], X_data[test_index]
    Y_train_full, Y_test = Y_data[train_index], Y_data[test_index]

    X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.1, random_state=42)

    
    model = get_hybrid_model(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    csv_logger = CSVLogger(f'training_history_fold_{fold + 1}.csv', append=False)

   
    history=model.fit(
        X_train, Y_train, 
        batch_size=16, 
        epochs=30, 
        validation_data=(X_val, Y_val), 
        shuffle=True,
          callbacks=[early_stopping, csv_logger]  
    )

  
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    

    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    eer = compute_eer(Y_test, y_pred_proba)
    mcc = matthews_corrcoef(Y_test, y_pred)
    
   
    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['eer'].append(eer)
    metrics['mcc'].append(mcc)

   
    print(f"Fold {fold + 1} Metrics:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, EER: {eer:.4f}, MCC: {mcc:.4f}")
    results = model.predict(X_test)
    predicted_probabilities = results.flatten()  
    from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    log_loss,
    matthews_corrcoef)
    class_report = classification_report(Y_test, (predicted_probabilities > 0.5).astype(int), target_names=['Deepfake', 'Real'])
    print("Classification Report:\n", class_report)
    


print("\nOverall Metrics (Mean ± Std):")
for metric, values in metrics.items():
    mean = np.mean(values)
    std_dev = np.std(values)
    print(f"{metric.capitalize()}: {mean:.4f} ± {std_dev:.4f}")




cm = confusion_matrix(Y_test, y_pred)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        annot[i, j] = f'{c}\n({p:.1f}%)'

# Plotting
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=annot, fmt='', cmap='BuPu', 
            xticklabels=['Deepfake', 'Real'], 
            yticklabels=['Deepfake', 'Real'], 
            linewidths=1.5, linecolor='black', cbar=True,
            annot_kws={"size": 13, "weight": "bold"})

plt.xlabel('Predicted Class', fontsize=12, weight='bold')
plt.ylabel('Actual Class', fontsize=12, weight='bold')
plt.title('Confusion Matrix', fontsize=14, weight='bold')
plt.savefig("Confusion matrix.png")
plt.tight_layout()
plt.show()



fpr, tpr, thresholds = roc_curve(Y_test, predicted_probabilities)
roc_auc = roc_auc_score(Y_test, predicted_probabilities)

fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
ax.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.5f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([0.0, 1.05])
ax.set_xticks(np.arange(0.0, 1.1, 0.1))
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
ax.legend(loc="lower right")
ax.grid(True)
plt.savefig("ROC.png")
plt.show()


import matplotlib.pyplot as plt  
plt.style.use('bmh')
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linestyle='-', marker='o', markersize=5)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linestyle='--', marker='x', markersize=5)
plt.title('Model Accuracy', fontsize=18)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0, 1.1) 
plt.grid(linewidth=2)
plt.legend(fontsize=18)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='red', linestyle='-', marker='o', markersize=5)
plt.plot(history.history['val_loss'], label='Validation Loss', color='green', linestyle='--', marker='x', markersize=5)
plt.title('Model Loss', fontsize=18)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(linewidth=2)
plt.legend(fontsize=18)


plt.savefig('training_validation_curves1.png', dpi=1000, bbox_inches='tight')
plt.show()
