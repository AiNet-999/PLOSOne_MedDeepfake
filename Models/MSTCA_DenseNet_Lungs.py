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
    LayerNormalization, BatchNormalization, Add, Concatenate,SpatialDropout2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
from itertools import cycle

warnings.filterwarnings('ignore')


#https://github.com/kobiso/CBAM-keras
#https://github.com/titu1994/keras-squeeze-excite-network

class DenseBlock(Layer):
    def __init__(self, num_layers, growth_rate, kernel_size=3):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.conv_layers = [Conv2D(growth_rate, kernel_size=kernel_size, padding='same') for _ in range(num_layers)]

    def call(self, inputs):
        x = inputs
        outputs = [x]  
        for conv in self.conv_layers:
            x = conv(x)
            x = LeakyReLU(alpha=0.05)(x)
            outputs.append(x) 
            x = Concatenate(axis=-1)(outputs) 
        return x

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

num_classes = 3  
data_dirs = {
    '/kaggle/input/meddata/MultiClass/MultiClass/Real': 0,
    '/kaggle/input/meddata/MultiClass/MultiClass/FM': 1,
    '/kaggle/input/meddata/MultiClass/MultiClass/FB': 2,  
   
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
Y_data = to_categorical(Y_data, num_classes=num_classes)
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
    x = DenseBlock(num_layers=3, growth_rate=32)(x)  # First dense block
    x = SEBlock(32)(x)  
    x = MaxPooling2D(2, 2)(x)

    x = DenseBlock(num_layers=3, growth_rate=64)(x)  # Second dense block
    x = SEBlock(64)(x)  
    x = MaxPooling2D(2, 2)(x)

    x = DenseBlock(num_layers=3, growth_rate=128)(x)  # Third dense block 
    x = SEBlock(128)(x) 
    x = MaxPooling2D(2, 2)(x)

    x = DenseBlock(num_layers=3, growth_rate=256)(x)  # Fourth dense block
    x = SEBlock(256)(x)  
    x = MaxPooling2D(2, 2)(x)
    


    


    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.05)(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=[input_img], outputs=[x])
    return model

 

def compute_multiclass_eer(y_true, y_pred_proba, num_classes):
    class_eers = []
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_proba_class = y_pred_proba[:, i] 
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba_class)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = fnr[np.nanargmin(np.abs(fnr - fpr))]
        class_eers.append(eer)
    return np.mean(class_eers) 


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
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

 
    history=model.fit(
        X_train, Y_train,
        batch_size=8,
        epochs=30,
        validation_data=(X_val, Y_val), 
        shuffle=True
    )

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    Y_test_classes = np.argmax(Y_test, axis=1)
    accuracy = accuracy_score(Y_test_classes , y_pred_classes)
    precision = precision_score(Y_test_classes , y_pred_classes,average='macro' )
    recall = recall_score(Y_test_classes , y_pred_classes,average='macro' )
    f1 = f1_score(Y_test_classes , y_pred_classes,average='macro' )

    eer = compute_multiclass_eer(Y_test_classes, y_pred, num_classes)
    mcc = matthews_corrcoef(Y_test_classes, y_pred_classes)

 
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
    roc_auc_score, confusion_matrix,
    classification_report,
    accuracy_score,
    log_loss,
    matthews_corrcoef)
    target_names = ['Real', 'FM', 'FB']  
    class_report = classification_report(Y_test_classes, y_pred_classes, target_names=target_names)
    print("Classification Report:\n", class_report)


print("\nOverall Metrics (Mean ± Std):")
for metric, values in metrics.items():
    mean = np.mean(values)
    std_dev = np.std(values)
    print(f"{metric.capitalize()}: {mean:.4f} ± {std_dev:.4f}")



cm = confusion_matrix(Y_test_classes, y_pred_classes)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        annot[i, j] = f'{c}\n({p:.1f}%)'

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=annot, fmt='', cmap='BuPu',
            xticklabels=target_names,
            yticklabels=target_names,
            linewidths=1.5, linecolor='black', cbar=True,
            annot_kws={"size": 13, "weight": "bold"})

plt.xlabel('Predicted Class', fontsize=12, weight='bold')
plt.ylabel('Actual Class', fontsize=12, weight='bold')
plt.title('Confusion Matrix', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("Confusion matrix.png")
plt.show()





Y_test_bin = label_binarize(Y_test_classes, classes=range(num_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
plt.figure(figsize=(7, 6))
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(target_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.01, 1.01])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Multi-class ROC Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
plt.tight_layout()
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
