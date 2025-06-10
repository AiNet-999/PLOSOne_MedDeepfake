import os
import random
import numpy as np
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)              
os.environ['TF_DETERMINISTIC_OPS'] = '1'             
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_curve, roc_auc_score, confusion_matrix, classification_report
)

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


from tensorflow.keras.regularizers import l2

def ResidualBlock(filters):
    def block(x_input):
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.05)(x)

        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)

        x = Add()([x, x_input])
        x = LeakyReLU(alpha=0.05)(x)
        return x
    return block




from tensorflow.keras.layers import Conv2D

def ResidualBlock(filters):
    def block(x_input):
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = SpatialDropout2D(0.2)(x)

        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)

        # If number of filters doesn't match, project input
        if x_input.shape[-1] != filters:
            x_input = Conv2D(filters, (1,1), padding='same', kernel_regularizer=l2(1e-4))(x_input)
            x_input = BatchNormalization()(x_input)

        x = Add()([x, x_input])
        x = LeakyReLU(alpha=0.05)(x)
        return x
    return block


class AttentionModule(Layer):
    def __init__(self, num_heads):
        super(AttentionModule, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=256)
        self.layer_norm = LayerNormalization()

    def call(self, patches):
        attn_output = self.attention(patches, patches)
        return self.layer_norm(attn_output + patches)  
        
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
# Define training and test directories with class labels
train_dirs = {
    '/kaggle/input/dataset2/Train/MultiClass/Real': 0,
    '/kaggle/input/dataset2/Train/MultiClass/FM': 1,
    '/kaggle/input/dataset2/Train/MultiClass/FB': 1
}

test_dirs = {
    '/kaggle/input/dataset2/Test/MultiClass/Real': 0,
    '/kaggle/input/dataset2/Test/MultiClass/FM': 1,
    '/kaggle/input/dataset2/Test/MultiClass/FB': 1
}

# Initialize empty lists
X_train, Y_train = [], []
X_test, Y_test = [], []

# Load training data
for directory, label in train_dirs.items():
    images, labels = load_and_preprocess_images(directory, label, img_rows=224, img_cols=224)
    X_train.extend(images)
    Y_train.extend(labels)

# Load test data
for directory, label in test_dirs.items():
    images, labels = load_and_preprocess_images(directory, label, img_rows=224, img_cols=224)
    X_test.extend(images)
    Y_test.extend(labels)

# Convert to NumPy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Reshape based on image format
img_rows, img_cols = 224, 224
if tf.keras.backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# Normalize and one-hot encode
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

#Y_train = to_categorical(Y_train, num_classes=num_classes)
#Y_test = to_categorical(Y_test, num_classes=num_classes)

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
    
    """x = VGGBlock(num_convs=1, filters=32)(x)
    x = SEBlock()(x)
    x = VGGBlock(num_convs=1, filters=64)(x)  
    x = SEBlock()(x)  
    x = VGGBlock(num_convs=1, filters=128)(x)
    x = SEBlock()(x)
    x = VGGBlock(num_convs=1, filters=256)(x) 
    x = SEBlock()(x)

    x = ResidualBlock(32)(x)
    x = SEBlock()(x)  
    x = MaxPooling2D(2, 2)(x)

    x = ResidualBlock(64)(x)
    x = SEBlock()(x)  
    x = MaxPooling2D(2, 2)(x)

    x = ResidualBlock(128)(x)
    x = SEBlock()(x) 
    x = MaxPooling2D(2, 2)(x)

    x = ResidualBlock(256)(x)
    x = SEBlock()(x)  
    x = MaxPooling2D(2, 2)(x)"""
    
    x = DenseBlock(num_layers=3, growth_rate=32)(x)  # First dense block
    x = SEBlock()(x)  
    x = MaxPooling2D(2, 2)(x)
    x = DenseBlock(num_layers=3, growth_rate=64)(x)  # Second dense block
    x = SEBlock()(x)  
    x = MaxPooling2D(2, 2)(x)
    x = DenseBlock(num_layers=3, growth_rate=128)(x)  # Third dense block 
    x = SEBlock()(x) 
    x = MaxPooling2D(2, 2)(x)
    x = DenseBlock(num_layers=3, growth_rate=256)(x)  # Fourth dense block
    x = SEBlock()(x)  
    x = MaxPooling2D(2, 2)(x)
    
    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.05)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_img], outputs=[x])
    return model

 


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

  
model = get_hybrid_model(input_shape)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

 
history=model.fit(
        X_train, Y_train,
        batch_size=8,
        epochs=60,
        #validation_split=0.1,
        #shuffle=False
    )




def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fnr[np.nanargmin(np.abs(fnr - fpr))]
    return eer



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


print("\nOverall Metrics:")
for metric, values in metrics.items():
    mean = np.mean(values)
    print(f"{metric.capitalize()}: {mean:.4f}")

