import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.models import load_model
import tensorflow as tf
import pickle
from tqdm import tqdm
from numba import cuda
from glob import glob
import re


def load_mars_dataset(max_train=200, max_test=100):
    path_X = "Datasets\AI4Mars\msl\images\edr"
    path_y = "Datasets\AI4Mars\msl\images\mxy"


    images = "Datasets\AI4Mars\msl\images"
    tr_lab = "Datasets\AI4Mars\msl\labels\\train"
    
    test_lab = "Datasets/AI4Mars/msl/labels/test/masked-gold-min3-100agree"
    
    edr = images + "/edr"
    mxy = images + "/mxy" # not required
    rng = images + "/rng-30m" # not required


    edr_files = os.listdir(edr)
    rng_files = os.listdir(rng)
    mxy_files = os.listdir(mxy)
    trlab_files = os.listdir(tr_lab)
    telab_files = os.listdir(test_lab)

    X_mars = []
    y_mars = []
    X_test_mars = []
    y_test_mars = []
    c = 0

    # preparing X and y
    for lab_name in tqdm(trlab_files):
        img_name = lab_name[:-4] + ".JPG"
        
        if img_name in edr_files:
            
            img_path = os.path.join(edr, img_name)
            img_arr = cv2.imread(img_path)
            img_arr = cv2.resize(img_arr, dsize = (224, 224))
            
            lab_path = os.path.join(tr_lab, lab_name)
            lab_arr = cv2.imread(lab_path, 0)
            lab_arr = cv2.resize(lab_arr, (224, 224), interpolation = cv2.INTER_NEAREST)
            
            X_mars.append(img_arr)
            y_mars.append(lab_arr)
            
        c += 1
        if c >= max_train:
            break
    c = 0
    for lab_name in tqdm(telab_files):
        img_name = lab_name[:-11] + ".JPG"
        
        image_found = False
        
        if img_name in edr_files:
            img_path = os.path.join(edr, img_name)
            image_found = True
        else:
            img_name = lab_name[:-11] + ".png"
            print(img_name)
            if img_name in rng_files:
                img_path = os.path.join(rng, img_name)
                image_found = True
            if img_name in mxy_files:
                img_path = os.path.join(mxy, img_name)
                image_found = True
        
        if image_found:
            img_arr = cv2.imread(img_path)
            img_arr = cv2.resize(img_arr, dsize = (224, 224))
            
            lab_path = os.path.join(test_lab, lab_name)
            lab_arr = cv2.imread(lab_path, 0)
            lab_arr = cv2.resize(lab_arr, (224, 224), interpolation = cv2.INTER_NEAREST)
            
            X_test_mars.append(img_arr)
            y_test_mars.append(lab_arr)
            
        c += 1
        if c >= max_test:
            break
            
    X_test_mars = np.asarray(X_test_mars, dtype = np.float32) / 255.0
    y_test_mars = np.array(y_test_mars, dtype = np.uint8)

    # 0 - soil
    # 1 - bedrock
    # 2 - sand
    # 3 - big rock
    # 255 -> 4 - NULL (no label)


    # keeping integer values in labels will help us in segmentation task (UNet)
    y_test_mars[y_test_mars==255] = 4
    y_test_mars = y_test_mars > 2 * 1
    
    return X_mars, y_mars, X_test_mars, y_test_mars


def read_images(folder_path):
    images = []
    c = 0
    for filename in tqdm(os.listdir(folder_path)):
        if c % 100 == 0:
            img = cv2.imread(os.path.join(folder_path,filename))
            if img is not None:
                images.append(img)
        c += 1
    return images


def load_test_set():
    X_test = []
    y_test = []
    img_path = "Datasets/test_data/img"
    mask_path = "Datasets/test_data/masks"
    for image in tqdm(os.listdir(img_path)):
        X_test.append(cv2.imread(os.path.join(img_path, image)))

    for image in tqdm(os.listdir(mask_path)):
        y_test.append(cv2.imread(os.path.join(mask_path, image)))
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_test, y_test


# resize images to 224x224
def resize_images(images):
    resized_images = []
    for image in tqdm(images):
        resized_image = cv2.resize(image, (224, 224))
        resized_images.append(resized_image)
    return resized_images


def normalize(images):
    """
    Normalize an array of images between 0 and 1.

    Args:
        images (ndarray): Array of images.

    Returns:
        ndarray: Array of normalized images.
    """
    # Convert the images to a NumPy array if it's not already
    images = np.asarray(images)

    # Normalize the images between 0 and 1
    images = images / 255.0

    return images


def plot_roc_curve(y_test, y_pred):
    from sklearn.metrics import roc_curve, auc
    if len(y_test.shape) == 4:
        y_test_bin = (y_test[:, :, :, 0].reshape(-1) > 1) * 1
    else:
        y_test_bin = (y_test.reshape(-1) > 1) * 1
    #y_test_bin = (np.array(y_test)[:, :, :, 0].reshape(-1) > 1) * 1
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_pred.reshape(-1))
    auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc))
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    plt.show()
    
def show_results(model, X_test, y_test, count=None):
    if count is None:
        count = len(X_test)
    y_pred = model.predict(X_test)
    for i in range(count):
        plt.subplot(1,3,1)
        plt.title("Original Image")
        plt.imshow(X_test[i])

        plt.subplot(1,3,2)
        plt.title("Mask")
        plt.imshow((y_test[i] > 1) * 255, cmap='gray')

        plt.subplot(1,3,3)
        plt.title("Predicted Mask")
        plt.imshow(y_pred[i].reshape(224, 224), cmap='gray')
        plt.show()
        
        
from keras import layers
def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

def build_unet_model(base_units=8, b_w=False):
 # inputs
   if b_w:
       inputs = layers.Input(shape=(224,224, 1))
   else:
       inputs = layers.Input(shape=(224,224, 3))
   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, base_units)
   # 2 - downsample
   f2, p2 = downsample_block(p1, base_units*2)
   # 3 - downsample
   f3, p3 = downsample_block(p2, base_units*4)
   # 4 - downsample
   f4, p4 = downsample_block(p3, base_units*8)
   # 5 - bottleneck
   bottleneck = double_conv_block(p4, base_units*16)
   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, base_units*8)
   # 7 - upsample
   u7 = upsample_block(u6, f3, base_units*4)
   # 8 - upsample
   u8 = upsample_block(u7, f2, base_units*2)
   # 9 - upsample
   u9 = upsample_block(u8, f1, base_units)
   # outputs
   outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)
   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
   return unet_model


def load_data(path, test=False):
    images = sorted(glob(os.path.join(path, "images/*")))
    if test:
        masks = sorted(glob(os.path.join(path, "masks/*")), key=lambda x: int(re.findall(r'\d+', x)[-1]))
    else:
        masks = sorted(glob(os.path.join(path, "masks/*")))
    return images, masks

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (224, 224))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask(y)

        return x, y

    images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    images.set_shape([224, 224, 3])
    masks.set_shape([224, 224, 1])

    return images, masks

def tf_dataset(x, y, batch=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset

class CustomAugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, seed=42, **kwargs):
        super(CustomAugmentationLayer, self).__init__(**kwargs)
        self.seed = seed
        self.augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02, fill_mode='nearest'),
            tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.3, 0.0), width_factor=(-0.3, 0.0), fill_mode='nearest'),
            tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
        ])

    def call(self, inputs):
        image, mask = inputs
        image_and_mask = tf.concat([image, mask], axis=-1)  # Concatenate image and mask along the color dimension
        augmented_image_and_mask = self.augmentation_layers(image_and_mask, training=True)
        augmented_image, augmented_mask = tf.split(augmented_image_and_mask, num_or_size_splits=[3, 1], axis=-1)  # Split the augmented image and mask
        return augmented_image, augmented_mask


def augment(image, mask):
    
    augmentation_pipeline_only_images = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.3), # Randomly adjust contrast
        #tf.keras.layers.RandomBrightness(factor=0.2, seed=42),  # Randomly adjust brightness
        ])
    
    custom_augment = CustomAugmentationLayer()
    
    image, mask = custom_augment((image, mask))
    
    image = augmentation_pipeline_only_images(image, training=True)
    
    
    return image, mask     

def augment_dataset(dataset):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    # Apply the custom augmentation layer to the dataset
    augmented_dataset = dataset.map(lambda x, y: augment(x, y)).prefetch(AUTOTUNE)
    return augmented_dataset