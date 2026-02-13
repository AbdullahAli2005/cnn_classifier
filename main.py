import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = 'training_set'
validation_dir = 'test_set'

# data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,     # randomly shift images horizontally
    height_shift_range=0.2,    # randomly shift images vertically
    shear_range=0.2,           # randomly shear images
    zoom_range=0.2,            # randomly zoom in on images
    horizontal_flip=True,      # randomly flip images horizontally
    fill_mode='nearest'        # fill pixels that may have been lost after transformation
)

# for the validation data, we just rescale (no data augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# load training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150), # resize all images 150x150
    batch_size=32,
    class_mode='binary' # cat or dog
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

model = models.Sequential()