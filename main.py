import tensorflow as tf
from keras import layers, models
# from keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = 'training_set/training_set'
validation_dir = 'test_set/test_set'

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

# define cnn model
model = models.Sequential()

# first convolutional layer
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2,2)))

# second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# third convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fourth convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# flatten the output from the convolutional layers and add fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) # Output layer for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# print a summary of the model
model.summary()

# train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,      # no. of batches per epoch
    epochs=20,                # no. of epochs to train
    validation_data=validation_generator,
    validation_steps=50       # no. of batches for validation
)

# plot training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# from keras.preprocessing import image
from keras.src.legacy.preprocessing import image
import numpy as np

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # load the image
    img_array = image.img_to_array(img)  # convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array /= 255.0  # normalize the image (rescale pixel values to [0, 1])
    
    prediction = model.predict(img_array)  
    
    if prediction[0] > 0.5:
        print(f"The image is predicted to be a Dog with a confidence of {prediction[0][0]:.2f}")
    else:
        print(f"The image is predicted to be a Cat with a confidence of {1 - prediction[0][0]:.2f}")

predict_image(model, 'cat.jpg')