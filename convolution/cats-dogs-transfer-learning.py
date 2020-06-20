#wget --no-check-certificate \
  #https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
  #-O /tmp/cats_and_dogs_filtered.zip

import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model
from tensorflow.keras import layers
import PIL

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = './data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None
)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dropout(0.2)(x)

x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)


data_location = 'data/cats_and_dogs_filtered.zip'

zip_ref = zipfile.ZipFile(data_location)
zip_ref.extractall('data/')
zip_ref.close()


base_dir = 'data/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cats_fnames = os.listdir(train_cats_dir)
train_dogs_fnames = os.listdir(train_dogs_dir)
validation_cats_fnames = os.listdir(validation_cats_dir)
validation_dogs_fnames = os.listdir(validation_dogs_dir)

print('Training')
print('Cats: ', len(train_cats_fnames))
print('Dogs: ', len(train_dogs_fnames))
print('Validation')
print('Cats: ', len(validation_cats_fnames))
print('Dogs: ', len(validation_dogs_fnames))

model.summary()

model.compile(optimizer=RMSprop(lr = 0.001), loss = 'binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs = 15,
    validation_steps = 50,
    verbose = 2
)

