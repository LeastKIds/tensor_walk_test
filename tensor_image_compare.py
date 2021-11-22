import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os.path

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './PhotoData/',
    image_size=(64,64),
    batch_size=10,
    subset='training',
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './PhotoData/',
    image_size=(64,64),
    batch_size=10,
    subset='validation',
    validation_split=0.2,
    seed=1234
)

def fff(i, result):
    i = tf.cast( i/255.0, tf.float32)
    return i, result

train_ds = train_ds.map(fff)
val_ds = val_ds.map(fff)


if not os.path.isdir('./model'):
    model = tf.keras.Sequential([


        tf.keras.layers.Conv2D(32,(3,3), padding="same", activation='relu', input_shape=(64,64,3)),
        tf.keras.layers.MaxPooling2D( (2,2) ),
        tf.keras.layers.Conv2D(32,(3,3), padding="same", activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D( (2,2) ),
        tf.keras.layers.Conv2D(32,(3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D( (2,2) ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    model = tf.keras.load_model('./model')


model.fit(train_ds, validation_data=val_ds, epochs=100)

model.save('./model')


# img = image.load_img('./PhotoData/answer/0.png', target_size=(64, 64))
img = cv2.imread('./PhotoData/answer/0.png')
img = cv2.resize(img, (64,64))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
# img_preprocessed = preprocess_input(img_batch)
prediction = model.predict(img_batch)
print(int(prediction[0]))
