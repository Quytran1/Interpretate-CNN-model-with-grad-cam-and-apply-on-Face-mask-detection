import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from dataset import get_data
from model import get_model


if __name__ == '__main__':
    # get data
    data, labels = get_data()
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

    # get model
    model = get_model()
    # train model
    INIT_LR = 1e-4
    EPOCHS = 2
    BS = 32

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    trainY_one_hot = tf.one_hot(trainY, depth=2)
    testY_one_hot = tf.one_hot(testY, depth=2)
    optimizer = Adam(learning_rate=INIT_LR)#, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    H = model.fit(
    aug.flow(trainX, trainY_one_hot, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY_one_hot),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
    )

    print("[INFO] saving mask detector model...")
    # Lưu mô hình dưới dạng SavedModel
    os.makedirs('Detector')

        # Lưu mô hình vào thư mục
    model.save('Detector/Train1.h5')