from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt

from data_process import normalize, load_image_test, load_image_train, load_dataset_oxford_iiit_pet
from vis_utils import DisplayCallback, display
from model import unet_model

def train_model(model, train_dataset, test_dataset, info):
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    EPOCHS = 20
    VAL_SUBSPLITS = 5
    BATCH_SIZE = 64
    TRAIN_LENGTH = info.splits['train'].num_examples
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_steps=VALIDATION_STEPS,
                                validation_data=test_dataset,
                                callbacks=[DisplayCallback()])

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(EPOCHS)

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    return model


def main():
    train, test, train_dataset, test_dataset, info = load_dataset_oxford_iiit_pet()

    for image, mask in train.take(6):
        sample_image, sample_mask = image, mask
    display([sample_image, sample_mask])

    model = unet_model()

    train_model(model, train_dataset, test_dataset, info)

if __name__ == "__main__":
    main()