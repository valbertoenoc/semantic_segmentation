import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, ReLU

def conv2d_t(filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)

    layer = Sequential()
    layer.add(Conv2DTranspose(filters, size, strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))

    layer.add(BatchNormalization())
    layer.add(ReLU())

    return layer


def unet_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
    
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   
        'block_3_expand_relu',   
        'block_6_expand_relu',   
        'block_13_expand_relu',  
        'block_16_project',      
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Encoder 
    encoder = tf.keras.Model(inputs=base_model.input, outputs=layers)

    encoder.trainable = False

    decoder = [
        conv2d_t(512, 3), 
        conv2d_t(256, 3), 
        conv2d_t(128, 3), 
        conv2d_t(64, 3), 
    ]

    # This is the last layer of the model
    output_channels = 3
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, 
                padding='same', activation='softmax') 

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # Downsampling through the model (Decoder) - Image -> Features
    skips = encoder(x)
    # Last layer index
    x = skips[-1]
    # Reverting layers to compose Encoder
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections (Encoder) -> Features to Image
    for up, skip in zip(decoder, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = last(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    return model