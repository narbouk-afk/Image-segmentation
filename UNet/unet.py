import tensorflow as tf
import numpy as np

import tensorflow.keras.layers as tkl
import pandas

def down_conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    conv = tkl.Conv2D(n_filters,3,activation="relu",padding="same",kernel_initializer="he_normal")(inputs)
    conv = tkl.Conv2D(n_filters,3,activation="relu",padding="same",kernel_initializer="he_normal")(conv)

    if dropout_prob > 0:
        conv = tkl.Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = tkl.MaxPooling2D(pool_size=(2, 2))(conv)
        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection

def up_conv_block(input, skip, n_filters=32):

    """
    Convolutional upsampling block
    
    Arguments:
        input -- Input tensor from previous layer
        skip -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """"""
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """

    up = tkl.Conv2DTranspose(n_filters,3,strides=(2,2),padding="same")(input)
    
    # Merge the previous output and the skip
    merge = tkl.concatenate([up, skip], axis=3)
    
    conv = tkl.Conv2D(n_filters,3,activation='relu',padding="same",kernel_initializer="HeNormal")(merge)
    conv = tkl.Conv2D(n_filters,3,activation='relu',padding="same",kernel_initializer="HeNormal")(conv)
    
    return conv


def unet(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs = tkl.Input(input_size)
    
    # Contracting Path (encoding)
    cblock1 = down_conv_block(inputs, n_filters)

    cblock2 = down_conv_block(cblock1[0], 2*n_filters)
    cblock3 = down_conv_block(cblock2[0], 4*n_filters)
    cblock4 = down_conv_block(cblock3[0], 8*n_filters, dropout_prob=0.3) 

    cblock5 = down_conv_block(cblock4[0], 16*n_filters, dropout_prob=0.3, max_pooling=False) 

    
    # Expanding Path (decoding)

    ublock6 = up_conv_block(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = up_conv_block(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = up_conv_block(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = up_conv_block(ublock8, cblock1[1],  n_filters)


    conv9 = tkl.Conv2D(n_filters,3,activation='relu',padding='same',kernel_initializer='he_normal')(ublock9)


    conv10 = tkl.Conv2D(n_classes, 1, padding="same")(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

