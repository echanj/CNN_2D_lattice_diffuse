from tensorflow import keras
from tensorflow.keras import layers
from keras.initializers import HeNormal as he_normal


def reconstruct_small_cfs_model(nout,my_weights):

    input_shape = (64, 64, 1)
    model = keras.Sequential(
        [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",name='conv1'),
        layers.BatchNormalization(name='norm1'),
        layers.MaxPooling2D(pool_size=(2, 2),name='maxpool1'),
        layers.Dropout(0.2,name='drop1'),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu",name='conv2'),
        layers.BatchNormalization(name='norm2'),
        layers.MaxPooling2D(pool_size=(2, 2),name='maxpool2'),
        layers.Flatten(name='flatten1'),
        layers.Dropout(0.5,name='drop2'),
        layers.Dense(units=nout, activation="linear",name='dense_out'),
        ],name='seq_CNN_sm'
    )
    # Load the saved weights into the model
    model.load_weights(my_weights)
    return model


def reconstruct_medium_cfs_model(nout,my_weights):

	inputs = keras.Input(shape=(64, 64, 1))

	#First block with downsampling
	x1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(inputs)
	x1 = layers.BatchNormalization()(x1)
	x1 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x1)  # Downsample with stride=2
	x1 = layers.BatchNormalization()(x1)
	x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
	x1 = layers.Dropout(0.2)(x1)


	#Second block with skip connection
	x2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x1)
	x2 = layers.BatchNormalization()(x2)
	x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
	x2 = layers.Dropout(0.25)(x2)
	x2 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x2)
	x2 = layers.BatchNormalization()(x2)

	x1_downsampled = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x1)  #Downsample with stride=2
	x2 = layers.Add()([x2, x1_downsampled])  #Add skip connection
	x2 = layers.Activation('relu')(x2)
	x2 = layers.Dropout(0.25)(x2)

	# route to Dense Block
	x3 = layers.Flatten()(x2)
	x3 = layers.Dense(units=128, activation="relu", kernel_initializer=he_normal())(x3)
	x3 = layers.BatchNormalization()(x3)
	x3 = layers.Dropout(0.5)(x3)
	x3 = layers.Dense(units=nout, activation='linear', kernel_initializer=he_normal())(x3)
	model = keras.Model(inputs=inputs, outputs=x3, name='CNN_with_skip_medium')

#	model_md.summary()

	# Load the saved weights into the model
	model.load_weights(my_weights)
	return model



def reconstruct_large_cfs_model(nout,my_weights):

   inputs = keras.Input(shape=(64, 64, 1))
   #First block with downsampling
   x1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(inputs)
   x1 = layers.BatchNormalization()(x1)
   x1 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x1)  # Downsample with stride=2
   x1 = layers.BatchNormalization()(x1)
   x1 = layers.Dropout(0.25)(x1)
   #Second block with skip connection
   x2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x1)
   x2 = layers.BatchNormalization()(x2)
   x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
   x2 = layers.Dropout(0.25)(x2)
   x2 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x2)
   x2 = layers.BatchNormalization()(x2)
   x1_downsampled = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x1)  #Downsample with stride=2
   x2 = layers.Add()([x2, x1_downsampled])  #Add skip connection
   x2 = layers.Activation('relu')(x2)
   x2 = layers.Dropout(0.25)(x2)
   #Third block with skip connection
   x3 = layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer=he_normal())(x2)
   x3 = layers.BatchNormalization()(x3)
   x3 = layers.MaxPooling2D(pool_size=(2, 2))(x3)
   x3 = layers.Dropout(0.25)(x3)
   x3 = layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer=he_normal())(x3)
   x3 = layers.BatchNormalization()(x3)
   x2_downsampled = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x2)  #Downsample with stride=2
   x3 = layers.Add()([x3, x2_downsampled])  #Add skip connection
   x3 = layers.Activation('relu')(x3)
   x3 = layers.Dropout(0.25)(x3)
   #Fourth block with skip connection and additional dropout
   x4 = layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer=he_normal())(x3)
   x4 = layers.BatchNormalization()(x4)
   x4 = layers.MaxPooling2D(pool_size=(2, 2))(x4)
   x4 = layers.Dropout(0.25)(x4)
   x4 = layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer=he_normal())(x4)
   x4 = layers.BatchNormalization()(x4)
   x3_downsampled = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x3)  #Downsample with stride=2
   x4 = layers.Add()([x4, x3_downsampled])  #Add skip connection
   x4 = layers.Activation('relu')(x4)
   x4 = layers.Dropout(0.5)(x4)
   x4 = layers.Flatten()(x4)
   x4 = layers.Dense(units=128, activation="relu", kernel_initializer=he_normal())(x4)
   x4 = layers.BatchNormalization()(x4)
   x4 = layers.Dropout(0.5)(x4)
   x4 = layers.Dense(units=nout, activation='linear', kernel_initializer=he_normal())(x4)
   model = keras.Model(inputs=inputs, outputs=x4, name='CNN_with_skip_large')
#   model.summary()

   # Load the saved weights into the model
   model.load_weights(my_weights)
   return model


def construct_new_small_cfs_model(nout):

    input_shape = (64, 64, 1)
    model = keras.Sequential(
        [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",name='conv1'),
        layers.BatchNormalization(name='norm1'),
        layers.MaxPooling2D(pool_size=(2, 2),name='maxpool1'),
        layers.Dropout(0.2,name='drop1'),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu",name='conv2'),
        layers.BatchNormalization(name='norm2'),
        layers.MaxPooling2D(pool_size=(2, 2),name='maxpool2'),
        layers.Flatten(name='flatten1'),
        layers.Dropout(0.5,name='drop2'),
        layers.Dense(units=nout, activation="linear",name='dense_out'),
        ],name='seq_CNN_sm'
    )
    return model


def construct_new_medium_cfs_model(nout):

	inputs = keras.Input(shape=(64, 64, 1))

	#First block with downsampling
	x1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(inputs)
	x1 = layers.BatchNormalization()(x1)
	x1 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x1)  # Downsample with stride=2
	x1 = layers.BatchNormalization()(x1)
	x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
	x1 = layers.Dropout(0.2)(x1)


	#Second block with skip connection
	x2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x1)
	x2 = layers.BatchNormalization()(x2)
	x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
	x2 = layers.Dropout(0.25)(x2)
	x2 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x2)
	x2 = layers.BatchNormalization()(x2)

	x1_downsampled = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x1)  #Downsample with stride=2
	x2 = layers.Add()([x2, x1_downsampled])  #Add skip connection
	x2 = layers.Activation('relu')(x2)
	x2 = layers.Dropout(0.25)(x2)

	# route to Dense Block
	x3 = layers.Flatten()(x2)
	x3 = layers.Dense(units=128, activation="relu", kernel_initializer=he_normal())(x3)
	x3 = layers.BatchNormalization()(x3)
	x3 = layers.Dropout(0.5)(x3)
	x3 = layers.Dense(units=nout, activation='linear', kernel_initializer=he_normal())(x3)
	model = keras.Model(inputs=inputs, outputs=x3, name='CNN_with_skip_medium')

#	model_md.summary()

	return model



def construct_new_large_cfs_model(nout):

   inputs = keras.Input(shape=(64, 64, 1))
   #First block with downsampling
   x1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(inputs)
   x1 = layers.BatchNormalization()(x1)
   x1 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x1)  # Downsample with stride=2
   x1 = layers.BatchNormalization()(x1)
   x1 = layers.Dropout(0.25)(x1)
   #Second block with skip connection
   x2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x1)
   x2 = layers.BatchNormalization()(x2)
   x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
   x2 = layers.Dropout(0.25)(x2)
   x2 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x2)
   x2 = layers.BatchNormalization()(x2)
   x1_downsampled = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x1)  #Downsample with stride=2
   x2 = layers.Add()([x2, x1_downsampled])  #Add skip connection
   x2 = layers.Activation('relu')(x2)
   x2 = layers.Dropout(0.25)(x2)
   #Third block with skip connection
   x3 = layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer=he_normal())(x2)
   x3 = layers.BatchNormalization()(x3)
   x3 = layers.MaxPooling2D(pool_size=(2, 2))(x3)
   x3 = layers.Dropout(0.25)(x3)
   x3 = layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer=he_normal())(x3)
   x3 = layers.BatchNormalization()(x3)
   x2_downsampled = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x2)  #Downsample with stride=2
   x3 = layers.Add()([x3, x2_downsampled])  #Add skip connection
   x3 = layers.Activation('relu')(x3)
   x3 = layers.Dropout(0.25)(x3)
   #Fourth block with skip connection and additional dropout
   x4 = layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer=he_normal())(x3)
   x4 = layers.BatchNormalization()(x4)
   x4 = layers.MaxPooling2D(pool_size=(2, 2))(x4)
   x4 = layers.Dropout(0.25)(x4)
   x4 = layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer=he_normal())(x4)
   x4 = layers.BatchNormalization()(x4)
   x3_downsampled = layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=he_normal(), strides=2)(x3)  #Downsample with stride=2
   x4 = layers.Add()([x4, x3_downsampled])  #Add skip connection
   x4 = layers.Activation('relu')(x4)
   x4 = layers.Dropout(0.5)(x4)
   x4 = layers.Flatten()(x4)
   x4 = layers.Dense(units=128, activation="relu", kernel_initializer=he_normal())(x4)
   x4 = layers.BatchNormalization()(x4)
   x4 = layers.Dropout(0.5)(x4)
   x4 = layers.Dense(units=nout, activation='linear', kernel_initializer=he_normal())(x4)
   model = keras.Model(inputs=inputs, outputs=x4, name='CNN_with_skip_large')
#   model.summary()

   return model




