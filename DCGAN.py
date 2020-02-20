from datetime import time

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import generateDataset

def make_generator_model(original_w,noise_dim = 100):
    g = 128
    s16 = original_w // 32
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense((g * 8 * s16 * s16), use_bias=False, input_shape=(noise_dim,)))
    #model.add(tf.keras.layers.Dense((g * 4 * s16 * s16), activation=tf.nn.leaky_relu))
    #model.add(tf.keras.layers.Dense((g * 4 * s16 * s16), activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape([ s16,s16,8*g], input_shape=(noise_dim,)))
    print(">>",model.output_shape)

    model.add(tf.keras.layers.Conv2DTranspose(filters=8*g, kernel_size=(4, 4), strides=(2, 2), padding="same",   kernel_initializer='he_uniform'))
    #model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    print(model.output_shape)


    model.add(tf.keras.layers.Conv2DTranspose(filters=4*g, kernel_size=(5, 5), strides=(2, 2), padding="same",  kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    print(model.output_shape)


    model.add(tf.keras.layers.Conv2DTranspose(filters=2*g, kernel_size=(5, 5), strides=(2, 2), padding="same", kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    print(model.output_shape)

    model.add(tf.keras.layers.Conv2DTranspose(filters=g, kernel_size=(5, 5), strides=(2, 2), padding="same", kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    print(model.output_shape)

    model.add(  tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), padding="same",  kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Activation(activation=tf.nn.tanh))
    print( model.output_shape)
    assert model.output_shape == (None,original_w,original_w,3)
    return model

def make_discriminator_model(original_w, noise_var ):
    g = 512
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GaussianNoise( noise_var ) )

    model.add(tf.keras.layers.Conv2D(g//16, (5, 5), strides=(2, 2), padding='same', input_shape=[None,original_w, original_w, 3]))
    #model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(g//8, (5, 5), strides=(2, 2), padding='same'))
    #model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(g//4, (5, 5), strides=(2, 2), padding='same'))
    #model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(g//2, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(g, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(g//4,activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(g//8,activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(1 ))
    return model


original_w = 64
batch_size = 16
EPOCHS = 900
num_examples_to_generate = 25
noise_dim = 100

seed = tf.random.normal([num_examples_to_generate, noise_dim])

noise_var = tf.Variable(initial_value=0.05)

generator = make_generator_model(original_w, noise_dim)
discriminator = make_discriminator_model(original_w,noise_var)




if True :
   noise = tf.random.normal([1, noise_dim])
   generated_image = generator(noise, training=False)
   decision = discriminator(generated_image)
   print(decision)
   #plt.imshow( generated_image[0] *0.5 + 0.5  )
   #plt.show()
generator.summary()
discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_fake_loss(  fake_output):
    #real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output)  + 0.3*tf.random.uniform( shape= fake_output.shape), fake_output)
    #total_loss = real_loss + fake_loss
    return fake_loss

def discriminator_real_loss(real_output ):
    real_loss = cross_entropy(tf.ones_like(real_output)-0.3 + 0.5*tf.random.uniform( shape= real_output.shape), real_output)
    #fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    #total_loss = real_loss + fake_loss
    return real_loss



def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0004, beta_1=0.5)

print("Start Loading Dataset")
training_dataset = generateDataset.getImageDataSet(original_w).batch(batch_size)
print("End Loading Dataset")
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, D = False ):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

@tf.function
def train_step_real(images ):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
              real_output = discriminator(images, training=True)
              disc_r_loss = discriminator_real_loss(real_output )
              gradients_of_real_discriminator = disc_tape.gradient(disc_r_loss, discriminator.trainable_variables)
              discriminator_optimizer.apply_gradients(zip(gradients_of_real_discriminator, discriminator.trainable_variables))

@tf.function
def train_step_fake( ):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          noise = tf.random.normal([batch_size, noise_dim])
          generated_images = generator(noise, training=True)
          fake_output = discriminator(generated_images, training=True)
          disc_f_loss = discriminator_fake_loss(fake_output )
          gradients_of_fake_discriminator = disc_tape.gradient(disc_f_loss, discriminator.trainable_variables)
          discriminator_optimizer.apply_gradients(zip(gradients_of_fake_discriminator, discriminator.trainable_variables))



def getError(model,test_input):
    predictions = model(test_input, training=False)
    fake_values = np.mean(discriminator(predictions, training=False))
    print(fake_values)
    return fake_values

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  fake_values = np.mean( discriminator(predictions, training=False))
  print(fake_values)


  fig = plt.figure(figsize=(5,5))
  for i in range(predictions.shape[0]):
      plt.subplot(5, 5, i+1 )
      plt.imshow(predictions[i  ] *0.5 + 0.5 )
      plt.axis('off')
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.close('all')


  #plt.show()




def train(dataset, epochs):
  for epoch in range(epochs):
    print("Start Epoch", epoch)
    bb = 0
    for image_batch in dataset :
       train_step(image_batch, D = True )
       train_step_real(image_batch)
       train_step_fake()
       bb += 1
       if ((bb * batch_size) % ( 1024 ) ==0 ):
           print("Progress "+str(bb*batch_size),end='\r')
       #if ((bb*batch_size) > (  10 * 1024 ) ):
           #print("")
           #break


    print('')
    for i in range(4):
        r = getError(generator,  tf.random.normal([num_examples_to_generate, noise_dim]))
        if r > -3.5: break
        for image_batch in dataset:
            train_step(image_batch, D= False )

    noise_var.assign( 0.9*noise_var.value()) # reduz o ruido
    generator.save_weights("generator")
    generate_and_save_images(generator, epoch, seed)
    print ('epoch ',epoch )
    dataset.shuffle(buffer_size=  1024)

  # Generate after the final epoch
  generate_and_save_images(generator,epochs,seed)


train(training_dataset, EPOCHS)
