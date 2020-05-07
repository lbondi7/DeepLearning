def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

RESOLUTION_SCALE = 2
RESOLUTION = 32 * RESOLUTION_SCALE
RGB_CHANNELS = 3

ROW_COUNT = 1
COL_COUNT = 1
IMAGE_MARGIN = 0
SAVE_FREQ = 10

SEED_SIZE = 10

EPOCHS = 1000
BATCH_SIZE = 32


DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + "\\dataset"
OUTPUT_PATH = os.path.dirname(os.path.realpath(__file__)) + "\\generated_images"

name_of_output_data = input("Please enter a name for data, e.g 'generated bricks' \n")
trainingDataName = input("Please enter a name for the data you want to train from, e.g 'bricks'\n")


training_binary_path = os.path.join(DATA_PATH,trainingDataName + f'_{RESOLUTION}_{RESOLUTION}.npy')

print("Looking for file: "+ training_binary_path)

    
if not os.path.isfile(training_binary_path):

  training_data = []
  trainingDataDir = os.path.join(DATA_PATH, trainingDataName)
  for filename in tqdm(os.listdir(trainingDataDir)):
      path = os.path.join(trainingDataDir,filename)
      image = Image.open(path).resize((RESOLUTION, RESOLUTION),Image.ANTIALIAS)
      training_data.append(np.asarray(image))
  training_data = np.reshape(training_data,(-1,RESOLUTION, RESOLUTION,RGB_CHANNELS))
  training_data = training_data / 127.5 - 1.

  np.save(training_binary_path,training_data)
  
else:
  training_data = np.load(training_binary_path)

def build_generator(seed_size, channels):
    model = Sequential()

    model.add(Dense(4*4*256, activation = "relu", input_dim = seed_size))
    model.add(Reshape((4, 4, 256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size = 3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size = 3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    for i in range(RESOLUTION_SCALE):
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size = 3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

    model.add(Conv2D(channels, kernel_size = 3, padding ="same"))
    model.add(Activation("tanh"))

    input = Input(shape=(seed_size,))
    generated_image = model(input)

    return Model(input, generated_image)
        
def build_discriminator(image_shape):
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    

    input_image = Input(shape=image_shape)

    validity = model(input_image)
    
    return Model(input_image, validity)

def save_images(cnt, noise):
  image_array = np.full(( 
      IMAGE_MARGIN + (ROW_COUNT * (RESOLUTION+IMAGE_MARGIN)), 
      IMAGE_MARGIN + (COL_COUNT * (RESOLUTION+IMAGE_MARGIN)), 3), 
      255, dtype=np.uint8)
  
  generated_images = generator.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row  in range(ROW_COUNT):
      for col in range(COL_COUNT):
          r = row * (RESOLUTION+ 16) + IMAGE_MARGIN
          c = col * (RESOLUTION+ 16) + IMAGE_MARGIN
          image_array[r:r+RESOLUTION,c:c+RESOLUTION] = generated_images[image_count] * 255
          image_array[r:r+RESOLUTION,c:c+RESOLUTION] = generated_images[image_count] * 255
          image_count += 1
    
  if not os.path.isdir(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
            
  output_path = os.path.join(OUTPUT_PATH, name_of_output_data)
  if not os.path.isdir(output_path):
      os.mkdir(output_path)
      
  print(output_path)
  
  filename = os.path.join(output_path, name_of_output_data + f"_train-{cnt}.png")
  im = Image.fromarray(image_array)
  im.save(filename)


image_shape = (RESOLUTION, RESOLUTION, RGB_CHANNELS)
optimizer = Adam(1.5e-4, 0.5)

discriminator = build_discriminator(image_shape)
discriminator.compile(loss = "binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
generator = build_generator(SEED_SIZE, RGB_CHANNELS)

random_input = Input(shape=(SEED_SIZE,))

generated_image = generator(random_input)

discriminator.trainable = False

validity = discriminator(generated_image)

combined = Model(random_input, validity)
combined.compile(loss="binary_crossentropy", optimizer= optimizer, metrics=["accuracy"])

y_real = np.ones((BATCH_SIZE, 1))
y_fake = np.zeros((BATCH_SIZE, 1))

fixed_seed = np.random.normal(0, 1, (ROW_COUNT* COL_COUNT, SEED_SIZE))

cnt = 1
a = ["start"]
for epoch in range(EPOCHS):
    idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)
    x_real = training_data[idx]

    seed = np.random.normal(0, 1, (BATCH_SIZE, SEED_SIZE))
    x_fake = generator.predict(seed)


    discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
    discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)
    discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)

    generator_metric = combined.train_on_batch(seed, y_real)

    if epoch % SAVE_FREQ == 0:
        save_images(cnt, fixed_seed)
        cnt += 1
        print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")
        a += f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}"
        

file1 = open(os.path.join(DATA_PATH, trainingDataName +"SaveData.txt"),"w")
for b in a:
  file1.write(b+'\n')
file1.close()
generator.save(os.path.join(DATA_PATH, trainingDataName +"_generator.h5"))