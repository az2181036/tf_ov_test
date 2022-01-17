import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
import pathlib
import sys

import mo_tf
from openvino.inference_engine import IECore

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory


# download and explore dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

img_height, img_width, batch_size = 180, 180, 32
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training", seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

num_classes = 5

model = Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 10
# history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model_dir = "model"
model_fname = f"{model_dir}/flower"
# model.save(model_fname)
model = tf.saved_model.load(model_fname)
model_name = "flower"
model_path = pathlib.Path(model_fname)
ir_data_type = "FP16"
ir_model_name = "flower_ir"

# Get the path to the Model Optimizer script
mo_path = str(pathlib.Path(mo_tf.__file__))

# Construct the command for Model Optimizer
mo_command = f""""mo
                 --saved_model_dir "{model_fname}"
                 --input_shape "[180,180,3]"
                 --data_type "{ir_data_type}" 
                 --output_dir "{model_fname}"
                 --model_name "{ir_model_name}"
                 """
mo_command = " ".join(mo_command.split())

print("Exporting TensorFlow model to IR... This may take a few minutes.")
os.system(mo_command)

model_xml = f"{model_fname}/flower_ir.xml"

# Load network to the plugin
ie = IECore()
net = ie.read_network(model=model_xml)
exec_net = ie.load_network(network=net, device_name="CPU")
del net
input_layer = next(iter(exec_net.input_info))
output_layer = next(iter(exec_net.outputs))

OUTPUT_DIR = "output"
inp_file = "./test.jpg"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def pre_process_image(imagePath, img_height=180):
    # Model input format
    n, c, h, w = [1, 3, img_height, img_height]
    image = Image.open(imagePath)
    image = image.resize((h, w), resample=Image.BILINEAR)

    # Convert to array and change data layout from HWC to CHW
    image = np.array(image)
    image = image.transpose((2, 0, 1))
    input_image = image.reshape((n, c, h, w))

    return input_image


class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
input_image = pre_process_image(inp_file)
res = exec_net.infer(inputs={input_layer: input_image})
res = res[output_layer]
score = tf.nn.softmax(res[0])

image = input_image[0].transpose((1, 2, 0))
image = Image.fromarray(image.astype('uint8'), 'RGB')
plt.imshow(image)
plt.show()
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
