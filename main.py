# installing normal libraries
import cv2
import numpy as np
import os
import uuid
import random
from matplotlib import pyplot as plt


# installing tensorflow libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


POS_PATH = os.path.join("data", "positive")
NEG_PATH = os.path.join("data", "negative")
ANC_PATH = os.path.join("data", "anchor")

# make directories
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

# moving files to negative folder

# for directory in os.listdir("lfw-main"):
#     for file in os.listdir(os.path.join("lfw-main", directory)):
#         EX_PATH = os.path.join("lfw-main", directory, file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH, NEW_PATH)


####

# use camera to take a picture

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()

#     # cut frame
#     frame = frame[500 : 500 + 250, 700 : 700 + 250, :]

#     # collect anchor image
#     if cv2.waitKey(1) & 0xFF == ord("a"):
#         imgname = os.path.join(ANC_PATH, "{}.jpg".format(uuid.uuid1()))
#         cv2.imwrite(imgname, frame)

#     # collect positive image
#     if cv2.waitKey(1) & 0xFF == ord("p"):
#         imgname = os.path.join(POS_PATH, "{}.jpg".format(uuid.uuid1()))
#         cv2.imwrite(imgname, frame)

#     cv2.imshow("Image collection", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()

# cv2.destroyAllWindows()

# take pictures from folders

anchor = tf.data.Dataset.list_files(ANC_PATH + "/*.jpg").take(50)
positive = tf.data.Dataset.list_files(POS_PATH + "/*.jpg").take(50)
negative = tf.data.Dataset.list_files(NEG_PATH + "/*.jpg").take(50)

dir_test = anchor.as_numpy_iterator()

# print(dir_test.next())


def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img


img = preprocess("data/anchor/e2228406-4f3a-11ef-8f5e-22772e7ac823.jpg")

# img_np = img.numpy()

# # Display the image
# plt.imshow(img_np)
# plt.show()


positives = tf.data.Dataset.zip(
    (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor))))
)
negatives = tf.data.Dataset.zip(
    (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor))))
)
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()
example = samples.next()

# print(example)


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


res = preprocess_twin(*example)


# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

samples = data.as_numpy_iterator()
samp = samples.next()

# plt.imshow(samp[1])
# plt.show()

# print(samp[2])

# Training partition
train_data = data.take(round(len(data) * 0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# # Testing partition
test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


def make_embedding():
    inp = Input(shape=(100, 100, 3), name="input_image")

    # First block
    c1 = Conv2D(64, (10, 10), activation="relu")(inp)
    m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation="relu")(m1)
    m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation="relu")(m2)
    m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation="relu")(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation="sigmoid")(f1)

    return Model(inputs=[inp], outputs=[d1], name="embedding")


embedding = make_embedding()


# siamese L1Distance class
# Siamese L1 Distance class
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__()

    def call(self, input_embedding, validation_embedding):
        # Ensure that the inputs are tensors
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model():

    # Anchor image input in the network
    input_image = Input(name="input_img", shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name="validation_img", shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = "distance"
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation="sigmoid")(distances)

    return Model(
        inputs=[input_image, validation_image],
        outputs=classifier,
        name="SiameseNetwork",
    )


siamese_model = make_siamese_model()

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


@tf.function
def train_step(batch):
    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)

        # Reshape y to match the shape of yhat
        y = tf.reshape(y, yhat.shape)

        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss


def train(data, EPOCHS):
    # loop through epochs
    for epoch in range(1, EPOCHS + 1):
        print("\n Epoch {}/{}".format(epoch, EPOCHS))
        progrbar = tf.keras.utils.Progbar(len(train_data))

        # loop through batches
        for idx, batch in enumerate(train_data):
            # run training step
            train_step(batch)
            progrbar.update(idx + 1)

        # save checkpoint
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


EPOCHS = 50

# train(train_data, EPOCHS)


