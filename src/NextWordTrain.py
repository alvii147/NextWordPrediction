import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import random

# --------------------------------
# Extracting news articles dataset
# --------------------------------
DATA = []
categories = ["business", "entertainment", "politics", "sport", "tech"]
sentence_data = []

for cat in categories:
    for filename in os.listdir("../dataset/bbc_articles/" + cat + '/'):
        with open("../dataset/bbc_articles/" + cat + '/' + filename, 'r') as datafile:
            for line in datafile.readlines():
                for sentence in line.split('.'):
                    if sentence.strip():
                        sentence_data.append(sentence.strip())

# --------------------------------------------------
# Randomly selecting 1000 sentences to train model
# Ideally I would've liked to use the entire dataset
# But my laptop did not have enough GPU power
# --------------------------------------------------
sentence_data = random.choices(sentence_data, k = 1000)

# -------------------------------------------------------
# Setting up dataset of words and their preceding 5 words
# -------------------------------------------------------
num_prev_words = 5
sentence_prev_data_flattened = []
for data in sentence_data:
    split_data = data.split()
    for i in range(len(split_data) - num_prev_words):
        sentence_prev_data_flattened.append(' '.join(split_data[i:i + num_prev_words]))
        sentence_prev_data_flattened.append(split_data[i + num_prev_words])
# -------------------------------------------------------------------------
# Converting words from dataset to vectors using Universal Sentence Encoder
# -------------------------------------------------------------------------
url = "https://tfhub.dev/google/universal-sentence-encoder/1"
embed = hub.Module(url)

tf.logging.set_verbosity(tf.logging.ERROR)
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    sentence_embeddings = session.run(embed(sentence_prev_data_flattened))

x_train = []
y_train = []
for i, sentence_embedding in enumerate(np.array(sentence_embeddings).tolist()):
    if i % 2:
        y_train.append([x for x in sentence_embedding])
    else:
        x_train.append([x for x in sentence_embedding])

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# --------------------------
# Setting up 3-layered model
# --------------------------
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation = "relu"))
model.add(tf.keras.layers.Dense(1024, activation = "relu"))
model.add(tf.keras.layers.Dense(512, activation = "softmax"))

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["acc"])

# ------------------------
# Fitting and saving model
# ------------------------
history = model.fit(x_train, y_train, epochs = 5, validation_split = 0.1, shuffle = True, batch_size = 1)
model.save("nextword.model")