import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

test_sentence = "Why is she in her"
data = []
data.append(test_sentence)

# --------------------------------
# Opening commonly used words list
# --------------------------------
with open("../dataset/words/words.txt", 'r') as wordsfile:
    for i in wordsfile.readlines():
        data.append(i.strip())

# ------------------------------------------------------------
# Converting words to vectors using Universal Sentence Encoder
# ------------------------------------------------------------
embed = hub.Module("../sentence_wise_email/module/module_useT")
tf.logging.set_verbosity(tf.logging.ERROR)
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embeddings = session.run(embed(data))
embeddings = np.array(embeddings)

# ---------------------
# Loading trained model
# ---------------------
loaded_model = tf.keras.models.load_model("nextword.model")

# --------------------------
# Testing using loaded model
# --------------------------
pred = loaded_model.predict(np.array([embeddings[0]]))

# ----------------------------------------------------------------------
# Using inner product to find similarity between result and common words
# ----------------------------------------------------------------------
inner_product = np.inner(pred, embeddings[1:])
print("Test Sentence: " + test_sentence)
print("Best Match: " + test_sentence + ' ' + data[np.argmax(inner_product[0]) + 1])