import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np
print(tf.__version__)


# ================= Download the IMDB dataset ==================
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True
)

# ================== Try the Encoder ==========================
encoder = info.features['text'].encoder
print ('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print ('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print ('The original string: "{}"'.format(original_string))

assert original_string == sample_string

for ts in encoded_string:
  print ('{} ----> {}'.format(ts, encoder.decode([ts])))

# =================== Explore the Data =======================
for train_example, train_label in train_data.take(1):
    print('Encoded text:', train_example[:10].numpy())
    print('Label:', train_label.numpy())
    print(encoder.decode(train_example))

# =================== Prepare the Data for Training ==========
BUFFER_SIZE = 1000

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32, ([None],())))

test_batches = (
    test_data
    .padded_batch(32, ([None],())))

for example_batch, label_batch in train_batches.take(2):
  print("Batch shape:", example_batch.shape)
  print("label shape:", label_batch.shape)

# ================== Build the Model =========================

