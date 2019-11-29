train = data.load_pos_data('data/train')
test = data.load_pos_data('data/test')
vocab = data.load_vocabulary()
pos_vocab = data.pos_vocabulary

import numpy as np
from week_8.backstage.data import *

## Hint
sample = train[0]
word_ids = [vocab[word] for word in sample.text]
tag_ids = [pos_vocab[tag] for tag in sample.labels]

print(sample.text, word_ids)
print(sample.labels, tag_ids)

train_x = [[vocab[w] for w in s.text] for s in train]
train_y = [[pos_vocab[w] for w in s.labels] for s in train]
test_x = [[vocab[w] for w in s.text] for s in test]
test_y = [[pos_vocab[w] for w in s.labels] for s in test]


train_x = keras.preprocessing.sequence.pad_sequences(train_x, padding='post')
train_y = keras.preprocessing.sequence.pad_sequences(train_y, padding='post')
test_x = keras.preprocessing.sequence.pad_sequences(test_x, padding='post')
test_y = keras.preprocessing.sequence.pad_sequences(test_y, padding='post')

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

# TRAINING
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.initializers import Constant
from datetime import datetime

class POSTagger(keras.Model):

    def __init__(self):
        super(POSTagger, self).__init__()
        self.emb = Embedding(
            input_dim=len(vocab), 
            output_dim=300, 
            mask_zero=True,
            weights=[data.embedding_matrix(vocab)]
        )
        self.lstm = Bidirectional(LSTM(300, return_sequences=True))
        self.dense = Dense(17, activation='softmax')

    def call(self, inputs):
        x = self.emb(inputs)
        mask = self.emb.compute_mask(inputs)
        x = self.lstm(x, mask=mask)
        output = self.dense(x)
        return output


model = POSTagger()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


logdir = "logs/bi-lstm" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
model.fit(
    x=train_x,
    y=train_y,
    batch_size=10,
    epochs=30,
    validation_data=(test_x, test_y),
    callbacks=[tensorboard_callback]
)
