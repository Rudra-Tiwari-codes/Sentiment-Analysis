%matplotlib inline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random

def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
def show_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()

    
print('Using TensorFlow version', tf.__version__)

dataset = nlp.load_dataset('emotion')

dataset
train = dataset['train']
val = dataset['validation']
test = dataset['test']


def get_tweet(data):
  tweets = [x['text'] for x in data]
  labels = [x['label']for x in data]
  return tweets, labels

tweets, labels = get_tweet(train)

tweets[1], labels[1] 
from tensorflow.keras.preprocessing.text  import Tokenizer

tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)

tokenizer.texts_to_sequences([tweets[0]])
tweets[0]

lengths = [len(t.split(' ')) for t in tweets]
plt.hist(lengths, bins = len(set(lengths)))
plt.show()

maxlen = 50

from tensorflow.keras.preprocessing.sequence import pad_sequences
def get_sequences (tokenizer,tweets):
  sequences = tokenizer.texts_to_sequences(tweets)
  padded = pad_sequences(sequences, truncating = 'post', padding= 'post', maxlen=maxlen)
  return padded

padded_train_seq = get_sequences(tokenizer, tweets)
padded_train_seq[0]


classes = set(labels)
print(classes)

plt.hist(labels, bins =11)
plt.show

class_to_index = dict((c,i) for i, c in enumerate(classes))
index_to_class = dict((v, k) for k, v in class_to_index.items())

class_to_index
index_to_class

names_to_ids = lambda labels: np.array([class_to_index.get(x) for x in labels])
train_labels = names_to_ids(labels)
print(train_labels[0])
model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000, 16, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),   
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(6,activation='softmax')          
])

model.compile(
    
    loss='sparse_categorical_crossentropy',
    optimizer= 'adam',
    metrics=['accuracy']

)

model.summary()

val_tweets, val_labels = get_tweet(val)
val_seq = get_sequences(tokenizer, val_tweets)
val_labels = names_to_ids(val_labels)

val_tweets[0], val_labels[0]

h = model.fit(
    padded_train_seq, train_labels,
    validation_data= (val_seq, val_labels),
    epochs =20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor= 'val_accuracy', patience = 2)
    ]
)

show_history(h)
test_tweets, test_labels = get_tweet(test)
test_seq = get_sequences(tokenizer, test_tweets)
test_labels = names_to_ids(test_labels)

_ = model.evaluate(test_seq, test_labels)

i = random.randint(0, len(test_labels) - 1)

print ('Sentence:', test_tweets[i])
print ("Emotion:", index_to_class[test_labels[i]])

p= model.predict(np.expand_dims(test_seq[i], axis=0))[0]
pred_class = index_to_class[np.argmax(p).astype('uint8')]

print ('Predicted Emotion:', pred_class)


preds_x = model.predict(test_seq)
classes_x = np.argmax(preds_x, axis =1)

# preds = model.predict_classes(test_seq)

show_confusion_matrix(test_labels, classes_x, list(classes))


