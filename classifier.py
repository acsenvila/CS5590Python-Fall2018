import keras
from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)
word_index = reuters.get_word_index()

print(len(x_train))
print(len(x_test))

print(len(y_train))
print(len(y_test))

num_classes = max(y_train)+1
print(num_classes)

print(x_train[0])
print(y_train[0])

print(word_index['the'])

index_to_word = {}

for key, value in word_index.items():
    index_to_word[value] = key

print(' '.join(index_to_word[x] for x in x_train[9]))

from keras_preprocessing.text import Tokenizer
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

print(x_train.shape)
print(x_train[0])

print(y_train.shape)
print(y_train[0])

from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation

model = Sequential()
model.add(Dense(512, input_shape=(max_words, )))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.metrics_names)

batch_size = 32
epochs = 5

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

print("test loss: {}" .format(score[0]))
print("test accuracy {}". format(score[1]))