from typing import List

import tensorflow as tf
from tensorflow import keras

from processing.text_to_data import BagOfWords


class EmbeddingModel:

    def __init__(self, vocab_size: int = 10_000, tokenizer=None):
        self.bow = BagOfWords(vocab_size=vocab_size, tokenizer=tokenizer)
        self._vocab_size = vocab_size
        self.model = None

    def load_newline_txt(self, label: str, filepath: str, encoding: str = 'utf-8'):
        self.bow.load_newline_txt(label, filepath, encoding=encoding)

    def load_model(self, model_path: str, vocab_path: str, labels: List[str]):
        self.model = self._create_model()
        self.model.load_weights(model_path)
        self.bow.load_vocab_labels(vocab_path, labels)

    def save_model(self, model_name: str):
        if self.model:
            self.bow.save_vocab('{}.pkl'.format(model_name))
            self.model.save('{}.h5'.format(model_name))

    def _create_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(self._vocab_size, 16))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        return model

    def _unpack_data(self, test_split: float):
        train, test = self.bow.test_train_split(test_split=test_split)
        return train[0], train[1], test[0], test[1]

    @staticmethod
    def _validation_split(train_data, train_labels, validation_split: int):
        x_val, partial_x_train = train_data[:validation_split], train_data[validation_split:]
        y_val, partial_y_train = train_labels[:validation_split], train_labels[validation_split:]
        return x_val, partial_x_train, y_val, partial_y_train

    def train_model(self, validation_split: int, padding: int = 256, test_split: float = 0.1):
        self.model = self._create_model()
        self.bow.prepare_data()
        train_data, train_labels, test_data, test_labels = self._unpack_data(test_split)
        train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=0, padding='post', maxlen=padding)
        test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=0, padding='post', maxlen=padding)
        x_val, partial_x_train, y_val, partial_y_train = self._validation_split(train_data, test_labels,
                                                                                validation_split)
        self.model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val),
                       verbose=1)
        results = self.model.evaluate(test_data, test_labels)
        print('Loss: {}'.format(results[0]))
        print('Accuracy: {}'.format(results[1]))

    def predict(self, sentence: str):
        if not self.model:
            raise AttributeError('No model trained or loaded')
        sent = self.bow.sentence_to_data(sentence)
        result = self.model.predict(sent)
        prob, label_int = result[0][0], result[1][0]
        label = sorted(self.bow._labels)[int(label_int)]
        return prob, label


if __name__ == '__main__':
    model = EmbeddingModel()
    #model.load_newline_txt('russian', 'russian.txt')
    #model.load_newline_txt('bulgarian', 'bulgarian.txt')
    #model.train_model(5000)
    #model.save_model('my_model')
    model.load_model('my_model.h5', 'my_model.pkl', ['russian', 'bulgarian'])
    result = model.predict('Привет меня зовут')
    print(result)
    result = model.predict('Автобус се заби на спирка в Пловдив, шофьорът издъхна')
    print(result)
