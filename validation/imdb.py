from transformers import TFXLNetModel
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_datasets as tfds


class XLNet_Model:
    def __init__(self, embedding_path):
        self.embeding_path = embedding_path
        self.model = self.build_model()

    def build_model(self):
        inputs = tf.keras.Input(shape=(None, ), name='text', dtype='int32')

        xlnet = TFXLNetModel.from_pretrained(self.embeding_path)
        xlnet_encodings = xlnet(inputs).hidden_states[2]

        layer_stack = [
            L.Bidirectional(L.LSTM(units=128)),
            L.Dropout(rate=0.2)
        ]
        tensor = xlnet_encodings
        for layer in layer_stack:
            tensor = layer(xlnet_encodings)

        # query_value_encodings = L.Attention()([tensor, tensor])
        output_tensor = L.Dense(units=2, activation='softmax', name='output')(tensor)

        model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
        )
        return model

    def fit(self, **args):
        self.model.fit()

    def corpus(self, data):
        for batch in data:
            yield batch[0], batch[1]


if __name__ == '__main__':
    vals_ds = tfds.load('imdb_reviews', split=[
        f'train[{k}%:{k + 10}%]' for k in range(0, 100, 10)
    ])
    trains_ds = tfds.load('imdb_reviews', split=[
        f'train[:{k}%]+train[{k + 10}%:]' for k in range(0, 100, 10)
    ])
    xlnet_model = XLNet_Model('')


