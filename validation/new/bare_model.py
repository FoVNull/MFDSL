from typing import Dict, Any

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from kashgari_local.abc_feature_model import ABCClassificationModel
from kashgari.layers import L


class Bare_Model(ABCClassificationModel):
    def __init__(self, embedding, **params):
        super().__init__(embedding)
        self.feature_D = params["feature_D"]

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict

        activation_function list:
        {softmax, elu, selu, softplus, softsign, swish,
        relu, gelu, tanh, sigmoid, exponential,
        hard_sigmoid, linear, serialize, deserialize, get}
        """
        return {
            'layer_bilstm1': {
                'units': 128,
                'return_sequences': True
            },
            'layer_time_distributed': {},
            'conv_layer1': {
                'filters': 128,
                'kernel_size': 4,
                'padding': 'valid',
                'activation': 'relu'
            },
            'layer_output1': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        """
        build model architectural

        BiLSTM + Convolution + Attention
        """
        features = tf.keras.Input(shape=self.feature_D, name="features")
        print(features)
        exit(99)
        l1_reg = tf.keras.regularizers.l1(0.01)
        l2_reg = tf.keras.regularizers.L2(0.01)

        output_dim = self.label_processor.vocab_size
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model
        # Define layers for BiLSTM
        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_bilstm1'])),
            L.Dropout(rate=0.2),
        ]

        # tensor flow in Layers {tensor:=layer(tensor)}
        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        '''
        define attention layer
        as a nlp-rookie im wondering whether this is a right way XD
        '''
        # query_value_attention_seq = L.Attention()([tensor, tensor])
        query_value_attention_seq = L.MultiHeadAttention(
            num_heads=4, key_dim=2, dropout=0.5
        )(tensor, tensor)

        query_encoding = L.GlobalMaxPool1D()(tensor)
        query_value_attention = L.GlobalMaxPool1D()(query_value_attention_seq)

        tensor_2d = L.Concatenate(axis=-1)([query_encoding, query_value_attention])

        # without multi-head-att
        # input_tensor = L.GlobalMaxPool1D()(tensor)

        # extend features
        features_tensor = L.Dense(64, kernel_regularizer=l1_reg)(features)
        input_tensor = L.Concatenate(axis=-1)([features_tensor, tensor_2d])

        # output tensor
        input_tensor = L.Dropout(rate=0.1)(input_tensor)
        output_tensor = L.Dense(
            output_dim, activation='sigmoid', name="output0",
            kernel_regularizer=l2_reg
        )(input_tensor)
        self.tf_model = tf.keras.Model(inputs=[embed_model.inputs, features], outputs=output_tensor)

        # plot_model(self.tf_model, to_file="D:/PycProject/TripleC/reference/model.png")
