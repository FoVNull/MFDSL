from typing import Dict, Any

import tensorflow as tf
from kashgari_local.abc_feature_model import ABCClassificationModel
from kashgari.layers import L


class XLNet_linear(ABCClassificationModel):
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
            }
        }

    def build_model_arc(self):
        features = tf.keras.Input(shape=(None, self.feature_D), name="features")

        l1_reg = tf.keras.regularizers.l1(0.01)
        l2_reg = tf.keras.regularizers.L2(0.01)

        output_dim = self.label_processor.vocab_size
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model
        
        tensor = embed_model.output
        layer_stack = [
            L.LSTM(256, return_sequences=False)
        ]
        for layer in layer_stack:
            tensor = layer(tensor)
        output_tensor = L.Dense(
            output_dim, activation='sigmoid', name="output0",
            kernel_regularizer=l2_reg
        )(tensor)
        self.tf_model = tf.keras.Model(inputs=[embed_model.inputs, features], outputs=output_tensor)
