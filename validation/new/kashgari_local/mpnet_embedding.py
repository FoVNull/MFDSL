
'''
author: FoVNull
@hikki.top
based on huggingface@https://huggingface.co/transformers/model_doc/mpnet.html
'''

from typing import Dict, List, Any, Optional
import os
import json
import codecs

from transformers import MPNetTokenizer, TFMPNetModel
import tensorflow as tf

from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.logger import logger


class MPNetEmbedding(ABCEmbedding):
    def to_dict(self) -> Dict[str, Any]:
        info_dic = super(MPNetEmbedding, self).to_dict()
        info_dic['config']['mpnet_path'] = self.mpnet_path
        return info_dic

    def __init__(self,
                 mpnet_path: str,
                 **kwargs: Any):
        """

        Args:
            config_path: model config path, example `config.json`
            spiece_path: SentencePiece Model, example 'spiece.model'
            使用huggingface transformers读取MPNet，Tensorflow版本
            @https://huggingface.co/models
            kwargs: additional params
        """
        self.mpnet_path = mpnet_path
        self.config_path = mpnet_path + '/config.json'
        self.vocab_path =mpnet_path + '/vocab.txt'
        self.vocab_list: List[str] = []
        kwargs['segment'] = True
        super(MPNetEmbedding, self).__init__(**kwargs)

    # MPNet Tokenizer
    def load_embed_vocab(self) -> Optional[Dict[str, int]]:
        token2idx: Dict[str, int] = {}
        with codecs.open(self.vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.vocab_list.append(token)
                token2idx[token] = len(token2idx)
        top_words = [k for k, v in list(token2idx.items())[:50]]
        logger.debug('------------------------------------------------')
        logger.debug("Loaded mpnet's vocab")
        logger.debug(f'config_path       : {self.config_path}')
        logger.debug(f'vocab_path      : {self.vocab_path}')
        logger.debug(f'Top 50 words    : {top_words}')
        logger.debug('------------------------------------------------')

        return token2idx

    def build_embedding_model(self,
                              *,
                              vocab_size: int = None,
                              force: bool = False,
                              **kwargs: Dict) -> None:
        if self.embed_model is None:
            mpnet_model = self.create_mpnet()
            for layer in mpnet_model.layers:
                layer.trainable = False
            self.embed_model = mpnet_model
            self.embedding_size = mpnet_model.output.shape[-1]

    def create_mpnet(self):
        inputs = tf.keras.Input(shape=(None, ), name='data', dtype='int32')
        targets = tf.keras.Input(shape=(None, ), name='output', dtype='int32')

        mpnet = TFMPNetModel.from_pretrained(self.mpnet_path)
        mpnet_encodings = mpnet(inputs).hidden_states[3]

        model = tf.keras.Model(inputs=[inputs, targets], outputs=mpnet_encodings)

        return model
