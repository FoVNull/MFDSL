
'''
author: FoVNull
@hikki.top
based on huggingface@https://huggingface.co/transformers/model_doc/xlnet.html
'''

from typing import Dict, List, Any, Optional
import os
import json

from transformers import TFXLNetModel, XLNetTokenizer
import tensorflow as tf

from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.logger import logger


class XLNetEmbedding(ABCEmbedding):
    def to_dict(self) -> Dict[str, Any]:
        info_dic = super(XLNetEmbedding, self).to_dict()
        info_dic['config']['xlnet_path'] = self.xlnet_path
        return info_dic

    def __init__(self,
                 xlnet_path: str,
                 corpus_gen,
                 **kwargs: Any):
        """

        Args:
            config_path: model config path, example `config.json`
            spiece_path: SentencePiece Model, example 'spiece.model'
            corpus_gen：参与任务的语料，XLNet中用spiece model替代词表，所以需要根据语料进行分词，类型为生成器
            使用huggingface transformers读取XLNet，注意保证格式正确
            @https://huggingface.co/models
            kwargs: additional params
        """
        self.xlnet_path = xlnet_path
        self.config_path = os.path.join(xlnet_path, 'config.json')
        self.spiece_path = os.path.join(xlnet_path, 'spiece.model')
        self.corpus_gen = corpus_gen
        self.vocab_list: List[str] = []
        kwargs['segment'] = True
        super(XLNetEmbedding, self).__init__(**kwargs)

    # XLNet Tokenizer
    def load_embed_vocab(self) -> Optional[Dict[str, int]]:
        token2idx: Dict[str, int] = {}
        tokenizer = XLNetTokenizer.from_pretrained(self.xlnet_path)
        
        vocab_set = set()
        for line in self.corpus_gen:
            for token in tokenizer.tokenize(line):
                vocab_set.add(token)
        self.vocab_list = ['[CLS]', '[SEP]', '[PAD]', '[MASK]'] + list(vocab_set)
        for idx, token in enumerate(self.vocab_list):
            token2idx[token] = idx

        # top_words = [k for k, v in list(token2idx.items())[:50]]
        logger.debug('------------------------------------------------')
        logger.debug("Loaded spiece model's vocab")
        logger.debug(f'config_path       : {self.config_path}')
        logger.debug(f'spiece_model_path      : {self.spiece_path}')
        logger.debug('------------------------------------------------')

        return token2idx

    def build_embedding_model(self,
                              *,
                              vocab_size: int = None,
                              force: bool = False,
                              **kwargs: Dict) -> None:
        if self.embed_model is None:
            xlnet_model = self.create_xlnet()
            for layer in xlnet_model.layers:
                layer.trainable = False
            self.embed_model = xlnet_model
            self.embedding_size = xlnet_model.output.shape[-1]

    def create_xlnet(self):
        inputs = tf.keras.Input(shape=(None, ), name='data2', dtype='int32')
        targets = tf.keras.Input(shape=(None, ), name='output', dtype='int32')

        xlnet = TFXLNetModel.from_pretrained(self.xlnet_path)
        xlnet_encodings = xlnet(inputs).hidden_states[2]

        model = tf.keras.Model(inputs=[inputs, targets], outputs=[xlnet_encodings])

        return model
