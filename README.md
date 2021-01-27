# MCBSL
## Sentiment_Lexicon generate method for specific domin. --Based on muti-domin corpus.

> ## 项目结构
>- /corpus：训练语料  
>- /reference： 参考资料
>>- /output：各个过程的输出结果  
>>- /wc_output：词向量训练结果  
>- /validation：不同的文本表示方法进行验证  
>>- bert.py 使用预训练bert(bert-with-service必须 & tensorflow==1.1x)
>>- one_hot.py 使用one_hot进行文本表示(基于情感词典)
>>- classify.py 使用MCBSL、词向量进行文本表示。
>- /test 测试文件夹
>- sopmi.py 提供sopmi值的计算
>- run.py 项目入口
>- utils.py 主要业务逻辑
 
