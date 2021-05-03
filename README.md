# MFDSL
### Multi-source Knowledge Fusion Sentiment Lexicon

> ## 项目结构
>- /corpus：训练语料 
>>- /amazon 英文实验数据，5领域亚马逊评论语料
>>- /hotel 中文实验数据，谭松波酒店评论语料
>>- NYT_comment_cut.txt 纽约时报评论语料(分词后)
>- /reference： 参考资料
>>- /output：各个过程的输出结果  
>>- /wc_output：词向量训练结果  
>- /validation：不同的文本表示方法进行验证  
>>- bert.py 使用预训练bert(bert-with-service必须 & tensorflow==1.1x)
>>- one_hot.py 使用one_hot进行文本表示(基于情感词典)
>>- classify.py 使用MFDSL-、词向量进行文本表示。
>- /test 测试文件夹
>- sopmi.py 提供sopmi值的计算
>- run.py 项目入口
>- utils.py 主要业务逻辑

## 说明
生成情感词典方法
```
!python run.py --corpus ./corpus/amazon/dvd/all_cut.txt 
               --weight True --weight_schema mix 
               --select_seeds True --dimension 50 
               --model fasttext --language en
```
参数：  
>corpus 指定分词后的语料  
>weight 是否重新计算权重  
>weight_schema 权重计算方式[tf-idf, mix]  
>select_seeds 是否重新选择种子词  
>dimension 种子词的一半维度 dimension=x即正负种子词各x  
>model 相似度计算所使用的词向量模型 [word2vec fasttext]  
>language 语料的语言[en, zh]

输出结果：./reference/output/sv.pkl   
输出结果为<情感词, 向量>的字典: [str, List[]]，可以使用pickle包直接读取
 
