texts: 2d list [['词1','词2','词3',...],[....],....]
labels:[1,0,1,.....] (same size of texts)
labels_index: {'a': 0, 'b': 1}
index_lables: {0: 'a', 1: 'b'}

### 参数  

```python
args['vocb_size'] = nb_words   # 固定词表的大小，用于定义词向量矩阵的行数
args['max_len'] = max_len     # 文章最大长度（词数量）
args['n_class'] = n_class   # 类别数
args['dim'] = word_dim     # 词向量维度
args['embedding_matrix'] = torch.Tensor(embedding_matrix) # 词向量矩阵
```

### 重要中间变量

* 词表 : `word_vocb`   训练语料中所有词的 1d 列表（set）

* 词表与词索引的map   : `word_to_idx`   {"词"：索引}

  ​                                     `idx_to_word`     {"索引"：词}   

  

* `embedding_matrix`  与词表中的词索引对应的词向量矩阵   

  ​                                         <font color='pink'>2d list ,  shape: { nb_words, dim}  (固定词表大小 X 词向量维度) </font>

* `texts_with_id`  以索引表示的全部训练数据：

  ​                                  <font color='pink'>2d list ,  shape: {num_texts , max_len}  (文档数量 X 每篇的最大词数量)   </font>

  每篇文章中的每个词对应一个索引，即为该词的词表索引，通过这个索引能找到对应的词向量   

  每行为一篇文章：[ 词索引0，词多因1，.... 词索引 m ]  (m=max_len)

  

#### textCNN 的结构

1. 第一层是 embeding 层  ，将索引表示的训练数据转化为真实词向量

   输入：以索引表示训练数据，从`texts_with_id` 里分批取的批量数据，

   ​                                               shape: {batch_size,max_len}

   输出：批量数据对应的词向量 3d Tensor 

    <font color='pink'> shape : {batch_size, max_len, word_dim} (批次大小 X 每篇文章的最大词数量 X 词向量维度)  </font> 

   

2. 把第一层的输出展开，torch.view ()  , 维度变为    

   <font color='pink'> {batch_size, in_channel, max_len, word_dim}  </font>  in_channel: 输入通道数（默认为1） ，后面两个维度 max_len, word_dim 可以类比图片的宽和高 w x h

3. 四步卷积  ，每步都包括：一层卷积，RELU激活，一层池化 max pool

4. 卷积结束后进行 Flatten  : 将（batch，outchanel,w,h) 展平为 (batch，$outchanel* w * h$）  

5. 最后一层为全连接， 输出 shape= (batch_size, n_classes)