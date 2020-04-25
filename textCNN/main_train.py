from sklearn.model_selection import train_test_split
import arrow
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from textCNN.pre_process import *
from textCNN.textCNN_model import TextCNN
from textCNN.Config import *

fileConfig=FileConfig()
cnnConfig=CnnConfig()
trainConfig=TrainConfig()


def data_process():
    start_time = arrow.now()

    # -- 数据预处理（读取，分词）
    segment_df, labels = data_read()
    # 保存标签
    label_df=pd.DataFrame(labels)
    label_df.to_csv(fileConfig.labels_filename, header=False, index=False)

    print('Total sample num: ', segment_df.shape[0])
    # 训练 w2v 并保存模型文件
    wordsList = [text.split(" ") for text in segment_df['segment_text']]  # 用于训练word2vec模型，2d list，每篇文章是一个词列表

    w2v_model = train_word2vec(wordsList, fileConfig.w2v_modelname)

    # 词表索引(词-索引 字典)，词向量矩阵
    word_to_idx, embedding_matrix = get_vocab_vectors(wordsList,w2v_model,cnnConfig.word_dim,cnnConfig.vocab_size)
    print("word_to_idx size: {} ; embedding_matrix size: {}".format(len(word_to_idx), embedding_matrix.shape))

    # 保存词表索引和词向量矩阵
    index_df = pd.DataFrame.from_dict(word_to_idx, orient='index', columns=['word'])
    index_df = index_df.reset_index().rename(columns={"index": "id"})
    index_df.to_csv(fileConfig.word_index_filename, header=True, index=False)

    w2v_df = pd.DataFrame(embedding_matrix)
    w2v_df.to_csv(fileConfig.w2v_filename, header=False, index=False)

    # 生成以索引表示的训练矩阵
    texts_with_id = get_train_matrix(wordsList, cnnConfig.max_len, word_to_idx)
    train_df = pd.DataFrame(texts_with_id)
    train_df.to_csv(fileConfig.train_data, header=False, index=False)
    print("whole input matrix size: ", texts_with_id.shape)

    print("\n Total data process time :{} ".format(arrow.now() - start_time))

    return embedding_matrix,texts_with_id,labels


def load_train_data():
    embedding_matrix = pd.read_csv(fileConfig.w2v_filename, header=None).values
    train_indexes = pd.read_csv(fileConfig.train_data, header=None).values.astype(np.int32)
    labels = pd.read_csv(fileConfig.labels_filename, header=None).values.astype(np.int32)[:, 1]
    labels = labels[1:]
    return embedding_matrix,train_indexes,labels


def config_model():
    # 定义 textCNN 模型
    args = {}
    # args['vocb_size'] = embedding_matrix.shape[0]
    args['vocb_size'] = cnnConfig.vocab_size  # 固定词表大小
    args['max_len'] = cnnConfig.max_len
    args['n_class'] = cnnConfig.n_class
    args['word_dim'] = cnnConfig.word_dim
    args['embedding_matrix'] = torch.Tensor(embedding_matrix)
    args['kernel_sizes']=cnnConfig.kernel_sizes
    args['channel_num'] = cnnConfig.channel_num
    args['drop_rate'] = cnnConfig.drop_rate

    textCNN = TextCNN(args)

    # print("textCNN model arguments: vocb_size: {} \t w2v_dim :{} \t max_len :{}\n model structure: {}"
    #       .format())
    return textCNN


def train_model(cnn_model,train_x,train_y):
    print("\n --------------Train model -------------- \n")
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=cnnConfig.LR,weight_decay=cnnConfig.L2_Reg)
    # 学习率衰减
    scheduler = lr_scheduler.StepLR(optimizer, cnnConfig.STEP_SIZE, cnnConfig.GAMMA)
    # 交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    total_step=0

    startTime_0 = arrow.now()
    startTime=arrow.now()
    for epoch in range(trainConfig.EPOCH):
        train_batch=trainConfig.train_batch_size
        for i in range(0, (int)(len(train_x) / train_batch)):
            b_x = Variable(torch.LongTensor(train_x[i * train_batch:i * train_batch + train_batch]))
            b_y = Variable(torch.LongTensor(train_y[i * train_batch:i * train_batch + train_batch]))
            output = cnn_model(b_x)
            loss = loss_function(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_step+=1
            if total_step%trainConfig.log_interval==0:
                pred_y = torch.max(output, 1)[1].data.squeeze()
                acc = (b_y == pred_y)
                acc = acc.numpy().sum()
                accuracy = acc / (b_y.size(0))
                print("Train step: {},\t loss: {},\t accuracy: {} \t Running time on {}"
                      .format(total_step,loss.data.item(),accuracy,arrow.now()-startTime_0))
                startTime_0=arrow.now()

        # 跑完 Epoch 后进行学习率衰减
        scheduler.step()
        print("Epoch {} , Learning rate: ".format(epoch,optimizer.param_groups))

    torch.save(cnn_model.state_dict(), fileConfig.model_filename)
    print("Training completed. Total running time: ", arrow.now()-startTime)
    return cnn_model


def evaluate_model(cnn_model,test_x,test_y):
    print("\n --------------Evaluate model -------------- \n")
    total_acc=0.0
    cnn_model.eval()
    test_batch=trainConfig.test_batch_size
    for i in range(0, (int)(len(test_x) / test_batch)):
        b_x = Variable(torch.LongTensor(test_x[i * test_batch:i * test_batch + test_batch]))
        b_y = Variable(torch.LongTensor(test_y[i * test_batch:i * test_batch + test_batch]))
        output=cnn_model(b_x)

        pred_y = torch.max(output, 1)[1].data.squeeze()
        acc = (b_y == pred_y).numpy().sum()
        total_acc+=acc

    avg_acc=total_acc/test_x.shape[0]
    print('Average testing accuracy: ',avg_acc)


if __name__=="__main__":
    # 数据预处理
    embedding_matrix, train_indexes, labels=data_process()

    # 加载预处理后的训练数据
    # embedding_matrix, train_indexes, labels=load_train_data()

    textCNN=config_model()
    # 划分训练数据和测试数据
    x_train, x_test, y_train, y_test = train_test_split(train_indexes, labels, test_size=0.2, random_state=42)
    print("Train sample num : {} \t Test sample num :{}".format(x_train.shape[0],x_test.shape[0]))

    start_time=arrow.now()
    trained_cnn=train_model(textCNN,x_train,y_train)
    print('Total training time{}'.format(arrow.now()-start_time))

    evaluate_model(textCNN, x_test, y_test)




