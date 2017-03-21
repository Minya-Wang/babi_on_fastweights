#coding:utf-8

import os
import numpy as np
import _pickle as cPickle
import sys


def parse(data_dir, t):

    def data_path(data_dir):
        test_data_path = []
        train_data_path = []
        for filename in os.listdir(data_dir):
            if filename[-8:-4] == "test":
                test_data_path.extend([data_dir + filename])
            else:
                train_data_path.extend([data_dir + filename])
        return train_data_path, test_data_path

    train_data_path, test_data_path = data_path(data_dir)


    t = int(t)  #要解析的数据

    train_data_path = [train_data_path[t]]
    test_data_path = [test_data_path[t]]
    print(train_data_path, test_data_path)

    dict = {}
    dict['nil'] = 1

    include_question = False
    story = np.zeros([20, 1000, 1000])
    story_ind = -1
    sentence_ind = -1
    max_words = 0
    max_sentences = 0

    questions = np.zeros([10, 1000])
    question_ind = -1

    qstory = np.zeros([20, 1000])

    fi = 0    #先从第一个文件开始
    fd = open(train_data_path[fi])
    print(train_data_path[fi])
    line_ind = 0

    while True:
        line = fd.readline()
        # print(line)
        if line == '':
            fd.close()
            if fi < len(train_data_path)-1:
                fi += 1
                fd = open(train_data_path[fi])
                line_ind = 0
                line = fd.readline()
            else:
                break

        line_ind += 1
        words = line.split()

        if words[0] == '1':
            story_ind += 1
            sentence_ind = -1
            map = []

        if line[-2] == '.':
            is_question = False
            sentence_ind += 1
        else:
            is_question = True
            question_ind += 1
            questions[0, question_ind] = story_ind
            questions[1, question_ind] = sentence_ind
            if include_question:
                sentence_ind += 1

        map.append(sentence_ind)

        for k in range(1, len(words)):
            w = words[k]
            w = w.lower()
            if w[-1] == '.' or w[-1] == '?':
                w = w[0:-1]
            if w not in dict:
                dict[w] = len(dict) + 1

            max_words = max(max_words, k)

            if is_question is False:
                story[k-1, sentence_ind, story_ind] = dict[w]
            else:
                qstory[k-1, question_ind] = dict[w]
                if include_question is True:
                    story[k-1, sentence_ind, story_ind] = dict[w]

                if words[k][-1] == '?':

                    answer = words[k+1]
                    if answer not in dict:
                        dict[answer] = len(dict) + 1

                    questions[2, question_ind] = dict[answer]
                    for h in range(k+2, len(words)):
                        questions[1+h-k, question_ind] = map[int(words[h])-1]
                    questions[9, question_ind] = line_ind
                    break
        max_sentences = max(max_sentences, sentence_ind)

    story = story[0:max_words, 0:max_sentences+1, 0:story_ind+1]
    questions = questions[:, 0:question_ind+1]
    qstory = qstory[0:max_words, 0:question_ind+1]

    story[story == 0] = dict['nil']
    qstory[qstory == 0] = dict['nil']

    # print(story[:, :, 0])
    # print(questions[:, 0:20])
    # print(qstory[:, 0:10])

    return story, questions, qstory, dict


def generate_epoch(X, y, num_epochs, batch_size):  #num_epoch是遍历几趟数据

    for epoch_num in range(num_epochs):
        yield generate_batch(X, y, batch_size)


def generate_batch(X, y, batch_size): #batch_size是一遍数据要分成num_batches份，每份batch_size的大小

    data_size = len(X)

    num_batches = (data_size // batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield X[start_index:end_index], y[start_index:end_index] #产生一个X和Y的访问序列

if __name__ == "__main__":

    story, questions, qstory, dict = parse('en/', sys.argv[1])
    print(len(dict))
    print(sorted(dict.items(), key=lambda d: d[1]))
    # input("==========")
    print("story.shape=", story.shape)
    print("questions.shape", questions.shape)
    print("qstory.shape", qstory.shape)

    X = []
    for i in range(questions.shape[1]):

        #取出sentence和qstory，并连接，放在一个list中
        X_i = (story[:, 0:int(questions[1, i])+1, int(questions[0, i])])  #一个question前面的句子
        X_i = np.c_[X_i, qstory[:, i]] #加入question
        X_i = np.transpose(X_i).reshape((1, -1))[0]
        X_i = [int(each) for each in X_i]

        #填充
        max_words = story.shape[0]
        max_sentence = story.shape[1]
        input_dim = max_words * max_sentence + max_words * 1   #sentence的单词数和question的单词数
        padding = [1 for padding_num in range(input_dim - len(X_i))]
        padding.extend(X_i)
        X_i = padding

        # one-hot
        X_i = np.eye(len(dict)+1)[np.array(X_i).astype('int')]  #对应的词为index,[1,0,...,0]没有对应的单词，从1开始对应
        X_i = X_i.tolist()
        # print(X_i)
        # input("=======")

        #放在X中
        X.append(X_i)
        # print(X)
        # input("===========")
    X = np.array(X)
    #答案
    y = questions[2, :]  # dict(answer)
    y = np.eye(len(dict)+1)[np.array(y).astype('int')]
    # y = y.tolist()

    print("X.shape = ", np.shape(X), type(X))
    print("y.shape = ", np.shape(y), type(y))
    #拆分数据
    train_X = X[0: int(len(X)*0.8)] #X为列表，列表中的每个元素是一个QA对，QA对的数量与question的数量一致,是拥有1000个元素的list
    train_y = y[0: int(len(X)*0.8)] #y为array，维度是1*len(X)，每个元素是一个标量值，表示dict(answer)
    valid_X = X[int(len(X)*0.8):]
    valid_y = y[int(len(X)*0.8):]

    # Save data into pickle files
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/train.p', 'wb') as f:
        cPickle.dump([train_X, train_y], f)
    with open('data/valid.p', 'wb') as f:
        cPickle.dump([valid_X, valid_y], f)