import _pickle as cPickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import sys
import random

import parse
from parse import (
    generate_epoch,
)

from model import (
    fast_weights_model,
)


class parameters():

    def __init__(self):

        self.input_dim = 88     #每个task的值不同，input_dim = max_words * max_sentence + max_words * 1
        self.num_classes = 28   #每个task的值不同，num_classes = dict(answer)

        self.num_epochs = 1000  #一个epoch是遍历一趟数据
        self.batch_size = 200    #一个batch是一趟数据的一部分

        self.num_hidden_units = 50
        self.l = 0.95 # decay lambda
        self.e = 0.5 # learning rate eta
        self.S = 1 # num steps to get to h_S(t+1)
        self.learning_rate = 1e-4
        self.learning_rate_decay_factor = 0.99 # don't use this decay
        self.max_gradient_norm = 5.0

        self.data_dir = 'data/'
        self.ckpt_dir = 'checkpoints/'
        self.save_every = max(1, self.num_epochs//4) # save every 500 epochs

        #story, _, _, dict = parse(self.data_dir)
        #self.input_dim = story.shape[0] * (story.shape[1]+1)
        # #一句话的最大单词数，乘一个story的最大句子数


def create_model(sess, FLAGS):

    fw_model = fast_weights_model(FLAGS)

    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Restoring old model parameters from %s" %
                             ckpt.model_checkpoint_path)
        fw_model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created new model.")
        sess.run(tf.initialize_all_variables())

    return fw_model


def train(FLAGS):
    """
    Train the model on the associative retrieval task.
    """

    # Load the train/valid datasets
    print("Loading datasets:")
    with open(os.path.join(FLAGS.data_dir, 'train.p'), 'rb') as f:
        train_X, train_y = cPickle.load(f)
        print("train_X:", np.shape(train_X), ",train_y:", np.shape(train_y))
    with open(os.path.join(FLAGS.data_dir, 'valid.p'), 'rb') as f:
        valid_X, valid_y = cPickle.load(f)
        print("valid_X:", np.shape(valid_X), ",valid_y:", np.shape(valid_y))

    with tf.Session() as sess:

        # Load the model
        model = create_model(sess, FLAGS)
        start_time = time.time()

        # Start training
        train_epoch_loss = []; valid_epoch_loss = []
        train_epoch_accuracy = []; valid_epoch_accuracy = []
        train_epoch_gradient_norm = []
        for train_epoch_num, train_epoch in enumerate(generate_epoch(
            train_X, train_y, FLAGS.num_epochs, FLAGS.batch_size)):
            print("EPOCH:", train_epoch_num)
            # Assign the learning rate
            sess.run(tf.assign(model.lr, FLAGS.learning_rate))

            #sess.run(tf.assign(model.lr, FLAGS.learning_rate))
            # Decay the learning rate
            #sess.run(tf.assign(model.lr, FLAGS.learning_rate * \
            #    (FLAGS.learning_rate_decay_factor ** epoch_num)))

            #if epoch_num < 1000:
            #    sess.run(tf.assign(model.lr, FLAGS.learning_rate))
            #elif epoch_num >= 1000: # slow down now
            #    sess.run(tf.assign(model.lr, 1e-4))

            # Custom decay (empirically decided)
            #if (epoch_num%1000 == 0):
            #    sess.run(tf.assign(model.lr,
            #        FLAGS.learning_rate/(10**(epoch_num//1000))))

            # Train set
            train_batch_loss = []
            train_batch_accuracy = []
            train_batch_gradient_norm = []
            for train_batch_num, (batch_X, batch_y) in enumerate(train_epoch):
                # batch_X: batch_size个QA对, array类型，维度：batch_size*max_words*(num_sentence+1)
                # batch_y: batch_size个数字，维度batch_size*1

                # FLAGS.input_dim = len(batch_X[0]) #下面3句可以不要
                #
                # batch_X = np.reshape(batch_X, [FLAGS.batch_size, FLAGS.input_dim, FLAGS.num_classes])
                # batch_y = np.reshape(batch_y, [FLAGS.batch_size, FLAGS.num_classes])

                h, w_softmax, b_softmax, logits, y, loss, accuracy, norm, _ = model.step(sess, batch_X, batch_y,
                    FLAGS.l, FLAGS.e, forward_only=False)
                # print("h:", h)  #0
                # print(np.sum(h!=0))
                # print(np.sum(logits != 0))
                # print("w_softmax", w_softmax)
                # print("b_softmax", b_softmax)
                # print("result:", tf.matmul(h, w_softmax) + b_softmax)
                # print("logits:", logits)
                # print("y:", y)
                # input("=======")
                train_batch_loss.append(loss)
                train_batch_accuracy.append(accuracy)
                train_batch_gradient_norm.append(norm)

            train_epoch_loss.append(np.mean(train_batch_loss))
            train_epoch_accuracy.append(np.mean(train_batch_accuracy))
            train_epoch_gradient_norm.append(np.mean(train_batch_gradient_norm))
            print('Epoch: [%i/%i] time: %.4f, loss: %.7f,'
                ' acc: %.7f, norm: %.7f' % (train_epoch_num, FLAGS.num_epochs,
                        time.time() - start_time, train_epoch_loss[-1],
                        train_epoch_accuracy[-1], train_epoch_gradient_norm[-1]))
#####################################################################################################
            # Validation set
            valid_batch_loss = []
            valid_batch_accuracy = []
            for valid_epoch_num, valid_epoch in enumerate(generate_epoch(
                valid_X, valid_y, num_epochs=1, batch_size=FLAGS.batch_size)):

                for valid_batch_num, (batch_X, batch_y) in enumerate(valid_epoch):

                    # FLAGS.input_dim = len(batch_X[0])
                    # batch_X = np.reshape(batch_X, [FLAGS.batch_size, FLAGS.input_dim, FLAGS.num_classes])
                    # batch_y = np.reshape(batch_y, [FLAGS.batch_size, FLAGS.num_classes])

                    loss, accuracy = model.step(sess, batch_X, batch_y,
                        FLAGS.l, FLAGS.e, forward_only=True)
                    valid_batch_loss.append(loss)
                    valid_batch_accuracy.append(accuracy)

            valid_epoch_loss.append(np.mean(valid_batch_loss))
            valid_epoch_accuracy.append(np.mean(valid_batch_accuracy))

            # Save the model
            if (train_epoch_num % FLAGS.save_every == 0 or
                train_epoch_num == (FLAGS.num_epochs-1)) and \
                (train_epoch_num > 0):
                if not os.path.isdir(FLAGS.ckpt_dir):
                    os.makedirs(FLAGS.ckpt_dir)
                checkpoint_path = os.path.join(FLAGS.ckpt_dir,
                    "%s.ckpt" % model_name)
                print("Saving the model.")
                model.saver.save(sess, checkpoint_path,
                                 global_step=model.global_step)

        plt.plot(train_epoch_accuracy, label='train accuracy')
        plt.plot(valid_epoch_accuracy, label='valid accuracy')
        plt.legend(loc=4)
        plt.title('%s_Accuracy' % FLAGS.model_name)
        plt.show()

        plt.plot(train_epoch_loss, label='train loss')
        plt.plot(valid_epoch_loss, label='valid loss')
        plt.legend(loc=3)
        plt.title('%s_Loss' % FLAGS.model_name)
        plt.show()

        plt.plot(train_epoch_gradient_norm, label='gradient norm')
        plt.legend(loc=4)
        plt.title('%s_Gradient Norm' % FLAGS.model_name)
        plt.show()

        # Store results for global plot
        with open('%s_results.p' % FLAGS.model_name, 'wb') as f:
            cPickle.dump([train_epoch_accuracy, valid_epoch_accuracy,
                train_epoch_loss, valid_epoch_loss,
                train_epoch_gradient_norm], f)

def test(FLAGS):
    """
    Sample inputs of your own.
    """
    with tf.Session() as sess:

        # Inputs need to real inputs of batch_size 128
        # because we use A(t) which updates even during testing

        # Load the model
        model = create_model(sess, FLAGS)

        # Load real samples
        with open(os.path.join(FLAGS.data_dir, 'train.p'), 'rb') as f:
            train_X, train_y = cPickle.load(f)

        rand_index = random.randint(0, len(train_X))  #从train_X中随机选择一个QA对
        logits = model.logits.eval(feed_dict={model.X: train_X[rand_index],
            model.l: FLAGS.l, model.e: FLAGS.e})

        print("INPUT:", train_X[rand_index])
        print("PREDICTION:", logits)
        print("TRUE ANSWER:", train_y[rand_index])


def plot_all():
    """
    Plot the results.
    """

    with open('RNN-LN-FW_results.p', 'rb') as f:
        RNN_LN_FW_results = cPickle.load(f)

    # Plotting accuracy
    fig = plt.figure()

    plt.plot(RNN_LN_FW_results[1], label='RNN-LN-FW accuracy')

    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc=4)
    fig.savefig('accuracy.png')
    #plt.show()

    # Plotting loss
    fig = plt.figure()

    plt.plot(RNN_LN_FW_results[3], label='RNN-LN-FW loss')

    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    fig.savefig('loss.png')
    #plt.show()


if __name__ == '__main__':

    FLAGS = parameters()
    model_name = "RNN-LN-FW"
    FLAGS.ckpt_dir = FLAGS.ckpt_dir + model_name
    FLAGS.model_name = model_name

    if sys.argv[1] == 'train':
        train(FLAGS)
    elif sys.argv[1] == 'test':
        test(FLAGS)
    elif sys.argv[1] == 'plot':
        plot_all()



