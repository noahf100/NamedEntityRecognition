import tensorflow as tf
import numpy as np
import sys
import pickle
import os

class Model:
    def __init__(self, vocab_size, charSize):
        #vars
        self.lrate = 0.001
        self.epochs = 1
        self.bSize = 1
        self.wordEmbedSize = 128
        self.charEmbedSize = 128
        self.charHiddenSize = 128
        self.lstmSize = 128
        self.dropoutProb = 0.68
        self.vocab_size = vocab_size
        self.char_size = charSize

    def add_placeholders(self):
        #model
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")
        self.input = tf.placeholder(tf.int32, shape=[None, None], name="input")
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None],name="chars")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

    def add_word_embeddings_op(self):
        E = tf.Variable(initial_value=tf.random_normal(shape=[self.vocab_size, self.wordEmbedSize], stddev = 0.1), name="E")
        lookup = tf.nn.embedding_lookup(E, self.input, name="lookup")

        E2 = tf.Variable(initial_value=tf.random_normal(shape=[self.char_size, self.charEmbedSize], stddev = 0.1),dtype=tf.float32)
        lookup2 = tf.nn.embedding_lookup(E2, self.chars, name="char_lookup")

        s = tf.shape(lookup2)
        char_embeddings = tf.reshape(lookup2,shape=[s[0]*s[1], s[-2], self.charEmbedSize])
        word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

        cell_fw = tf.contrib.rnn.LSTMCell(self.charHiddenSize, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(self.charHiddenSize, state_is_tuple=True)
        _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings, sequence_length=word_lengths, dtype=tf.float32)

        _, ((_, output_fw), (_, output_bw)) = _output
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.reshape(output, shape=[s[0], s[1], 2*self.charHiddenSize])

        word_embeddings = tf.concat([lookup, output], axis=-1)

        self.embed = tf.nn.dropout(word_embeddings, self.dropoutProb, name="e_dropout")
        #self.embed = tf.expand_dims(e_dropout, 0, name="embed")

    def add_logits_op(self):
        with tf.variable_scope("bi-lstm"):
            # Forward direction cell
            fw_cell = tf.contrib.rnn.LSTMCell(self.lstmSize)

            # Backward direction cell
            bw_cell = tf.contrib.rnn.LSTMCell(self.lstmSize)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, self.embed, 
            sequence_length=self.sequence_lengths, dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropoutProb)
            

        with tf.variable_scope("proj"):
            W = tf.Variable(initial_value=tf.random_normal(shape=[2*self.lstmSize, self.vocab_size], stddev=0.1), name="Weights")
            b = tf.Variable(initial_value=tf.random_normal(shape=[self.vocab_size], stddev=0.1), name="bias")

            o1 = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.lstmSize], name="output")
            logits = tf.matmul(output, W) + b
            self.logits = tf.reshape(logits, [-1, o1, self.vocab_size], name="logits")

    def add_loss_op(self):
        #Not crf
        
        softmax = tf.nn.softmax(self.logits)
        self.softmax = tf.cast(tf.argmax(self.logits, 1), tf.int32, name="softmax")

        xEnt = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.sequence_lengths)
        xEnt = tf.boolean_mask(xEnt, mask)

        self.loss = tf.reduce_mean(xEnt, name="loss")
        

        '''
        #crf
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.sequence_lengths)
        self.trans_params = tf.identity(trans_params, name="trans_params")
        self.loss = tf.reduce_mean(-log_likelihood, name="loss")
        '''

    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate)
        self.train_op = optimizer.minimize(self.loss, name="train_op")

    def initialize_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()
        self.add_training_op()
        self.initialize_session()


    def train(self, data, dict, dataLabels, data_test, resDict, dataLabels_test, charDict, word_lengths, chars, word_lengths_test, chars_test):
        total_loss = 0.0
        for j in range(self.epochs):
            for i in range(len(data)):
                
                if dict["STOP"] in data[i]:
                    stopIndex = data[i].tolist().index(dict["STOP"])
                    sentence = data[i][:stopIndex]
                    labs = dataLabels[i][:stopIndex]
                    characters = chars[i][:stopIndex]#[:]
                    wordLengths = word_lengths[i][:stopIndex]
                else:
                    sentence = data[i]
                    labs = dataLabels[i]
                    characters = chars[i]
                    wordLengths = word_lengths[i]
                
                '''
                
                stopIndex = data[i].tolist().index(dict["STOP"]) + 1 if dict["STOP"] in data[i] else len(data[i])
                sentence = data[i]
                labs = dataLabels[i]
                characters = chars[i][:stopIndex]
                wordLengths = word_lengths[i]
                '''

                labs = labs.reshape(1, -1)
                sentence = sentence.reshape(1, -1)
                characters = characters.reshape(1, characters.shape[0], -1)
                wordLengths = wordLengths.reshape(1, -1)
                
                l, _ = self.session.run([self.loss, self.train_op], 
                    feed_dict={
                        self.input: sentence,
                        self.labels: labs,
                        self.sequence_lengths: [stopIndex],
                        self.word_lengths: wordLengths,
                        self.chars: characters
                      })
                total_loss += l
            
                # Print Loss every so often
                if i % 1000 == 0:
                    print 'Iteration %d\tLoss Value: %.3f' % (i, l)

            self.save()
            self.test(data_test, dict, resDict, charDict, dataLabels_test, word_lengths_test, chars_test)


    def test(self, data_test, dict, resDict, charDict, dataLabels_test, word_lengths_test, chars_test):
        total_loss = 0.0
        total_correct = 0.0
        total_diff = 0.0
        total_count = 0.0

        for i in range(len(data_test)):
            sentence = []
            labs = []
            if dict["STOP"] in data_test[i]:
                stopIndex = data_test[i].tolist().index(dict["STOP"])
                sentence = data_test[i][:stopIndex]
                labs = dataLabels_test[i][:stopIndex]
                characters = chars_test[i][:stopIndex]#[:]
                wordLengths = word_lengths_test[i][:stopIndex]
            else:
                sentence = data_test[i]
                labs = dataLabels_test[i]
                characters = chars_test[i]
                wordLengths = word_lengths_test[i]
                
            labs = labs.reshape(1, -1)
            sentence = sentence.reshape(1, -1)
            characters = characters.reshape(1, characters.shape[0], -1)
            wordLengths = wordLengths.reshape(1, -1)

            bad = False
            for i in wordLengths[0]:
                if i < 0:
                    bad = True
                    break
            if bad:
                continue

            l, pred = self.session.run([self.loss, self.softmax], 
                feed_dict={
                self.input: sentence, 
                self.labels: labs, 
                self.sequence_lengths: [len(sentence)],
                self.word_lengths: wordLengths,
                self.chars: characters
                })
            
            total_loss += l
            for i in range(len(pred)):
                if pred[i] is not resDict['O']:
                    total_diff += 1
                if pred[0][i] == labs[0][i]:
                    total_correct += 1
                total_count += 1

        r = total_correct/float(total_diff) if total_diff > 0 else 0
        p = total_correct/float(total_count) if total_count > 0 else 0
        f1  = 2 * p * r / (p + r) if p + r > 0 else 0

        print("Total Loss: " + str(total_loss))
        print("F1: " + str(f1))
        print("Accuracy: " + str(p))


    def save(self):
        self.saver.save(self.session, 'models/model')
