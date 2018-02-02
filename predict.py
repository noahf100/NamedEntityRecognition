import tensorflow as tf
import sys
import pickle
import numpy as np

'''
args = sys.argv
if len(args) != 2:
    print(len(args))
    print("Arguement mismatch. Proper input = sentence")
    sys.exit(0)

sentence = args[1]
'''

dict = pickle.load( open( "models/dict.pkl", "rb" ) )
resDict = pickle.load( open( "models/resDict.pkl", "rb" ) )
charDict = pickle.load( open( "models/charDict.pkl", "rb" ) )

revResDict = {}
for i in resDict.keys():
    revResDict[resDict[i]] = i

revDict = {}
for i in dict.keys():
    revDict[dict[i]] = i

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    #saver = tf.train.Saver()
    # Restore variables from disk.
    saver = tf.train.import_meta_graph("models/model.meta")
    saver.restore(sess, tf.train.latest_checkpoint('./models/'))
    print("Model restored.")

    graph = tf.get_default_graph()
    sequence_lengths = graph.get_tensor_by_name("sequence_lengths:0")
    labels = graph.get_tensor_by_name("labels:0")
    input = graph.get_tensor_by_name("input:0")
    charss = graph.get_tensor_by_name("chars:0")
    word_lengths = graph.get_tensor_by_name("word_lengths:0")
    #logits = graph.get_tensor_by_name("logits:0")
    #trans_params = graph.get_tensor_by_name("trans_params:0")
    softmax = graph.get_tensor_by_name("softmax:0")

    while True:
        try:
            # for python 2                                                      
            sentence = raw_input("input> ")
        except NameError:
            # for python 3                                                      
            sentence = input("input> ")

        words = sentence.split()
        labs = [0 for i in range(len(words))]
        w = [0 for i in range(len(words))]
        lens = [0 for i in range(len(words))]
        
        maxC = 0

        for i in range(len(words)):
            maxC = max(maxC, len(words[i]))
            lens[i] = len(words[i])
            if words[i].lower() in dict:
                w[i] = dict[words[i].lower()]
            else:
                w[i] = dict["UNKNOWN"]

        chars = np.zeros([1, len(words), maxC])
        for i in range(len(words)):
            for j in range(len(words[i])):
                chars[0][i][j] = charDict[words[i][j]]

        w = np.array(w).reshape(1, -1)
        lens = np.array(lens).reshape(1, -1)

        pred = sess.run([softmax], feed_dict={input: w, word_lengths: lens, charss: chars, sequence_lengths: [len(words)]})

        str1 = ""
        str2 = ""
        for i in range(len(words)):
            str1 += str(words[i]) + "\t"
            str2 += str(revResDict[pred[0][0][i]]) + "\t"
            #print(words[i] + " - " + revResDict[pred[0][0][i]])

        print(str1)
        print(str2)
