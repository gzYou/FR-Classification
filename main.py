# coding: utf-8

import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from tools import data_pipeline
from tools import Recording



# Machine Learning-Based Model for Prediction of Outcomes in Acute Stroke
# 3 hidden layers with 15 artificial neural network units each were used

tf.set_random_seed(0)

# parameters
init_learning_rate = 0.006
training_epochs = 600
batch_size = 10
display_step = 10
decay_rate = 0.96 
decay_steps = 50 
use_learning_decay = True
use_dropout = True

# Networks Parameters
X_train_scaled,y_train,X_test,y_test,X_scaler = data_pipeline() # here just be used to initialize two parameters of 'dim' and 'nclass' 
dim = X_train_scaled.shape[1]
nclass = y_train.shape[1]
n_hidden_1 = 15
n_hidden_2 = 15
n_hidden_3 = 15
n_input = dim
n_classes = nclass

# tf.Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
dropout_keep_prob = tf.placeholder('float')
learning_rate = tf.placeholder('float')
global_step = tf.Variable(tf.constant(0))
lr = tf.train.exponential_decay(init_learning_rate,global_step,decay_steps,decay_rate)

# Store layers weight & bias
stddev = 0.1  # important
weights = {
    "h1": tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    "h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    "h3": tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=stddev)),
    "out": tf.Variable(tf.random_normal([n_hidden_3, n_classes], stddev=stddev))
}

biases = {
    "b1": tf.Variable(tf.random_normal([n_hidden_1])),
    "b2": tf.Variable(tf.random_normal([n_hidden_2])),
    "b3": tf.Variable(tf.random_normal([n_hidden_3])),
    "out": tf.Variable(tf.random_normal([n_classes])),
}

# Create model
def multiplayer_perception(_X, _weights, _biases,_keep_prob):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights["h1"]), _biases["b1"]))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights["h2"]), _biases["b2"]))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, _weights["h3"]), _biases["b3"]))
    if use_dropout:
        layer_3 = tf.nn.dropout(layer_3,_keep_prob)
    out = tf.add(tf.matmul(layer_3, _weights["out"]) , _biases["out"]) 
    return out
print("Network Ready!")





# Construct Model
pred = multiplayer_perception(x,weights,biases,dropout_keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y))
optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))    
accu = tf.reduce_mean(tf.cast(corr, "float"))
print ("Functions ready")





def train(X_train_scaled,y_train,X_test,y_test,X_scaler):
    ntrain = y_train.shape[0]
    ntest = y_test.shape[0]
    # Initializing the variables
    init = tf.initialize_all_variables()
    # Launch the graph
    sess = tf.Session()
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        if use_learning_decay:
            lr_ = sess.run(lr,feed_dict={global_step:epoch})
        else:
            lr_ = init_learning_rate
        avg_cost = 0.
        total_batch = int(ntrain/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            randidx = np.random.randint(ntrain, size=batch_size)
            batch_xs = X_train_scaled[randidx, :]
            batch_ys = y_train[randidx, :]
            # Fit training using batch data
            if use_dropout:
                sess.run(optm, feed_dict={x: batch_xs, y: batch_ys,dropout_keep_prob:0.8,learning_rate:lr_})
            else:
                sess.run(optm, feed_dict={x: batch_xs, y: batch_ys,dropout_keep_prob:1.,learning_rate:lr_})
            # Compute average loss
            avg_cost += sess.run(cost, 
                    feed_dict={x: batch_xs, y: batch_ys,dropout_keep_prob:1.0})/total_batch
            # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print ("Epoch: %03d/%03d ,learning_rate:%.6f,cost: %.9f" % 
                   (epoch+1, training_epochs, lr_,avg_cost))
            train_acc = sess.run(accu, feed_dict={x: X_train_scaled, y: y_train,dropout_keep_prob:1.0})
            print (" Training accuracy: %.3f" % (train_acc))
            test_acc,auc = evaluate(sess,X_test,y_test,X_scaler)
            print (" Test accuracy: %.3f" % (test_acc))
            print (" Test ROC_AUC: %.3f" % (auc))
    print ("Optimization Finished!")
    test_acc,auc = evaluate(sess,X_test,y_test,X_scaler)
    sess.close()
    return test_acc , auc





def evaluate(sess,X_test,y_test,X_scaler):
    X_test_scaled = X_scaler.transform(X_test.astype(float))
    test_acc = sess.run(accu, feed_dict={x: X_test_scaled, y: y_test, dropout_keep_prob: 1.0})
    probs = sess.run(tf.sigmoid(pred),feed_dict={x: X_test_scaled, y: y_test,dropout_keep_prob:1.0})[:,1]
    auc = roc_auc_score(np.argmax(y_test,1),probs)
    return [test_acc,auc]





def experiments_and_recording():
    rec = Recording()
    record = rec.getRecordFile()
    accuracy_10_runs = []
    auc_10_runs = []
    one_10_runs = [
                    training_epochs,
                    batch_size,
                    False,
                    init_learning_rate,
                    decay_rate,
                    decay_steps,
                    3,
                    (n_hidden_1,n_hidden_2,n_hidden_3),
                    False,
                    1
                ]
    for i in range(10):
        X_train_scaled,y_train,X_test,y_test,X_scaler = data_pipeline()
        acc,auc = train(X_train_scaled,y_train,X_test,y_test,X_scaler)
        accuracy_10_runs.append(acc)
        auc_10_runs.append(auc)
    one_10_runs.extend(accuracy_10_runs)
    one_10_runs.extend(auc_10_runs)
    one_10_runs.append(np.mean(accuracy_10_runs))
    one_10_runs.append(np.mean(auc_10_runs))
    record = record.append(pd.Series(one_10_runs,index=rec.getIndex()),ignore_index=True)
    record.to_csv('record.csv')


experiments_and_recording()

