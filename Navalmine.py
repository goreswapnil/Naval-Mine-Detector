import numpy as np 
import pandas as pd  
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def read_dataset():
    df = pd.read_csv("sonar.csv")
    print("Dataset loaded succesfully")
    print("Number of Columns: ",len(df.columns))

    # Fetures of datset
    X = df[df.columns[0:60]].values

    # Label of dataset
    y = df[df.columns[60]]

    # Encoded the dependant variable 
    encoder = LabelEncoder()

    # Encode character labe into integer ie 1 or 0 (One hot encode)
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)

    print("X.shape",X.shape)

    return (X,Y)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels] = 1
    return one_hot_encode

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activationsd
    layer_1 = tf.add(tf.matmul(x, weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']

    return out_layer

def main():

    X,Y = read_dataset()
    X,Y = shuffle(X,Y,random_state = 1)
    train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.30, random_state = 415)

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)

    # Change in variable in each iteration
    learning_rate = 0.3

    # Total number of iterations to minimize the error
    trainig_epochs = 100

    n_dim = X.shape[1]
    print("n_dim = ",n_dim)

    n_class = 2

    cost_history = np.empty(shape=[1], dtype=float)

    # 4 Layer dense neural networks
    n_hidden_1 = 60
    n_hidden_2 = 60
    n_hidden_3 = 60
    n_hidden_4 = 60

    # define the weights and the biases for each layer
    # Create variable which contens random values
    weights = {
        'h1':tf.Variable(tf.random.truncated_normal([n_dim,n_hidden_1])),
        'h2':tf.Variable(tf.random.truncated_normal([n_hidden_1,n_hidden_2])),
        'h3':tf.Variable(tf.random.truncated_normal([n_hidden_2,n_hidden_3])),
        'h4':tf.Variable(tf.random.truncated_normal([n_hidden_3,n_hidden_4])),
        'out':tf.Variable(tf.random.truncated_normal([n_hidden_4,n_class]) ),
    }

    # Create variable which contains random value
    biases = {
        'b1':tf.Variable(tf.random.truncated_normal([n_hidden_1])),
        'b2':tf.Variable(tf.random.truncated_normal([n_hidden_2])),
        'b3':tf.Variable(tf.random.truncated_normal([n_hidden_3])),
        'b4':tf.Variable(tf.random.truncated_normal([n_hidden_4])),
        'out':tf.Variable(tf.random.truncated_normal([n_class])),
    }

    x = tf.placeholder(tf.float32, [None, n_dim])
    #W = tf.Variable(tf.zeros([n_dim, n_class]))
    #b = tf.Variable(tf.zeros([n_class]))
    y_ = tf.placeholder(tf.float32, [None, n_class])

    # Initialization of variables
    init = tf.compat.v1.global_variables_initializer()

    #saver = tf.compat.v1.train.Saver()

    # Call to model function for training
    y = multilayer_perceptron(x,weights,biases)

    # Define the cost function to calculate loss 
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))

    # Functiom to reduce loss
    training_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    # Initializing session
    sess = tf.Session()
    sess.run(init)

    #mse_hisory = []
    accuracy_history = []
    # Calculate the cost and the accuracy for each epoch

    for epoch in range(trainig_epochs):
        sess.run(training_step,feed_dict={x:train_x,y_:train_y})
        cost = sess.run(cost_function,feed_dict={x:train_x,y_:train_y})
        cost_history = np.append(cost_history,cost)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        pred_y = sess.run(y,feed_dict={x:test_x})
        mse = tf.reduce_mean(tf.square(pred_y - test_y))
        mse_ = sess.run(mse)
        accuracy = (sess.run(accuracy,feed_dict={x:train_x,y_:train_y}))
        accuracy_history.append(accuracy)
        print("epoch:",epoch,"-","cost:",cost,"-MSE:",mse_,"-Train Accuracy:",accuracy)

    # Print the final mean square error
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.square(pred_y - test_y))
    print("test Accuracy: ",(sess.run(y,feed_dict={x:test_x,y_:test_y})))

if __name__ == "__main__":
    main()
    

