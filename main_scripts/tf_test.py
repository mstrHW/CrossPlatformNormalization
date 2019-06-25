import numpy as np
import tensorflow as tf
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

np.random.seed(101)
tf.set_random_seed(101)


class Model(object):
    def __init__(self, n):
        self.X = tf.placeholder("float")
        self.Y = tf.placeholder("float")

        self.W = tf.Variable(np.random.randn(), name="W")
        self.b = tf.Variable(np.random.randn(), name="b")

        self.y_pred = tf.add(tf.multiply(self.X, self.W), self.b)

        # Mean Squared Error Cost Function
        self.cost = tf.reduce_sum(tf.pow(self.y_pred - self.Y, 2)) / (2 * n)

        # Gradient Descent Optimizer
        learning_rate = 0.01
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

        # Global Variables Initializer
        self.init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(self.init)

            weight = sess.run(self.W)
            bias = sess.run(self.b)
            print('init')
            print("W =", weight, "b =", bias)

    def fit(self, x, y):
        training_epochs = 50

        with tf.Session() as sess:

            # Initializing the Variables
            sess.run(self.init)

            weight = sess.run(self.W)
            bias = sess.run(self.b)
            print('start fit')
            print("W =", weight, "b =", bias)

            # Iterating through all the epochs
            for epoch in tqdm(range(training_epochs)):

                # Feeding each data point into the optimizer using Feed Dictionary
                for (_x, _y) in zip(x, y):
                    sess.run(self.optimizer, feed_dict={self.X: _x, self.Y: _y})

                    # Displaying the result after every 50 epochs
                if (epoch + 1) % 50 == 0:
                    # Calculating the cost a every epoch
                    c = sess.run(self.cost, feed_dict={self.X: x, self.Y: y})
                    # print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(self.W), "b =", sess.run(self.b))

                    # Storing necessary values to be used outside the Session
            training_cost = sess.run(self.cost, feed_dict={self.X: x, self.Y: y})
            weight = sess.run(self.W)
            bias = sess.run(self.b)

            print('end fit')
            print("W =", weight, "b =", bias)

    def predict(self):
        with tf.Session() as sess:

            # Initializing the Variables
            sess.run(self.init)
            tf.get_variable_scope().reuse_variables()

            weight = sess.run(self.W)
            bias = sess.run(self.b)
            print('predict')
            print("W =", weight, "b =", bias)


x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

# Adding noise to the random linear data
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

model = Model(len(x))
model.fit(x, y)
model.predict()
