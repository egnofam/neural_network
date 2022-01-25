import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

tf.disable_v2_behavior()


def generate_dataset():
    """
        function to generate datasets
        :return:  features and targets
    """
    nb_rows = 100  # generate number of rows
    covid_case = np.random.randn(nb_rows, 2) + [-3, -3]  # generate covid cases
    covid_case2 = np.random.randn(nb_rows, 2) + [3, 3]
    non_covid_case = np.random.randn(nb_rows, 2) + [-3, 3]  # generate non covid cases
    non_covid_case2 = np.random.randn(nb_rows, 2) + [3, -3]
    target_covidcase = np.zeros(2*(covid_case.shape[0])).reshape(2*nb_rows, 1)  # generate target for covid cases
    target_non_covidcase = np.ones(2*(non_covid_case.shape[0])).reshape(2*nb_rows, 1)  # generate target for non
    # covid cases
    target = np.concatenate((target_covidcase, target_non_covidcase), axis=0)  # concatenate target for covid cases
    # and non covid cases
    patients = np.concatenate((covid_case, covid_case2, non_covid_case, non_covid_case2), axis=0)  # concatenate
    # patients datasets
    # print(target_covidcase.shape)
    # print(target_non_covidcase.shape)
    return patients, target


def plot_figure(features, y):
    """
        function to plot figure
        :param data: should have two features
        :param y: target value
        :return: none
    """
    plt.scatter(features[:, 0], features[:, 1], c=y)
    plt.show()


if __name__ == "__main__":

    # get features, targets , weights, bias

    features, targets = generate_dataset()

    # plot figure
    # plot_figure(features, targets)
    tf_features = tf.placeholder(tf.float32, shape=[None, 2])
    tf_targets = tf.placeholder(tf.float32, shape=[None, 1])

    # first
    w1 = tf.Variable(tf.random_normal([2, 3]))
    b1 = tf.Variable(tf.zeros([3]))
    # operations 1
    z1 = tf.matmul(tf_features, w1) + b1
    a1 = tf.nn.sigmoid(z1)
    # second
    w2 = tf.Variable(tf.random_normal([3, 1]))
    b2 = tf.Variable(tf.zeros([1]))

    # operations 2
    z2 = tf.matmul(a1, w2) + b2
    py = tf.nn.sigmoid(z2)

    cost = tf.reduce_mean(tf.square(py - tf_targets))
    correct_predictions = tf.equal(tf.round(py), tf_targets)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    optimizer = GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # print("w = ", sess.run(w))
    # print("b = ", sess.run(b))
    for e in range(10000):
        print('cost = ', sess.run(train, feed_dict={tf_features: features, tf_targets: targets}))
        print("accuracy = ", sess.run(accuracy, feed_dict={tf_features: features, tf_targets: targets}))


    # print("py= ", sess.run(py, feed_dict={
    #     tf_features: [features[0]]
    # }))
    # print("cost= ", sess.run(cost, feed_dict={
    #     tf_features: [features[0]],
    #     tf_targets: [targets[0]]
    # }))



