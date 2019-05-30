import tensorflow as tf
from module.data_processing.noising_methods import set_zero, add_gaussian_noise


class DenoisingAutoencoder(object):

    def __init__(self, input_dim, hidden_dim, noising_method_name, noising_prob, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noising_method_name = noising_method_name
        self.noising_prob = noising_prob
        self.learning_rate = learning_rate

        self.build_model()

    def build_model(self):
        self.input_x, self.noised_x = self._create_placeholders(self.input_dim)

        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([self.input_dim, self.hidden_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([self.hidden_dim]), name='biases')
            self.encoded = tf.nn.sigmoid(tf.matmul(self.noised_x, weights) + biases, name='encoded')

        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([self.hidden_dim, self.input_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([self.input_dim]), name='biases')
            self.decoded = tf.matmul(self.encoded, weights) + biases

        self.loss = tf.losses.mean_squared_error(self.input_x, self.decoded)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def _create_placeholders(self, input_dim):
        input_x = tf.placeholder(tf.float32, [None, input_dim], name='input_x')
        noised_x = tf.placeholder(tf.float32, [None, input_dim], name='noised_x')

        return input_x, noised_x

    def _noise_data(self, x, noising_prob):
        if self.noising_method_name == 'set_zero':
            noised_x = set_zero(x, noising_prob)
        elif self.noising_method_name == 'gaussian':
            noised_x = gaussian_noise(x)
        elif self.noising_method_name == 'none':
            noised_x = x
        else:
            raise ValueError('noising method name')

        return noised_x

    def train(self, loader, batch_size, batches_count):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i, batch in enumerate(loader.get_batches(batch_size, batches_count)):
                noised_batch = self._noise_data(batch, self.noising_prob)
                loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.input_x: batch, self.noised_x: noised_batch})
                if i % 1000 == 0:
                    print('train loss : {}'.format(loss))

            print('train loss : {}'.format(loss))
            self.saver.save(sess, './model.ckpt')

    def test(self, loader):
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            loss, encoded, decoded = sess.run([self.loss, self.encoded, self.decoded], feed_dict={self.input_x: loader.data, self.noised_x: loader.data})
            print('test loss : {}'.format(loss))
            # mae = tf.metrics.mean_absolute_error(loader.data, decoded)
            # _mae = sess.run(mae)
            # print(_mae)


def main():
    from module.data_loader.data_loader import DataLoader

    hidden_dim = 30

    loader = DataLoader()
    print('input dim : {}'.format(loader.input_dim))

    model = DenoisingAutoencoder(loader.input_dim, hidden_dim, 'none', 0.1, 0.05)
    model.train(loader, 128, 5000)
    model.test(loader)


if __name__ == '__main__':
    main()
