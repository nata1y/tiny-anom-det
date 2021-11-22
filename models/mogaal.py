# from https://github.com/leibinghe/GAAL-based-outlier-detection
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import argparse
from tensorflow.keras.initializers import VarianceScaling, Identity
import tensorflow
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.keras.layers import TimeDistributed, Dropout, LSTM, RepeatVector

from utils import create_dataset, create_dataset_keras
anomaly_window = 60

print(f'Running on GPU {tensorflow.test.is_built_with_cuda()}. '
      f'Devices: {tensorflow.config.list_physical_devices("GPU")}')


def parse_args():
    parser = argparse.ArgumentParser(description="Run MO-GAAL.")
    parser.add_argument('--path', nargs='?', default='~/Documents/uni/GAAL-based-outlier-detection/Data/Annthyroid',
                        help='Input data path.')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of sub_generator.')
    parser.add_argument('--stop_epochs', type=int, default=500,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--lr_d', type=float, default=0.01,
                        help='Learning rate of discriminator.')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                        help='Learning rate of generator.')
    parser.add_argument('--decay', type=float, default=1e-6,
                        help='Decay.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    return parser.parse_args()


class MOGAAL:

    def __init__(self, dataset, args=None):
        if not args:
            args = parse_args()
        self.decay = args.decay
        self.k = args.k
        self.lr_d = args.lr_d
        self.lr_g = args.lr_g
        self.momentum = args.momentum
        self.path = args.path
        self.stop_epochs = args.stop_epochs
        self.dataset, self.type, self.file = dataset

    # Generator
    def create_generator(self, latent_size):
        gen = Sequential()
        gen.add(Dense(latent_size, input_dim=self.latent_size, activation='relu', kernel_initializer=Identity(gain=1.0)))
        gen.add(Dense(latent_size, activation='relu', kernel_initializer=Identity(gain=1.0)))
        latent = Input(shape=(self.latent_size,))
        fake_data = gen(latent)
        return Model(latent, fake_data)

    # Discriminator
    def create_discriminator(self):
        dis = Sequential()
        dis.add(LSTM(128, input_shape=(anomaly_window, 1)))
        dis.add(Dropout(rate=0.2))
        dis.add(RepeatVector(anomaly_window))
        dis.add(LSTM(128, return_sequences=True))
        dis.add(Dropout(rate=0.2))
        dis.add(TimeDistributed(Dense(1)))
        dis.compile(optimizer='adam', loss='mae')
        data = Input(shape=(anomaly_window, 1))
        fake = dis(data)
        return Model(data, fake)

        # dis = Sequential()
        # dis.add(Dense(math.ceil(math.sqrt(self.data_size)), input_dim=self.latent_size, activation='relu',
        #               kernel_initializer=tensorflow.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',
        #                                                                     distribution='normal', seed=None)))
        # dis.add(Dense(1, activation='sigmoid',
        #               kernel_initializer=tensorflow.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',
        #                                                                     distribution='normal', seed=None)))
        # data = Input(shape=(self.latent_size,))
        # fake = dis(data)
        # return Model(data, fake)

    # Load data
    def load_data(self):
        data = pd.read_table('{path}'.format(path=self.path), sep=',', header=None)
        data = data.sample(frac=1).reset_index(drop=True)
        id = data.pop(0)
        y = data.pop(1)
        data_x = data.values
        data_id = id.values
        data_y = y.values
        return data_x, data_y, data_id

    # Plot loss history
    def plot(self, train_history, name):
        dy = train_history['discriminator_loss']
        gy = train_history['generator_loss']
        auc_y = train_history['auc']
        for i in range(self.k):
            self.names['gy_' + str(i)] = train_history['sub_generator{}_loss'.format(i)]
        x = np.linspace(1, len(dy), len(dy))
        fig, ax = plt.subplots()
        ax.plot(x, dy, color='blue', label='Discriminator loss')
        ax.plot(x, gy, color='red', label='Generator loss')
        ax.plot(x, auc_y, color='yellow', linewidth='3', label='ROC AUC')
        for i in range(self.k):
            ax.plot(x, self.names['gy_' + str(i)], color='green', linewidth='0.5', label='Sub-generator loss')
        plt.legend()
        plt.savefig(f'results/imgs/{self.dataset}/{self.type}/mogaal/mogaal_{self.file.replace(".csv", "")}_loss.png')

        outliers = []
        # save potential outliers
        for i in range(self.k):
            outliers.append(self.names['generated_data' + str(i)])

    def predict(self, data=pd.DataFrame([])):
        if data.shape[0] == 0:
            data_x, data_y, data_id = self.load_data()
        else:
            data_x, data_y, data_id = data[['value', 'timestamp']].values, data['is_anomaly'].values, data['timestamp'].values

        data_y = pd.DataFrame(data_y)
        # Get the target value of sub-generator
        preds = self.discriminator.predict(data_x)
        p_value = pd.DataFrame(preds)
        result = np.concatenate((p_value, data_y), axis=1)
        result = pd.DataFrame(result, columns=['p', 'y'])
        result = result.sort_values('p', ascending=True)
        preds = [1 if x[0] > 0.9 else 0 for x in preds]

        return preds

    def _insert_anomalies(self, data_batch, anomalies, noise_idxs):
        data_batch.set_index('timestamp', inplace=True)
        # TODO: check whether output indeed normalized
        for (val, t) in anomalies:
            if int(t) in data_batch.index:
                data_batch.loc[int(t), 'value'] = val
                noise_idxs.append(int(t))
        data_batch.reset_index(inplace=True)
        return data_batch

    def fit(self, data=pd.DataFrame([]), plot=True, batch_size=60):
        self.train = True

        # initialize dataset
        if data.shape[0] == 0:
            data_x, data_y, data_id = self.load_data()
        else:
            data_x, data_y, data_id = data[['value', 'timestamp']], data['is_anomaly'].values, data['timestamp'].values

        self.data_size = data_x.shape[0]
        self.latent_size = data_x.shape[1]
        print("The dimension of the training data :{}*{}".format(self.data_size, self.latent_size))
        self.last_batch_anomalies = []

        # data_x, _ = create_dataset(data_x[['value']], data_x[['value']], anomaly_window)

        if self.train:
            train_history = defaultdict(list)
            self.names = locals()
            epochs = self.stop_epochs * 3
            stop = 0
            k = self.k

            # Create discriminator
            self.discriminator = self.create_discriminator()
            self.discriminator.compile(optimizer=SGD(lr=self.lr_d, decay=self.decay, momentum=self.momentum),
                                  loss='binary_crossentropy')

            # Create k combine models
            for i in range(k):
                self.names['sub_generator' + str(i)] = self.create_generator(self.latent_size)
                latent = Input(shape=(self.latent_size,))
                self.names['fake' + str(i)] = tensorflow.expand_dims(tensorflow.gather(
                    self.names['sub_generator' + str(i)](latent), [0], axis=1), axis=0)
                self.discriminator.trainable = False

                tsvalues = tensorflow.expand_dims(tensorflow.gather(self.names['fake' + str(i)], [0], axis=1), axis=0)
                # X, y = create_dataset_keras(tsvalues, tsvalues, 60)
                self.names['fake' + str(i)] = self.discriminator(tsvalues) #r(self.names['fake' + str(i)])
                self.names['combine_model' + str(i)] = Model(latent, self.names['fake' + str(i)])
                self.names['combine_model' + str(i)].compile(optimizer=
                                                             SGD(lr=self.lr_g, decay=self.decay, momentum=self.momentum),
                                                             loss='binary_crossentropy')

            # Start iteration
            for epoch in range(epochs):
                print('Epoch {} of {}'.format(epoch + 1, epochs))
                batch_size = min(batch_size, self.data_size)
                num_batches = int(self.data_size / batch_size)

                for index in range(num_batches):
                    print('\nTesting for epoch {} index {}:'.format(epoch + 1, index + 1))

                    # Generate noise
                    noise_size = batch_size
                    # Normalization does not make sense?
                    # noise = np.random.uniform(0, 1, (int(noise_size), self.latent_size))
                    noise = np.random.uniform(data['value'].min(), data['value'].max(), (int(noise_size), 1))
                    # append artificial time index
                    noise = np.append(noise, [[x] for x in range(index * batch_size, (index + 1) * batch_size)], axis=1)

                    # Get training data
                    data_batch = data_x[index * batch_size: (index + 1) * batch_size]

                    # Generate potential outliers
                    block = ((1 + k) * k) // 2
                    for i in range(k):
                        if i != (k-1):
                            noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                            noise_end = int((((k + (k - i)) * (i + 1)) / 2) * (noise_size // block))
                            self.names['noise' + str(i)] = noise[noise_start:noise_end ]
                            self.names['generated_data' + str(i)] = self.names['sub_generator' + str(i)].predict(self.names['noise' + str(i)], verbose=0)
                        else:
                            noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                            self.names['noise' + str(i)] = noise[noise_start:noise_size]
                            self.names['generated_data' + str(i)] = self.names['sub_generator' + str(i)].predict(self.names['noise' + str(i)], verbose=0)

                    # Concatenate real data to generated data
                    noise_idxs = []
                    for i in range(k):
                        if i == 0:
                            X = self._insert_anomalies(data_batch, self.names['generated_data' + str(i)], noise_idxs)
                            # X = np.concatenate((data_batch, self.names['generated_data' + str(i)]))
                        else:
                            # TODO: shall we insert anomalies in last batch only or in all X history?
                            X = self._insert_anomalies(X, self.names['generated_data' + str(i)], noise_idxs)
                            # X = np.concatenate((X, self.names['generated_data' + str(i)]))

                    # instead of concatenating, we insert anomalies inside of time strean
                    # Y = np.array([1] * batch_size + [0] * int(noise_size))
                    Y = np.array([1 if x in noise_idxs else 0 for x in X.index.tolist()])

                    # Train discriminator
                    X = X['value'].to_numpy().reshape(1, anomaly_window, 1)
                    Y = Y.reshape(1, anomaly_window)
                    discriminator_loss = self.discriminator.train_on_batch(X, Y)
                    train_history['discriminator_loss'].append(discriminator_loss)
                    # ==================================================================================================

                    # Get the target value of sub-generator
                    threshold = 0.5
                    data_x, _ = create_dataset(data_x[['value']], data_x[['value']], anomaly_window)
                    p_value = self.discriminator.predict(data_x)

                    # transform lstm batched predictions to normal ones
                    loss = np.abs(X - p_value).ravel()[:X.shape[1]]
                    y_pred = [0 if loss[idx] <= threshold else 1 for idx in range(len(loss))]

                    # p_value = pd.DataFrame(p_value)
                    p_value = pd.DataFrame(loss)

                    for i in range(k):
                        self.names['T' + str(i)] = p_value.quantile(i/k)
                        self.names['trick' + str(i)] = np.array([float(self.names['T' + str(i)])] * noise_size)

                    # Train generator
                    noise = np.random.uniform(0, 1, (int(noise_size), self.latent_size))
                    if stop == 0:
                        for i in range(k):
                            noise = noise[:, 0].reshape(1, anomaly_window, 1)
                            self.names['trick' + str(i)] = np.array([[x] for x in self.names['trick' + str(i)]])
                            self.names['trick' + str(i)] = np.append(
                                self.names['trick' + str(i)], [[x] for x in range(index * batch_size, (index + 1)
                                                                                  * batch_size)], axis=1)
                            self.names['trick' + str(i)] = self.names['trick' + str(i)].reshape(1,
                                                                                                self.names['trick' + str(i)].shape[0],
                                                                                                self.names['trick' + str(i)].shape[1])
                            self.names['sub_generator' + str(i) + '_loss'] = self.names['combine_model' +
                                                                                        str(i)].train_on_batch(noise, self.names['trick' + str(i)])
                            train_history[f'sub_generator{i}_loss'].append(self.names['sub_generator' + str(i) + '_loss'])
                    else:
                        for i in range(k):
                            self.names['sub_generator' + str(i) + '_loss'] = self.names['combine_model' + str(i)].evaluate(noise, self.names['trick' + str(i)])
                            train_history[f'sub_generator{i}_loss'].append(self.names['sub_generator' + str(i) + '_loss'])

                    generator_loss = 0
                    for i in range(k):
                        generator_loss = generator_loss + self.names['sub_generator' + str(i) + '_loss']
                    generator_loss = generator_loss / k
                    train_history['generator_loss'].append(generator_loss)

                    # Stop training generator
                    if epoch + 1 > self.stop_epochs:
                        stop = 1

                if plot:
                    # Detection result
                    data_y = pd.DataFrame(data_y)
                    result = np.concatenate((p_value, data_y), axis=1)
                    result = pd.DataFrame(result, columns=['p', 'y'])
                    result = result.sort_values('p', ascending=True)

                    # Calculate the AUC
                    inlier_parray = result.loc[lambda df: df.y == 0.0, 'p'].values
                    outlier_parray = result.loc[lambda df: df.y == 1.0, 'p'].values
                    sum = 0.0
                    for o in outlier_parray:
                        for i in inlier_parray:
                            if o < i:
                                sum += 1.0
                            elif o == i:
                                sum += 0.5
                            else:
                                sum += 0
                    AUC = '{:.4f}'.format(sum / (len(inlier_parray) * len(outlier_parray)))
                    print('AUC:{}'.format(AUC))
                    for i in range(num_batches):
                        train_history['auc'].append(sum / (len(inlier_parray) * len(outlier_parray)))

        if plot:
            self.plot(train_history, 'loss')


if __name__ == '__main__':
    # initilize arguments
    args = parse_args()
    print(args)
    mogaal = MOGAAL(args)
    mogaal.fit()
