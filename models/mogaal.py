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

print(f'Running on GPU {tensorflow.test.is_built_with_cuda()}. '
      f'Devices: {tensorflow.config.list_physical_devices("GPU")}')


def parse_args():
    parser = argparse.ArgumentParser(description="Run MO-GAAL.")
    parser.add_argument('--path', nargs='?', default='~/Documents/uni/GAAL-based-outlier-detection/Data/Annthyroid',
                        help='Input data path.')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of sub_generator.')
    parser.add_argument('--stop_epochs', type=int, default=1500,
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

    def __init__(self, args=None):
        if not args:
            args = parse_args()
        self.decay = args.decay
        self.k = args.k
        self.lr_d = args.lr_d
        self.lr_g = args.lr_g
        self.momentum = args.momentum
        self.path = args.path
        self.stop_epochs = args.stop_epochs

    # Generator
    def create_generator(self, latent_size):
        gen = Sequential()
        gen.add(Dense(latent_size, input_dim=latent_size, activation='relu', kernel_initializer=Identity(gain=1.0)))
        gen.add(Dense(latent_size, activation='relu', kernel_initializer=Identity(gain=1.0)))
        latent = Input(shape=(latent_size,))
        fake_data = gen(latent)
        return Model(latent, fake_data)

    # Discriminator
    def create_discriminator(self):
        dis = Sequential()
        dis.add(Dense(math.ceil(math.sqrt(self.data_size)), input_dim=self.latent_size, activation='relu',
                      kernel_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        dis.add(Dense(1, activation='sigmoid', kernel_initializer=VarianceScaling(scale=1.0, mode='fan_in',
                                                                                  distribution='normal', seed=None)))
        data = Input(shape=(self.latent_size,))
        fake = dis(data)
        return Model(data, fake)

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
        ax.plot(x, dy, color='blue')
        ax.plot(x, gy,color='red')
        ax.plot(x, auc_y, color='yellow', linewidth = '3')
        for i in range(self.k):
            ax.plot(x, self.names['gy_' + str(i)], color='green', linewidth='0.5')
        plt.show()

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

    def fit(self, data=pd.DataFrame([]), plot=True, batch_size=500):
        self.train = True

        # initialize dataset
        if data.shape[0] == 0:
            data_x, data_y, data_id = self.load_data()
        else:
            data_x, data_y, data_id = data[['value']].values, data['is_anomaly'].values, data['timestamp'].values
        self.data_size = data_x.shape[0]
        self.latent_size = data_x.shape[1]
        print("The dimension of the training data :{}*{}".format(self.data_size, self.latent_size))

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
                self.names['fake' + str(i)] = self.names['sub_generator' + str(i)](latent)
                self.discriminator.trainable = False
                self.names['fake' + str(i)] = self.discriminator(self.names['fake' + str(i)])
                self.names['combine_model' + str(i)] = Model(latent, self.names['fake' + str(i)])
                self.names['combine_model' + str(i)].compile(optimizer=SGD(lr=self.lr_g, decay=self.decay, momentum=self.momentum), loss='binary_crossentropy')

            # Start iteration
            for epoch in range(epochs):
                print('Epoch {} of {}'.format(epoch + 1, epochs))
                batch_size = min(batch_size, self.data_size)
                num_batches = int(self.data_size / batch_size)

                for index in range(num_batches):
                    print('\nTesting for epoch {} index {}:'.format(epoch + 1, index + 1))

                    # Generate noise
                    noise_size = batch_size
                    noise = np.random.uniform(0, 1, (int(noise_size), self.latent_size))

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
                    for i in range(k):
                        if i == 0:
                            X = np.concatenate((data_batch, self.names['generated_data' + str(i)]))
                        else:
                            X = np.concatenate((X, self.names['generated_data' + str(i)]))
                    Y = np.array([1] * batch_size + [0] * int(noise_size))

                    # Train discriminator
                    discriminator_loss = self.discriminator.train_on_batch(X, Y)
                    train_history['discriminator_loss'].append(discriminator_loss)

                    # Get the target value of sub-generator
                    p_value = self.discriminator.predict(data_x)
                    p_value = pd.DataFrame(p_value)
                    for i in range(k):
                        self.names['T' + str(i)] = p_value.quantile(i/k)
                        self.names['trick' + str(i)] = np.array([float(self.names['T' + str(i)])] * noise_size)

                    # Train generator
                    noise = np.random.uniform(0, 1, (int(noise_size), self.latent_size))
                    if stop == 0:
                        for i in range(k):
                            self.names['sub_generator' + str(i) + '_loss'] = self.names['combine_model' +
                                                                                        str(i)].train_on_batch(noise, self.names['trick' + str(i)])
                            train_history['sub_generator{}_loss'.format(i)].append(self.names['sub_generator' + str(i) + '_loss'])
                    else:
                        for i in range(k):
                            self.names['sub_generator' + str(i) + '_loss'] = self.names['combine_model' + str(i)].evaluate(noise, self.names['trick' + str(i)])
                            train_history['sub_generator{}_loss'.format(i)].append(self.names['sub_generator' + str(i) + '_loss'])

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
                    print(result)

                    # Calculate the AUC
                    inlier_parray = result.loc[lambda df: df.y == 0.0, 'p'].values
                    outlier_parray = result.loc[lambda df: df.y == 1.0, 'p'].values
                    print(inlier_parray)
                    print(outlier_parray)
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
