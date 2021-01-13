import multiprocessing
import sys
from time import time
import argparse
from DataSet import DataSet
from Evaluator import evaluate, saveResult
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
sess = tf.Session(config=config)
KTF.set_session(sess)
from keras import initializers
from keras.regularizers import l1, l2
from keras.layers import Embedding, Input, Dense, Flatten, concatenate, dot, Add, Dropout, BatchNormalization, \
    Activation, LeakyReLU, Lambda, Multiply

from keras.optimizers import Adam, Adamax, Adagrad, Adadelta, SGD, Nadam
from keras.layers import Conv1D as cnn1
from keras.layers.normalization import BatchNormalization  as bn

import keras
import numpy as np

from keras.layers import Dense, Conv1D, BatchNormalization, Activation, Minimum
from keras.layers import AveragePooling1D, Input, GlobalAveragePooling1D, Concatenate, Reshape
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model

def main():
    parser = argparse.ArgumentParser(description="Parameter Settings")
    parser.add_argument('--dataType', default='LDA5ok_rt', type=str, help='Type of data:rt|tp.')
    parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-process.')
    parser.add_argument('--density', default=list(np.arange(0.05, 0.20, 0.05)), type=list, help='Density of matrix.')
    parser.add_argument('--epochNum', default=100, type=int, help='Numbers of epochs per run.')
    parser.add_argument('--batchSize', default=256, type=int, help='Size of a batch.')
    parser.add_argument('--layers', default=[2048, 1024, 512, 256, 128, 64, 32, 16, 1], type=list,
                        help='Layers of MLP.')
    parser.add_argument('--regLayers', default=[0, 0, 0, 0, 0, 0, 0, 0, 0], type=list, help='Regularizers.')
    parser.add_argument('--optimizer', default=Nadam, type=str, help='The optimizer:Adam|Adamax|Nadam|Adagrad.')
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate of the model.')
    parser.add_argument('--decay', default=0.000 , type=float, help='Decay factor of learning rate.')
    parser.add_argument('--verbose', default=3, type=int, help='Iterations per evaluation.')
    parser.add_argument('--store', default=False, type=bool, help='Whether to store the model and result.')
    parser.add_argument('--modelPath', default='./Model', type=str, help='Path to save the model.')
    parser.add_argument('--resultPath', default='./Result', type=str, help='Path to save the result.')
    args = parser.parse_args()

    if args.parallel:
        pool = multiprocessing.Pool()
        for density in args.density:
            pool.apply_async(LDCF, (args, density))
        pool.close()
        pool.join()
    else:
        for density in args.density:
            LDCF(args, density)


class LDCF:

    def __init__(self, args, density):

        self.dataset = DataSet(args.dataType, density)
        self.dataType = self.dataset.dataType
        self.density = self.dataset.density
        self.shape = self.dataset.shape
        self.train = self.dataset.train
        self.test = self.dataset.test
        self.epochNum = args.epochNum

        self.batchSize = args.batchSize
        self.layers = args.layers
        self.regLayers = args.regLayers
        self.lr = args.lr
        self.decay = args.decay
        self.optimizer = args.optimizer
        self.verbose = args.verbose
        self.store = args.store
        self.modelPath = args.modelPath
        self.resultPath = args.resultPath

        self.model = self.compile_model()

        self.run()

    def run(self):
        # Initialization
        x_test, y_test = self.dataset.getTestInstance(self.test)

        sys.stdout.write('\rInitializing...')
        mae, rmse = evaluate(self.model, x_test, y_test)
        sys.stdout.write('\rInitializing completes.MAE = %.4f|RMSE = %.4f.\n' % (mae, rmse))
        best_mae, best_rmse, best_epoch = mae, rmse, -1
        evalResults = np.zeros((self.epochNum, 3))
        # Training model
        print('=' * 14 + 'Start Training' + '=' * 22)
        for epoch in range(self.epochNum):
            sys.stdout.write('\rEpoch %d starts...' % epoch)
            start = time()
            x_train, y_train = self.dataset.getTrainInstance(self.train)

            # Training
            history = self.model.fit(x_train, y_train, batch_size=self.batchSize, epochs=1, verbose=0, shuffle=True)
            # , callbacks=[TensorBoard(log_dir='./Log')])
            end = time()
            sys.stdout.write('\rEpoch %d ends.[%.1fs]' % (epoch, end - start))
            # print(end - start)
            # Evaluation
            if epoch % self.verbose == 0:
                    sys.stdout.write('\rEvaluating Epoch %d...' % epoch)
                    print("\n")
                    if epoch>0:
                        mae, rmse = evaluate(self.model, x_test, y_test)
                        loss = history.history['loss'][0]
                        sys.stdout.write('\rEvaluating completes.[%.1fs] ' % (time() - end))
                        if mae + rmse < best_mae + best_rmse:
                              best_mae, best_rmse, best_epoch = mae, rmse, epoch
                        # if self.store:
                        # self.saveModel(self.model)
                        evalResults[epoch, :] = [mae, rmse,loss]
                        sys.stdout.write('\rEpoch %d : MAE = %.4f|RMSE = %.4f|Loss = %.4f\n' % (epoch, mae, rmse, loss))
        print('=' * 14 + 'Training Complete!' + '=' * 18)
        print('The best is at epoch %d : MAE = %.4f|RMSE = %.4f.' % (best_epoch, best_mae, best_rmse))
        if self.store:
            saveResult(self.resultPath, self.dataType, self.density, evalResults, ['MAE', 'RMSE','loss'])
            print('The model is stored in %s.' % self.modelPath)
            print('The result is stored in %s.' % self.resultPath)


    def compile_model(self):

        _model = self.build_model(self.shape[0], self.shape[1], self.layers, self.regLayers)
        _model.compile(optimizer=self.optimizer(lr=self.lr), loss='mean_absolute_error')

        return _model

    def build_model(self, num_users, num_item, layers, reg_layers):

        assert len(layers) == len(reg_layers)

        # Input Layer
        user_id_input = Input(shape=(1,), dtype='int64', name='user_id_input')
        item_id_input = Input(shape=(1,), dtype='int64', name='item_id_input')
        user_qz_input = Input(shape=(2,), dtype='int64', name='user_qz_input')
        user_qz2_input = Input(shape=(5,), dtype='int64', name='user_qz2_input')
        item_qz_input = Input(shape=(2,), dtype='int64', name='item_qz_input')
        item_qz2_input = Input(shape=(5,), dtype='int64', name='item_qz2_input')

        user_id_embedding = self.getEmbedding(num_users, int(layers[0] / 4), 1, reg_layers[0], 'user_id_embedding')
        item_id_embedding = self.getEmbedding(num_item, int(layers[0] / 4), 1, reg_layers[0], 'item_id_embedding')
        user_qz_embedding = self.getEmbedding(num_users, int(layers[0] / 4), 2, reg_layers[0], 'user_qz_embedding')
        user_qz2_embedding = self.getEmbedding(num_users, int(layers[0] / 4), 5, reg_layers[0], 'user_qz2_embedding')
        servie_embedding = self.getEmbedding(num_item, int(layers[0] / 4), 2, reg_layers[0], 'servie_embedding')
        servie2_embedding = self.getEmbedding(num_item, int(layers[0] / 4),5 , reg_layers[0], 'servie2_embedding')

        user_id_latent2 = user_id_embedding(user_id_input)
        item_id_latent2 = item_id_embedding(item_id_input)
        user_qz_latent2 = user_qz_embedding(user_qz_input)
        user_qz_latent3 = user_qz2_embedding(user_qz2_input)
        item_qz_latent2 = servie_embedding(item_qz_input)
        item_qz_latent3 = servie2_embedding(item_qz2_input)

        # concatenate
        predict_user_vector2 = concatenate([user_id_latent2,user_qz_latent2,user_qz_latent3], axis=1)
        predict_item_vector2 = concatenate([item_id_latent2,item_qz_latent2,item_qz_latent3], axis=1)
        # lf=concatenate([user_qz_latent3, item_qz_latent3], axis=1)
        inputs = concatenate([predict_user_vector2, predict_item_vector2], axis=1)
        # inputs=concatenate([user_id_latent2,item_id_latent2])
  #m5 n1
        # m+n *1.5
        # x = cnn1(int(layers[0] / 4), 12)(inputs)
        # x = bn()(x)
        # x = Activation('relu')(x)
        #
        # # M+N
        # x2 = cnn1(int(layers[0] / 4), 8, strides=5)(inputs)
        # x2 = bn()(x2)
        # x2 = Activation('relu')(x2)
        #
        # # N
        # x3 = cnn1(int(layers[0] / 4), 3, strides=5)(inputs)
        # x3 = bn()(x3)
        # x3 = Activation('relu')(x3)
        #
        # # m+n *2
        # x5 = cnn1(int(layers[0] / 4), 16)(inputs)
        # x5 = bn()(x5)
        # x5 = Activation('relu')(x5)
        #
        #
        # x = concatenate([x,x2,x3,x5], axis=1)

        x = Flatten()(inputs)
        # x = concatenate([x, mlp_vector], axis=0)
        x = Dense(int(layers[0] / 2))(x)
        x = bn()(x)

        x = Activation('relu')(x)

        x = Dense(int(layers[0] / 4))(x)
        x = bn()(x)
        x = Activation('relu')(x)
        prediction = Dense(1, activation='linear')(x)

        _model = Model(
            inputs=[user_id_input,item_id_input,user_qz_input,user_qz2_input,item_qz_input,item_qz2_input],
            outputs=prediction)


        print(_model.summary())
        return _model


    # One-hot encoding + 0-layer mlp
    def getEmbedding(self, input_dim, output_dim, input_length, reg_layers, name):
        _Embedding = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length,
                               embeddings_initializer=initializers.random_normal(),
                               embeddings_regularizer=l1(0), name=name)
        return _Embedding

    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return shape1[0], 1

    def saveModel(self, _model):
        _model.save_weights(self.modelPath + '/%s_%.2f_%s.h5'
                            % (self.dataType, self.density, self.layers), overwrite=True)


if __name__ == '__main__':
    main()
