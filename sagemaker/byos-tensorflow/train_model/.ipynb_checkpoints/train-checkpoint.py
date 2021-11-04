import argparse
import numpy as np
import os
import tensorflow as tf

from model_def import get_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def parse_args():
    
    parser = argparse.ArgumentParser()

    # 사용자가 전달한 하이퍼 파라미터를 command-line argument로 전달받아 사용함
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()


def get_train_data(train_dir):
    
    x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    print('x train', x_train.shape,'y train', y_train.shape)

    return x_train, y_train


def get_test_data(test_dir):
    
    x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'))
    print('x test', x_test.shape,'y test', y_test.shape)

    return x_test, y_test
   

if __name__ == "__main__":
    
    # 환경변수 또는 사용자 지정 hyperparameter로 전달된 argument를 읽는다.
    args, _ = parse_args()
    
    # ------------------------------------------------------------------------------------------------
    # TO DO 
    # training data를 가져온다. 위 코드에서 읽은 argument 중 'train'으로 전달된 값을 사용함. 
    # parse_args()를 통해 환경변수 'SM_CHANNEL_TRAIN'로 전달된 경로 'opt/ml/input/train/'가 'arg.train'으로 지정됨
    # x_train, y_train = get_train_data(args.train)
    # x_test, y_test = get_test_data(args.test)
    
    x_train, y_train = <TO DO>
    x_test, y_test = <TO DO>
    
    # ------------------------------------------------------------------------------------------------
    
    device = '/cpu:0' 
    print(device)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    with tf.device(device):
        
        model = get_model()
        optimizer = tf.keras.optimizers.SGD(learning_rate)
        model.compile(optimizer=optimizer, loss='mse')    
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(x_test, y_test))

        # evaluate on test set
        scores = model.evaluate(x_test, y_test, batch_size, verbose=2)
        print("\nTest MSE :", scores)
        
        # ------------------------------------------------------------------------------------------------
        # TO DO 
        # 결과모델 저장 - 'args.model_dir'에는 'SM_MODEL_DIR' 환경변수를 통해 지정된 '/opt/ml/model/' 경로가 지정된다.
        # model.save(args.model_dir + '/1')
        
        model.save(<TO DO>)
        
        # ------------------------------------------------------------------------------------------------
        