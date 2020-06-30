from training.train_cnn_timeseries import main as train_cnn_timeseries
from training.train_lstm import main as train_lstm
from testing.test_cnn_timeseries import main as test_cnn_timeseries
from testing.test_lstm import main as test_lstm


if __name__ == '__main__':
    # train_cnn_timeseries('dataset/train')
    # train_lstm('dataset/train')

    cnn_results = test_cnn_timeseries('models/cnn_timeseries_16_40_5_1.h5', 'dataset/test')
    lstm_results = test_lstm('models/lstm_16_40_5.h5', 'dataset/test')

    print('CNN Results \t|\t\tmse: {},\t\tmae: {}'.format(*cnn_results[:2]))
    print('LSTM Results\t|\t\tmse: {},\t\tmae: {}'.format(*lstm_results[:2]))
