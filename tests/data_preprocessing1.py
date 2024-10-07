from pybit.unified_trading import HTTP
import numpy as np
import time
from torch.nn.utils.rnn import pad_sequence
import torch
import click
import boto3

@click.command()
@click.argument('symbol')
def parse_symbol(symbol):
    print(symbol)
    main(symbol)
    return symbol

def main(SYMBOL):
    print(__file__, "in main")
    API_KEY = '9ka7e7xrx4YSQl5Gka'
    API_SECRET = 'JGSXmnFJjbbt8clfLkwegj32LVDvov3HgRNd'
    cl = HTTP(
    )

    START_TIMESTAMP = '1711929617184'
    time_series = 1000  # количество изначальных временных рядов в разные промежутки времени, с количеством измерений t_s_l
    time_series_length = 10  # количество сэмплов из ts, на которые мы делим временной ряд
    time_series_sample = 20  # количество сделок в одном временном ряду

    time_series_real_l = time_series_length + time_series_sample - 1
    time_stamps = [START_TIMESTAMP]

    train_data = np.ndarray(shape=(time_series, time_series_length, time_series_sample))
    volumes = np.ndarray(shape=time_series_real_l * time_series)
    all_price = np.ndarray(shape=time_series_real_l * time_series)

    for i in range(time_series):
        if i % 100 == 0:
            print(__file__, f"i = {i}")
        kline = cl.get_kline(
            category="spot",
            symbol=SYMBOL,
            start=time_stamps[-1],
            interval=1,
            limit=time_series_real_l
        )
        if i % 116 == 0:
            time.sleep(5)
        time_stamps.append(str(int(kline.get('result', []).get('list', [])[0][0]) + 10000))

        volumes[i * time_series_real_l:(i + 1) * time_series_real_l] = np.array(
            [i[5] for i in kline.get('result', []).get('list', [])[::-1]], dtype=float)
        all_price[i * time_series_real_l:(i + 1) * time_series_real_l] = np.array(
            [i[4] for i in kline.get('result', []).get('list', [])[::-1]], dtype=float)

        prices = np.array([i[4] for i in kline.get('result', []).get('list', [])[::-1]], dtype=float)
        train_data[i] = np.array(
            [prices[i:time_series_sample + i] for i in range(time_series_real_l - time_series_sample + 1)], dtype=float
        ).reshape(time_series_real_l - time_series_sample + 1, time_series_sample)

    print(train_data.shape)
    train_size = int(time_series * 0.8)
    prediction_length = 10
    idxs = np.random.choice((time_series-1), size=train_size)
    x_train, y_train, = torch.tensor(train_data[idxs]), torch.tensor(train_data[idxs+1][:, 0, :prediction_length])

    last_idxs = np.array([i for i in range(0, time_series-1) if not np.isin(i, idxs)])
    # Получение тестовых данных

    x_test, y_test = torch.tensor(train_data[last_idxs]), torch.tensor(train_data[last_idxs + 1][:, 0, :prediction_length])
    print(__file__, "synthetic data")

    synthetic_data_size = 2000
    #synthetic_data_x = np.ndarray(shape=(synthetic_data_size, time_series_length, time_series_sample))
    #synthetic_data_y = np.ndarray(shape=(synthetic_data_size, prediction_length))
    synthetic_data_x = []
    synthetic_data_y = []
    for i in range(synthetic_data_size):
        idx = int(np.random.rand()*(train_size-2)) #  можно взять предпоследний элемент, потому что по следующему эл-ту будет идти ошибка
        size_s = min(time_series_length-1, max(2, int(np.random.normal(time_series_length/2, time_series_length/4))))
        samples_idx = np.random.choice(time_series_length-1, size = size_s, replace=True)
        samples_idx.sort()
        samples_idx = np.append(samples_idx, time_series_length-1)
        el_x = x_train[idx][samples_idx]
        el_y = x_train[idx+1][0, :prediction_length]
        synthetic_data_x.append(el_x.clone().detach().requires_grad_(True))
        synthetic_data_y.append(el_y.clone().detach().requires_grad_(True))
    synthetic_data_x = pad_sequence(synthetic_data_x, batch_first=True)
    synthetic_data_y = torch.stack(synthetic_data_y)

    print(__file__, "saving")

    local_names = [f'train_x_{SYMBOL}.pt', f'train_y_{SYMBOL}.pt', f'test_x_{SYMBOL}.pt', f'test_y_{SYMBOL}.pt']
    torch.save(torch.concat((x_train, synthetic_data_x), dim=0), f'train_x_{SYMBOL}.pt')
    torch.save(torch.concat((y_train, synthetic_data_y), dim=0), f'train_y_{SYMBOL}.pt')
    torch.save(x_test, f'test_x_{SYMBOL}.pt')
    torch.save(y_test, f'test_y_{SYMBOL}.pt')

    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net'
    )
    bucket_name = 'test-actions'

    for local_name in local_names:
        s3_file_path = f'test/{local_name}' if local_name.startswith('test') else f'train/{local_name}'
        s3.upload_file(local_name, bucket_name, s3_file_path)

if __name__ == "__main__":
    smb = parse_symbol()
