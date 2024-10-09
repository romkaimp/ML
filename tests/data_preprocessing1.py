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
    API_KEY = ''
    API_SECRET = ''
    cl = HTTP(
    )

    START_TIMESTAMP = '1713929617184'
    time_series = 1000  # количество изначальных временных рядов в разные промежутки времени, с количеством измерений t_s_l
    time_series_length = 30  # количество сэмплов из ts, на которые мы делим временной ряд
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
        if kline.get('result', {}).get('list', [])[-1][0] == time_stamps[-1]:
            time_stamps[-1] = kline.get('result', {}).get('list', [])[-2][0]
            kline = cl.get_kline(
                category="spot",
                symbol=SYMBOL,
                start=time_stamps[-1],
                interval=1,
                limit=time_series_real_l
            )
        if i % 59 == 0:
            time.sleep(5)
        time_stamps.append(kline.get('result', {}).get('list', [])[0][0])

        volumes[i * time_series_real_l:(i + 1) * time_series_real_l] = np.array(
            [i[5] for i in kline.get('result', {}).get('list', [])[::-1]], dtype=float)
        all_price[i * time_series_real_l:(i + 1) * time_series_real_l] = np.array(
            [i[4] for i in kline.get('result', {}).get('list', [])[::-1]], dtype=float)

        prices = np.array([i[4] for i in kline.get('result', {}).get('list', [])[::-1]], dtype=float)
        train_data[i] = np.array(
            [prices[i:time_series_sample + i] for i in range(time_series_real_l - time_series_sample + 1)], dtype=float
        ).reshape(time_series_real_l - time_series_sample + 1, time_series_sample)

    train_size = int(time_series * 0.8)
    prediction_length = 10
    idxs = np.random.choice((time_series-1), size=train_size)
    train = torch.tensor(train_data[idxs])

    last_idxs = np.array([i for i in range(0, time_series-1) if not np.isin(i, idxs)])
    # Получение тестовых данных

    test = torch.tensor(train_data[last_idxs])
    all_price = torch.tensor(all_price)
    volumes = torch.tensor(volumes)

    print(__file__, "saving")

    local_names = [f'train_{SYMBOL}.pt', f'test_{SYMBOL}.pt', f'all_price_{SYMBOL}.pt', f'volumes_{SYMBOL}.pt']
    torch.save(train, f'train_{SYMBOL}.pt')
    torch.save(test, f'test_{SYMBOL}.pt')
    torch.save(all_price, f'all_price_{SYMBOL}.pt')
    torch.save(volumes, f'volumes_{SYMBOL}.pt')

    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net'
    )
    bucket_name = 'test-actions'

    for local_name in local_names:
        if local_name.startswith('train'):
            s3_file_path = f'train/{local_name}'
        elif local_name.startswith('test'):
            s3_file_path = f'test/{local_name}'
        else:
            s3_file_path = f'train/{local_name}'
        s3.upload_file(local_name, bucket_name, s3_file_path)

if __name__ == "__main__":
    smb = parse_symbol()
