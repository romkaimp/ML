import subprocess
from pybit.unified_trading import HTTP
import os
from pathlib import Path

# Создайте объект клиента API
client = HTTP()

# Получите информацию о рынке
market_data = client.get_tickers(category='spot')

# Отсортируйте токены по объему торгов, чтобы найти самые популярные
sorted_tickers = sorted(filter(lambda x: x.get('symbol').endswith('USDT') and not x.get('symbol').startswith('USD'), market_data.get('result', {}).get('list', [])), key=lambda x: float(x['turnover24h']), reverse=True)

# Выведите топ несколько популярных токенов
top_tickers = sorted_tickers[:10]  # например, топ 5
#for ticker in top_tickers:
#    print(f"Токен: {ticker['symbol']}, Объём за 24ч: {ticker['turnover24h']}")

symbols = [ticker['symbol'] for ticker in top_tickers]

for symbol in symbols:
    dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir, 'data_preprocessing1.py')
    dir = Path(os.path.join(dir, '..\\.venv\\Scripts\\python.exe')).resolve()
    cp = subprocess.run([dir, file, symbol])
    print(cp.returncode, cp.stdout)