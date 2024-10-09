from models.TS.GRUPipeline import GRUNetwork
from torch import optim
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset

import boto3


SYMBOL = "SOLUSDT"

session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='ru-central1'
)

bucket_name = 'test-actions'

x_train_file = f'train/trainX_{SYMBOL}.pt'
y_train_file = f'train/trainY_{SYMBOL}.pt'
x_test_file = f'test/testX_{SYMBOL}.pt'
y_test_file = f'test/testY_{SYMBOL}.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s3.download_file(bucket_name, x_train_file, x_train_file[6:])
s3.download_file(bucket_name, y_train_file, y_train_file[6:])
s3.download_file(bucket_name, x_test_file, x_test_file[5:])
s3.download_file(bucket_name, y_test_file, y_test_file[5:])

print(x_train_file[6:])
x_train, y_train = torch.load(f'trainX_{SYMBOL}.pt', weights_only=True).to(torch.float32).to(device), torch.load(
    f'trainY_{SYMBOL}.pt', weights_only=True).to(torch.float32).to(device)
x_test, y_test = torch.load(f'testX_{SYMBOL}.pt', weights_only=True).to(torch.float32).to(device), torch.load(
    f'testY_{SYMBOL}.pt', weights_only=True).to(torch.float32).to(device)

mean, std = x_train.squeeze().reshape(-1).detach().mean().item(), x_train.squeeze().reshape(
    -1).detach().std().item()
dataset = TensorDataset(x_train, y_train)
validation = TensorDataset(x_test, y_test)

batch_size = 5
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(validation, batch_size=2, shuffle=True)

gru = GRUNetwork(1, 30, 2, 10, components=1, mean=mean, scale=std)
gru.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(gru.parameters(), lr=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.3, last_epoch=10)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
print(optimizer.state_dict)
# Обучение модели
num_epochs = 50
patience = 3
target_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    for i, (batch_x, batch_y) in enumerate(dataloader):
        gru.train()
        outputs = gru(batch_x)
        optimizer.zero_grad()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        print("step =", i)
    scheduler.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    gru.eval()  # Переключаем модель в режим оценки
    val_loss = 0.0
    with torch.no_grad():  # Выключаем вычисление градиентов
        for batch_x, batch_y in val_dataloader:
            outputs = gru(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)  # Средняя ошибка на валидации
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    # Реализация ранней остановки
    if val_loss < target_loss:
        target_loss = val_loss
        patience_counter = 0  # Сбрасываем счетчик
    else:
        patience_counter += 1  # Увеличиваем счетчик

    # Проверяем, если сохранять, если ошибка стабильна
    if patience_counter >= patience:
        print("Ошибка валидации не уменьшается, остановка обучения.")
        break

model_file = f'gru_{SYMBOL}.onnx'
torch.onnx.export(gru, x_train, model_file, export_params=True)
s3.upload_file(f'gru_{SYMBOL}.onnx', bucket_name, 'model/' + model_file)

weights_file = f'weights_{SYMBOL}.pt'
torch.save(gru.state_dict(), weights_file)
s3.upload_file(weights_file, bucket_name, 'model/' + weights_file)