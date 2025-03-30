import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from rtdl_num_embeddings import LinearReLUEmbeddings, PiecewiseLinearEmbeddings, compute_bins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.datasets import fetch_california_housing
import math

# === Гиперпараметры ===
N_MODELS = 32  # Число моделей в ансамбле
D_EMBEDDING = 8  # Размерность эмбеддингов
HIDDEN_DIM = 32  # Размер скрытых слоёв
EPOCHS = 1000
LR = 1e-3


def make_mlp(input_dim, output_dim=1, hidden_dim=HIDDEN_DIM):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TabM(nn.Module):
    def __init__(self, n_cont_features, num_cat_features, cat_cardinalities, n_models=N_MODELS, bins=None):
        super().__init__()
        self.n_models = n_models

        # Общие эмбеддинги
        self.embedding = PiecewiseLinearEmbeddings(bins, D_EMBEDDING, activation=False, version="B")
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_card, D_EMBEDDING) for cat_card in cat_cardinalities
        ])

        # Умножаемые обучаемые векторы для каждого MLP
        self.scaling_vectors = nn.Parameter(torch.randn(n_models, n_cont_features * D_EMBEDDING + num_cat_features))

        # Ансамбль MLP
        self.mlps = nn.ModuleList([make_mlp(n_cont_features * D_EMBEDDING + num_cat_features) for _ in range(n_models)])

    def forward(self, x_cont, x_cat):
        # Применяем эмбеддинги
        cont_emb = self.embedding(x_cont).flatten(start_dim=1)
        cat_emb = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1) if x_cat.shape[
                                                                                                           1] > 0 else torch.zeros(
            x_cont.shape[0], 0, device=x_cont.device)
        x = torch.cat([cont_emb, cat_emb], dim=1)

        # Пропускаем через ансамбль
        outputs = []
        for i in range(self.n_models):
            scaled_x = x * self.scaling_vectors[i]  # Поэлементное умножение
            outputs.append(self.mlps[i](scaled_x))

        return torch.stack(outputs, dim=1)  # (batch, n_models, 1)



# === Подготовка данных ===
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test,
                                                                                        dtype=torch.float32).unsqueeze(
    1)
print(y_train.shape)
bins = compute_bins(
    X_train,
    # NOTE: requires scikit-learn>=1.0 to be installed.
    tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
    y=y_train.reshape(-1),
    regression=True,
)

# === Обучение ===
model = TabM(X_train.shape[1], num_cat_features=0, cat_cardinalities=[], bins=bins)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    outputs = model(X_train,
                    torch.empty(X_train.shape[0], 0, dtype=torch.long, device=X_train.device))

    batch_losses = []

    # Цикл по каждой модели для подсчета лосса
    for i in range(outputs.shape[1]):  # n_models — количество моделей в ансамбле
        model_output = outputs[:, i, :]  # Выбираем выходы i-й модели
        loss = criterion(model_output, y_train)  # Считаем лосс для данной модели
        batch_losses.append(loss)

    # Усредняем лоссы по всем моделям
    mean_loss = torch.stack(batch_losses).mean()

    # Оптимизация
    mean_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# === Тестирование ===
with torch.no_grad():
    outputs = model(X_test, torch.empty(X_test.shape[0], 0, dtype=torch.long, device=X_test.device))
    test_loss = criterion(outputs.mean(dim=1), y_test)
    print(f"Final Test Loss: {math.sqrt(test_loss.item()):.4f}")

print("GB RMSE: 0.4575")
