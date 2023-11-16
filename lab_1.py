import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Загрузка данных
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Определение DataLoader для тестирования
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Определение многослойного перцептрона
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Инициализация функции потерь
criterion = nn.CrossEntropyLoss()

# Размеры batch для тестирования
batch_sizes = [16, 32, 64, 128]

for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")
    # Инициализация модели и оптимизатора
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Инициализация DataLoader для обучения
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Обучение модели
    epochs = 5
    accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = correct_test / total_test
        accuracies.append(test_acc)

        print(f'Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}')

    # Визуализация процесса обучения для текущего размера batch
    plt.plot(range(1, epochs + 1), accuracies, label=f'Batch Size: {batch_size}')

# Визуализация всех графиков
plt.title('Accuracy vs Epochs for Different Batch Sizes')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
