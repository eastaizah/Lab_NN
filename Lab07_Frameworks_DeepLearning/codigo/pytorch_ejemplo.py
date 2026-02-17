"""
Ejemplo de Red Neuronal en PyTorch
Lab 07: Frameworks de Deep Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class RedNeuronalPyTorch(nn.Module):
    """Red neuronal simple en PyTorch."""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super(RedNeuronalPyTorch, self).__init__()
        
        # Definir capas
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def entrenar_pytorch():
    """Ejemplo completo de entrenamiento en PyTorch."""
    
    print("=" * 70)
    print("EJEMPLO DE PYTORCH")
    print("=" * 70)
    
    # 1. Preparar datos
    print("\n1. Preparando datos...")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convertir a tensores de PyTorch
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Crear DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test: {len(X_test)} muestras")
    
    # 2. Crear modelo
    print("\n2. Creando modelo...")
    model = RedNeuronalPyTorch(input_size=2, hidden_sizes=[16, 8], output_size=2)
    print(f"   Arquitectura: {model}")
    
    # 3. Definir pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 4. Entrenar
    print("\n3. Entrenando...")
    num_epochs = 100
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Evaluación
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test).float().mean()
            test_accuracies.append(accuracy.item())
        
        train_losses.append(epoch_loss / len(train_loader))
        
        if (epoch + 1) % 20 == 0:
            print(f"   Época {epoch+1}/{num_epochs} - Loss: {train_losses[-1]:.4f} - Acc: {accuracy:.4f}")
    
    # 5. Resultados finales
    print(f"\n4. Resultados finales:")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        final_accuracy = (predicted == y_test).float().mean()
    
    print(f"   Test Accuracy: {final_accuracy:.4f}")
    
    # 6. Visualizar
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Curva de Aprendizaje - PyTorch')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, linewidth=2, color='green')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.title('Accuracy en Test - PyTorch')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("✓ Entrenamiento en PyTorch completado!")
    print("=" * 70)


if __name__ == "__main__":
    entrenar_pytorch()
