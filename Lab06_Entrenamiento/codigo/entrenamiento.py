"""
Implementación Completa de Entrenamiento de Redes Neuronales
Laboratorio 06: Entrenamiento
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    """Red neuronal completa con entrenamiento."""
    
    def __init__(self, layer_sizes, activations, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1
        
        # Inicializar parámetros (He initialization)
        self.params = {}
        for i in range(self.num_layers):
            self.params[f'W{i+1}'] = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i])
            self.params[f'b{i+1}'] = np.zeros((layer_sizes[i+1], 1))
        
        # Caches para backprop
        self.cache = {}
        
        # Historia de entrenamiento
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_backward(self, dA, Z):
        dZ = dA.copy()
        dZ[Z <= 0] = 0
        return dZ
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    
    def sigmoid_backward(self, dA, Z):
        s = self.sigmoid(Z)
        return dA * s * (1 - s)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def forward(self, X):
        """Forward pass completo."""
        A = X
        self.cache['A0'] = X
        
        for i in range(1, self.num_layers + 1):
            Z = self.params[f'W{i}'] @ A + self.params[f'b{i}']
            self.cache[f'Z{i}'] = Z
            
            if self.activations[i-1] == 'relu':
                A = self.relu(Z)
            elif self.activations[i-1] == 'sigmoid':
                A = self.sigmoid(Z)
            elif self.activations[i-1] == 'softmax':
                A = self.softmax(Z)
            else:  # linear
                A = Z
            
            self.cache[f'A{i}'] = A
        
        return A
    
    def compute_loss(self, Y_hat, Y, loss_type='cross_entropy'):
        """Calcular pérdida."""
        m = Y.shape[1]
        
        if loss_type == 'cross_entropy':
            epsilon = 1e-15
            Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(Y * np.log(Y_hat), axis=0))
        elif loss_type == 'binary_cross_entropy':
            epsilon = 1e-15
            Y_hat = np.clip(Y_hat, epsilon, 1 - epsilon)
            loss = -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        else:  # MSE
            loss = np.mean((Y_hat - Y) ** 2)
        
        return loss
    
    def backward(self, Y, loss_type='cross_entropy'):
        """Backward pass completo."""
        m = Y.shape[1]
        grads = {}
        
        # Inicializar
        A_last = self.cache[f'A{self.num_layers}']
        
        if loss_type == 'cross_entropy' and self.activations[-1] == 'softmax':
            dZ = (A_last - Y) / m
        else:
            dA = -(np.divide(Y, A_last) - np.divide(1 - Y, 1 - A_last))
            dZ = self.sigmoid_backward(dA, self.cache[f'Z{self.num_layers}'])
        
        # Backprop a través de las capas
        for i in range(self.num_layers, 0, -1):
            A_prev = self.cache[f'A{i-1}']
            
            grads[f'W{i}'] = dZ @ A_prev.T
            grads[f'b{i}'] = np.sum(dZ, axis=1, keepdims=True)
            
            if i > 1:
                dA_prev = self.params[f'W{i}'].T @ dZ
                
                if self.activations[i-2] == 'relu':
                    dZ = self.relu_backward(dA_prev, self.cache[f'Z{i-1}'])
                elif self.activations[i-2] == 'sigmoid':
                    dZ = self.sigmoid_backward(dA_prev, self.cache[f'Z{i-1}'])
        
        return grads
    
    def update_parameters(self, grads):
        """Actualizar parámetros con gradient descent."""
        for i in range(1, self.num_layers + 1):
            self.params[f'W{i}'] -= self.learning_rate * grads[f'W{i}']
            self.params[f'b{i}'] -= self.learning_rate * grads[f'b{i}']
    
    def train(self, X_train, Y_train, X_val=None, Y_val=None, epochs=100, 
              batch_size=32, verbose=True, early_stopping_patience=None):
        """
        Entrenar la red neuronal.
        """
        m = X_train.shape[1]
        num_batches = m // batch_size
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle data
            permutation = np.random.permutation(m)
            X_shuffled = X_train[:, permutation]
            Y_shuffled = Y_train[:, permutation]
            
            epoch_loss = 0
            
            # Mini-batch training
            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]
                
                # Forward
                Y_hat = self.forward(X_batch)
                loss = self.compute_loss(Y_hat, Y_batch)
                epoch_loss += loss
                
                # Backward
                grads = self.backward(Y_batch)
                
                # Update
                self.update_parameters(grads)
            
            # Métricas de entrenamiento
            train_loss = epoch_loss / num_batches
            Y_train_pred = self.forward(X_train)
            train_acc = self.compute_accuracy(Y_train_pred, Y_train)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validación
            if X_val is not None:
                Y_val_pred = self.forward(X_val)
                val_loss = self.compute_loss(Y_val_pred, Y_val)
                val_acc = self.compute_accuracy(Y_val_pred, Y_val)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping
                if early_stopping_patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"\nEarly stopping en época {epoch+1}")
                        break
            
            # Imprimir progreso
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                if X_val is not None:
                    print(f"Época {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
                else:
                    print(f"Época {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")
    
    def compute_accuracy(self, Y_hat, Y):
        """Calcular accuracy."""
        predictions = np.argmax(Y_hat, axis=0)
        labels = np.argmax(Y, axis=0)
        return np.mean(predictions == labels)
    
    def predict(self, X):
        """Hacer predicciones."""
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=0)


def plot_training_history(history):
    """Visualizar historia de entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pérdida
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if history['val_loss']:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Pérdida', fontsize=12)
    ax1.set_title('Curva de Aprendizaje - Pérdida', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    if history['val_acc']:
        ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Curva de Aprendizaje - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("=" * 70)
    print("ENTRENAMIENTO COMPLETO DE RED NEURONAL")
    print("=" * 70)
    
    # 1. Generar datos
    print("\n1. Generando datos...")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Convertir a one-hot
    Y = np.eye(2)[y].T
    X = X.T
    
    # Dividir datos
    X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    X_train, X_val, X_test = X_train.T, X_val.T, X_test.T
    Y_train, Y_val, Y_test = Y_train.T, Y_val.T, Y_test.T
    
    print(f"   Train: {X_train.shape[1]} muestras")
    print(f"   Val: {X_val.shape[1]} muestras")
    print(f"   Test: {X_test.shape[1]} muestras")
    
    # 2. Crear y entrenar red
    print("\n2. Creando red neuronal...")
    model = NeuralNetwork(
        layer_sizes=[2, 16, 8, 2],
        activations=['relu', 'relu', 'softmax'],
        learning_rate=0.1
    )
    
    print("   Arquitectura: 2 → 16 → 8 → 2")
    print("   Activaciones: ReLU → ReLU → Softmax")
    
    print("\n3. Entrenando...")
    model.train(
        X_train, Y_train,
        X_val, Y_val,
        epochs=200,
        batch_size=32,
        verbose=True,
        early_stopping_patience=20
    )
    
    # 3. Evaluar
    print("\n4. Evaluando en conjunto de prueba...")
    Y_test_pred = model.forward(X_test)
    test_acc = model.compute_accuracy(Y_test_pred, Y_test)
    test_loss = model.compute_loss(Y_test_pred, Y_test)
    
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # 4. Visualizar
    print("\n5. Generando visualizaciones...")
    plot_training_history(model.history)
    
    print("\n" + "=" * 70)
    print("✓ Entrenamiento completado exitosamente!")
    print("=" * 70)


if __name__ == "__main__":
    main()
