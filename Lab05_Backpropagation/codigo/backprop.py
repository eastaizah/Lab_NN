"""
Implementación de Backpropagation desde Cero
Laboratorio 05: Backpropagation
"""

import numpy as np
import matplotlib.pyplot as plt


class Module:
    """Clase base para todos los módulos/capas."""
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError


class Linear(Module):
    """Capa totalmente conectada: y = Wx + b"""
    
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size, 1))
        
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: entrada de forma (input_size, batch_size)
        
        Returns:
            out: salida de forma (output_size, batch_size)
        """
        self.x = x
        out = self.W @ x + self.b
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Args:
            dout: gradiente upstream de forma (output_size, batch_size)
        
        Returns:
            dx: gradiente respecto a x de forma (input_size, batch_size)
        """
        batch_size = self.x.shape[1]
        
        # Gradientes de parámetros
        self.dW = dout @ self.x.T / batch_size
        self.db = np.sum(dout, axis=1, keepdims=True) / batch_size
        
        # Gradiente respecto a la entrada
        dx = self.W.T @ dout
        
        return dx


class ReLU(Module):
    """Activación ReLU."""
    
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x > 0)
        out = np.maximum(0, x)
        return out
    
    def backward(self, dout):
        dx = dout * self.mask
        return dx


class Sigmoid(Module):
    """Activación Sigmoid."""
    
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


class Softmax(Module):
    """Softmax + Categorical Cross-Entropy (combinados para estabilidad)."""
    
    def __init__(self):
        self.probs = None
        self.y_true = None
    
    def forward(self, x, y_true=None):
        """
        Forward pass de softmax.
        
        Args:
            x: logits de forma (num_classes, batch_size)
            y_true: etiquetas one-hot de forma (num_classes, batch_size)
        """
        # Softmax estable
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        self.probs = exp_x / np.sum(exp_x, axis=0, keepdims=True)
        
        if y_true is not None:
            self.y_true = y_true
            # Calcular pérdida
            epsilon = 1e-15
            probs_clipped = np.clip(self.probs, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(y_true * np.log(probs_clipped), axis=0))
            return self.probs, loss
        
        return self.probs
    
    def backward(self):
        """
        Backward pass (combinado con cross-entropy).
        Resultado notable: dL/dx = probs - y_true
        """
        batch_size = self.probs.shape[1]
        dx = (self.probs - self.y_true) / batch_size
        return dx


class MSELoss(Module):
    """Mean Squared Error Loss."""
    
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = np.mean((y_pred - y_true) ** 2)
        return loss
    
    def backward(self):
        batch_size = self.y_true.shape[1]
        dy_pred = 2 * (self.y_pred - self.y_true) / batch_size
        return dy_pred


class TwoLayerNet:
    """Red neuronal de dos capas con backpropagation completo."""
    
    def __init__(self, input_size, hidden_size, output_size):
        # Capas
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)
        self.sigmoid = Sigmoid()
        self.loss_fn = MSELoss()
        
        self.layers = [self.fc1, self.relu, self.fc2, self.sigmoid]
    
    def forward(self, x, y_true=None):
        """
        Forward pass completo.
        
        Args:
            x: entrada (input_size, batch_size)
            y_true: etiquetas (output_size, batch_size)
        
        Returns:
            out: predicciones
            loss: pérdida (si y_true es proporcionado)
        """
        # Capa 1
        out = self.fc1.forward(x)
        out = self.relu.forward(out)
        
        # Capa 2
        out = self.fc2.forward(out)
        out = self.sigmoid.forward(out)
        
        if y_true is not None:
            loss = self.loss_fn.forward(out, y_true)
            return out, loss
        
        return out
    
    def backward(self):
        """
        Backward pass completo usando backpropagation.
        """
        # Gradiente de la pérdida
        dout = self.loss_fn.backward()
        
        # Backward a través de las capas (en orden inverso)
        dout = self.sigmoid.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.relu.backward(dout)
        dout = self.fc1.backward(dout)
        
        return dout
    
    def get_params_and_grads(self):
        """Retorna parámetros y gradientes."""
        params = []
        grads = []
        
        params.append(self.fc1.W)
        params.append(self.fc1.b)
        params.append(self.fc2.W)
        params.append(self.fc2.b)
        
        grads.append(self.fc1.dW)
        grads.append(self.fc1.db)
        grads.append(self.fc2.dW)
        grads.append(self.fc2.db)
        
        return params, grads


def verificar_gradientes():
    """Verifica la implementación de backpropagation con gradientes numéricos."""
    
    print("=" * 60)
    print("VERIFICACIÓN DE GRADIENTES NUMÉRICOS")
    print("=" * 60)
    
    # Crear red pequeña
    np.random.seed(42)
    net = TwoLayerNet(2, 3, 1)
    
    # Datos de prueba
    x = np.random.randn(2, 5)
    y = np.random.randn(1, 5)
    
    # Forward y backward
    _, loss = net.forward(x, y)
    net.backward()
    
    # Obtener gradientes analíticos
    params, grads_analytical = net.get_params_and_grads()
    
    # Calcular gradientes numéricos
    epsilon = 1e-5
    
    for param, grad_analytical, name in zip(params, grads_analytical, 
                                             ['W1', 'b1', 'W2', 'b2']):
        grad_numerical = np.zeros_like(param)
        
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_value = param[idx]
            
            # f(x + h)
            param[idx] = old_value + epsilon
            _, loss_plus = net.forward(x, y)
            
            # f(x - h)
            param[idx] = old_value - epsilon
            _, loss_minus = net.forward(x, y)
            
            # Gradiente numérico
            grad_numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restaurar
            param[idx] = old_value
            it.iternext()
        
        # Comparar
        diff = np.linalg.norm(grad_analytical - grad_numerical) / \
               (np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical) + 1e-8)
        
        print(f"\n{name}:")
        print(f"  Diferencia relativa: {diff:.2e}")
        if diff < 1e-6:
            print(f"  ✓ CORRECTO")
        else:
            print(f"  ✗ ERROR - Revisar implementación")
    
    print("\n" + "=" * 60)


def visualizar_grafo_computacional():
    """Visualiza un grafo computacional simple."""
    
    print("\n" + "=" * 60)
    print("EJEMPLO DE GRAFO COMPUTACIONAL")
    print("=" * 60)
    
    # Ejemplo: z = (x + y) * w
    x, y, w = 2.0, 3.0, 4.0
    
    print("\nFunción: z = (x + y) * w")
    print(f"Valores: x={x}, y={y}, w={w}")
    
    # Forward pass
    print("\n--- FORWARD PASS ---")
    q = x + y
    print(f"q = x + y = {x} + {y} = {q}")
    z = q * w
    print(f"z = q * w = {q} * {w} = {z}")
    
    # Backward pass
    print("\n--- BACKWARD PASS ---")
    dz_dz = 1.0
    print(f"∂z/∂z = {dz_dz}")
    
    dz_dq = w
    dz_dw = q
    print(f"∂z/∂q = w = {dz_dq}")
    print(f"∂z/∂w = q = {dz_dw}")
    
    dz_dx = dz_dq * 1  # dq/dx = 1
    dz_dy = dz_dq * 1  # dq/dy = 1
    print(f"∂z/∂x = (∂z/∂q)(∂q/∂x) = {dz_dq} * 1 = {dz_dx}")
    print(f"∂z/∂y = (∂z/∂q)(∂q/∂y) = {dz_dq} * 1 = {dz_dy}")
    
    print("\nGradientes finales:")
    print(f"  ∂z/∂x = {dz_dx}")
    print(f"  ∂z/∂y = {dz_dy}")
    print(f"  ∂z/∂w = {dz_dw}")
    
    print("=" * 60)


def entrenar_ejemplo():
    """Ejemplo de entrenamiento usando backpropagation."""
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO CON BACKPROPAGATION")
    print("=" * 60)
    
    # Generar datos XOR
    np.random.seed(42)
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    print("\nProblema: XOR")
    print("X =", X.T)
    print("y =", y.T)
    
    # Crear red
    net = TwoLayerNet(input_size=2, hidden_size=4, output_size=1)
    
    # Parámetros de entrenamiento
    learning_rate = 0.5
    epochs = 1000
    
    losses = []
    
    # Entrenamiento
    for epoch in range(epochs):
        # Forward
        out, loss = net.forward(X, y)
        losses.append(loss)
        
        # Backward
        net.backward()
        
        # Actualizar parámetros
        params, grads = net.get_params_and_grads()
        for param, grad in zip(params, grads):
            param -= learning_rate * grad
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    # Resultados finales
    print(f"\nÉpoca final {epochs}: Loss = {losses[-1]:.6f}")
    print("\nPredicciones finales:")
    predictions = net.forward(X)
    for i in range(X.shape[1]):
        print(f"  Input: {X[:, i]} → Predicción: {predictions[0, i]:.4f}, Real: {y[0, i]}")
    
    # Visualizar curva de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida (MSE)', fontsize=12)
    plt.title('Curva de Aprendizaje - Problema XOR', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('backprop_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Red entrenada exitosamente usando backpropagation!")
    print("=" * 60)


if __name__ == "__main__":
    print("Laboratorio 05: Backpropagation")
    print("=" * 60)
    
    # 1. Visualizar grafo computacional
    visualizar_grafo_computacional()
    
    # 2. Verificar gradientes
    verificar_gradientes()
    
    # 3. Entrenar con backpropagation
    entrenar_ejemplo()
    
    print("\n✓ Laboratorio completado exitosamente!")
