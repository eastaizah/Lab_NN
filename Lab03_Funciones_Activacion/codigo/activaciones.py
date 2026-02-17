"""
Implementación de Funciones de Activación desde Cero
Laboratorio 03: Funciones de Activación
"""

import numpy as np
import matplotlib.pyplot as plt


class Sigmoid:
    """Función de activación Sigmoid: σ(x) = 1 / (1 + e^(-x))"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """
        Calcula la salida de sigmoid.
        
        Args:
            x: numpy array de cualquier forma
        
        Returns:
            out: salida, misma forma que x
        """
        out = 1 / (1 + np.exp(-x))
        self.cache = out
        return out
    
    def backward(self, dout):
        """
        Calcula el gradiente de sigmoid.
        
        Args:
            dout: gradiente upstream
        
        Returns:
            dx: gradiente respecto a x
        """
        out = self.cache
        dx = dout * out * (1 - out)
        return dx


class Tanh:
    """Función de activación Tanh: tanh(x) = (e^x - e^-x) / (e^x + e^-x)"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Calcula la salida de tanh."""
        out = np.tanh(x)
        self.cache = out
        return out
    
    def backward(self, dout):
        """Calcula el gradiente de tanh."""
        out = self.cache
        dx = dout * (1 - out**2)
        return dx


class ReLU:
    """Función de activación ReLU: ReLU(x) = max(0, x)"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Calcula la salida de ReLU."""
        out = np.maximum(0, x)
        self.cache = x
        return out
    
    def backward(self, dout):
        """Calcula el gradiente de ReLU."""
        x = self.cache
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx


class LeakyReLU:
    """Función de activación Leaky ReLU: LeakyReLU(x) = max(αx, x)"""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.cache = None
    
    def forward(self, x):
        """Calcula la salida de Leaky ReLU."""
        out = np.where(x > 0, x, self.alpha * x)
        self.cache = x
        return out
    
    def backward(self, dout):
        """Calcula el gradiente de Leaky ReLU."""
        x = self.cache
        dx = dout.copy()
        dx[x <= 0] *= self.alpha
        return dx


class Softmax:
    """Función de activación Softmax para clasificación multiclase"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """
        Calcula la salida de softmax de manera numéricamente estable.
        
        Args:
            x: numpy array de forma (N, C) donde N es batch size y C es número de clases
        
        Returns:
            out: probabilidades de forma (N, C)
        """
        # Restar el máximo para estabilidad numérica
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.cache = out
        return out
    
    def backward(self, dout):
        """
        Calcula el gradiente de softmax.
        Nota: Típicamente se combina con la pérdida cross-entropy
        """
        out = self.cache
        dx = out * (dout - np.sum(dout * out, axis=1, keepdims=True))
        return dx


def visualizar_activaciones():
    """Visualiza las funciones de activación y sus derivadas."""
    
    # Crear rango de valores
    x = np.linspace(-5, 5, 1000)
    
    # Instanciar funciones
    sigmoid = Sigmoid()
    tanh = Tanh()
    relu = ReLU()
    leaky_relu = LeakyReLU()
    
    # Calcular forward pass
    y_sigmoid = sigmoid.forward(x)
    y_tanh = tanh.forward(x)
    y_relu = relu.forward(x)
    y_leaky = leaky_relu.forward(x)
    
    # Calcular derivadas (usando dout = 1)
    dout = np.ones_like(x)
    dy_sigmoid = sigmoid.backward(dout)
    dy_tanh = tanh.backward(dout)
    dy_relu = relu.backward(dout)
    dy_leaky = leaky_relu.backward(dout)
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Funciones de Activación y sus Derivadas', fontsize=16, fontweight='bold')
    
    # Sigmoid
    axes[0, 0].plot(x, y_sigmoid, 'b-', linewidth=2)
    axes[0, 0].set_title('Sigmoid')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel('f(x)')
    
    axes[1, 0].plot(x, dy_sigmoid, 'r-', linewidth=2)
    axes[1, 0].set_title('Derivada de Sigmoid')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylabel("f'(x)")
    axes[1, 0].set_xlabel('x')
    
    # Tanh
    axes[0, 1].plot(x, y_tanh, 'b-', linewidth=2)
    axes[0, 1].set_title('Tanh')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(x, dy_tanh, 'r-', linewidth=2)
    axes[1, 1].set_title('Derivada de Tanh')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel('x')
    
    # ReLU
    axes[0, 2].plot(x, y_relu, 'b-', linewidth=2)
    axes[0, 2].set_title('ReLU')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 2].plot(x, dy_relu, 'r-', linewidth=2)
    axes[1, 2].set_title('Derivada de ReLU')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlabel('x')
    
    # Leaky ReLU
    axes[0, 3].plot(x, y_leaky, 'b-', linewidth=2)
    axes[0, 3].set_title('Leaky ReLU')
    axes[0, 3].grid(True, alpha=0.3)
    
    axes[1, 3].plot(x, dy_leaky, 'r-', linewidth=2)
    axes[1, 3].set_title('Derivada de Leaky ReLU')
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].set_xlabel('x')
    
    plt.tight_layout()
    plt.savefig('activaciones_comparacion.png', dpi=300, bbox_inches='tight')
    plt.show()


def comparar_saturacion():
    """Compara cómo se saturan diferentes funciones de activación."""
    
    x = np.linspace(-10, 10, 1000)
    
    sigmoid = Sigmoid()
    tanh = Tanh()
    relu = ReLU()
    
    y_sigmoid = sigmoid.forward(x)
    y_tanh = tanh.forward(x)
    y_relu = relu.forward(x)
    
    dout = np.ones_like(x)
    dy_sigmoid = sigmoid.backward(dout)
    dy_tanh = tanh.backward(dout)
    dy_relu = relu.backward(dout)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Funciones
    ax1.plot(x, y_sigmoid, label='Sigmoid', linewidth=2)
    ax1.plot(x, y_tanh, label='Tanh', linewidth=2)
    ax1.plot(x, y_relu, label='ReLU', linewidth=2)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Funciones de Activación - Rango Amplio', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Derivadas
    ax2.plot(x, dy_sigmoid, label='Sigmoid', linewidth=2)
    ax2.plot(x, dy_tanh, label='Tanh', linewidth=2)
    ax2.plot(x, dy_relu, label='ReLU', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel("f'(x)", fontsize=12)
    ax2.set_title('Gradientes - Problema de Saturación', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('saturacion_gradientes.png', dpi=300, bbox_inches='tight')
    plt.show()


def ejemplo_softmax():
    """Ejemplo de uso de Softmax para clasificación."""
    
    # Scores de 5 muestras para 3 clases
    scores = np.array([
        [2.0, 1.0, 0.1],   # Muestra 1
        [1.0, 3.0, 0.2],   # Muestra 2
        [0.5, 0.6, 2.5],   # Muestra 3
        [3.0, 0.5, 1.0],   # Muestra 4
        [0.1, 0.2, 0.3]    # Muestra 5
    ])
    
    softmax = Softmax()
    probabilidades = softmax.forward(scores)
    
    print("=" * 60)
    print("EJEMPLO DE SOFTMAX PARA CLASIFICACIÓN MULTICLASE")
    print("=" * 60)
    print("\nScores originales (logits):")
    print(scores)
    print("\nProbabilidades después de Softmax:")
    print(np.round(probabilidades, 4))
    print("\nSuma de probabilidades por fila (debe ser ~1.0):")
    print(np.sum(probabilidades, axis=1))
    print("\nClase predicha (argmax) para cada muestra:")
    print(np.argmax(probabilidades, axis=1))
    print("\nConfianza de la predicción (max probability):")
    print(np.round(np.max(probabilidades, axis=1), 4))
    print("=" * 60)


def test_gradientes_numericos():
    """Verifica que las derivadas implementadas sean correctas usando gradientes numéricos."""
    
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE GRADIENTES NUMÉRICOS")
    print("=" * 60)
    
    x = np.random.randn(3, 4)
    dout = np.random.randn(3, 4)
    epsilon = 1e-5
    
    funciones = {
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh(),
        'ReLU': ReLU(),
        'LeakyReLU': LeakyReLU()
    }
    
    for nombre, funcion in funciones.items():
        # Gradiente analítico
        out = funcion.forward(x)
        dx_analitico = funcion.backward(dout)
        
        # Gradiente numérico
        dx_numerico = np.zeros_like(x)
        
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            
            valor_original = x[idx]
            
            x[idx] = valor_original + epsilon
            funcion_cache = funcion.cache
            out_pos = funcion.forward(x)
            loss_pos = np.sum(dout * out_pos)
            
            x[idx] = valor_original - epsilon
            out_neg = funcion.forward(x)
            loss_neg = np.sum(dout * out_neg)
            
            dx_numerico[idx] = (loss_pos - loss_neg) / (2 * epsilon)
            
            x[idx] = valor_original
            funcion.cache = funcion_cache
            it.iternext()
        
        # Calcular diferencia
        diferencia = np.linalg.norm(dx_analitico - dx_numerico) / (np.linalg.norm(dx_analitico) + np.linalg.norm(dx_numerico))
        
        print(f"\n{nombre}:")
        print(f"  Diferencia relativa: {diferencia:.2e}")
        print(f"  {'✓ CORRECTO' if diferencia < 1e-7 else '✗ ERROR'}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("Laboratorio 03: Funciones de Activación")
    print("=" * 60)
    
    # 1. Visualizar funciones de activación
    print("\n1. Generando visualizaciones de funciones de activación...")
    visualizar_activaciones()
    
    # 2. Comparar saturación
    print("\n2. Comparando saturación de gradientes...")
    comparar_saturacion()
    
    # 3. Ejemplo de Softmax
    print("\n3. Ejemplo de Softmax:")
    ejemplo_softmax()
    
    # 4. Verificar gradientes
    print("\n4. Verificando gradientes numéricos...")
    test_gradientes_numericos()
    
    print("\n✓ Laboratorio completado exitosamente!")
