"""
Implementación de Funciones de Pérdida desde Cero
Laboratorio 04: Funciones de Pérdida y Optimización
"""

import numpy as np
import matplotlib.pyplot as plt


class MSE:
    """Mean Squared Error - Para problemas de regresión"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, y_pred, y_true):
        """
        Calcula MSE.
        
        Args:
            y_pred: predicciones (N, )
            y_true: valores verdaderos (N, )
        
        Returns:
            loss: valor escalar de la pérdida
        """
        self.cache = (y_pred, y_true)
        loss = np.mean((y_pred - y_true) ** 2)
        return loss
    
    def backward(self):
        """
        Calcula el gradiente de MSE.
        
        Returns:
            dy_pred: gradiente respecto a las predicciones
        """
        y_pred, y_true = self.cache
        n = y_true.shape[0]
        dy_pred = 2 * (y_pred - y_true) / n
        return dy_pred


class MAE:
    """Mean Absolute Error - Para problemas de regresión"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, y_pred, y_true):
        """Calcula MAE."""
        self.cache = (y_pred, y_true)
        loss = np.mean(np.abs(y_pred - y_true))
        return loss
    
    def backward(self):
        """Calcula el gradiente de MAE."""
        y_pred, y_true = self.cache
        n = y_true.shape[0]
        dy_pred = np.sign(y_pred - y_true) / n
        return dy_pred


class BinaryCrossEntropy:
    """Binary Cross-Entropy - Para clasificación binaria"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, y_pred, y_true):
        """
        Calcula Binary Cross-Entropy.
        
        Args:
            y_pred: probabilidades predichas (N, ) en rango (0, 1)
            y_true: etiquetas verdaderas (N, ) en {0, 1}
        
        Returns:
            loss: valor escalar de la pérdida
        """
        self.cache = (y_pred, y_true)
        # Clip para evitar log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def backward(self):
        """
        Calcula el gradiente de Binary Cross-Entropy.
        
        Returns:
            dy_pred: gradiente respecto a las predicciones
        """
        y_pred, y_true = self.cache
        n = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        dy_pred = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n
        return dy_pred


class CategoricalCrossEntropy:
    """Categorical Cross-Entropy - Para clasificación multiclase"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, y_pred, y_true):
        """
        Calcula Categorical Cross-Entropy.
        
        Args:
            y_pred: probabilidades predichas (N, C) donde C es número de clases
            y_true: etiquetas one-hot encoded (N, C)
        
        Returns:
            loss: valor escalar de la pérdida
        """
        self.cache = (y_pred, y_true)
        # Clip para evitar log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def backward(self):
        """
        Calcula el gradiente de Categorical Cross-Entropy.
        
        Returns:
            dy_pred: gradiente respecto a las predicciones
        """
        y_pred, y_true = self.cache
        n = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        dy_pred = -y_true / y_pred / n
        return dy_pred


class SoftmaxCrossEntropy:
    """
    Softmax + Categorical Cross-Entropy combinados.
    Esta combinación es más estable numéricamente.
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, logits, y_true):
        """
        Calcula Softmax + Cross-Entropy de manera estable.
        
        Args:
            logits: scores antes de softmax (N, C)
            y_true: etiquetas one-hot encoded (N, C)
        
        Returns:
            loss: valor escalar de la pérdida
        """
        # Softmax estable
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        self.cache = (probs, y_true)
        
        # Cross-entropy
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true * np.log(probs), axis=1))
        
        return loss
    
    def backward(self):
        """
        Calcula el gradiente combinado (¡muy simple!).
        
        Returns:
            dlogits: gradiente respecto a los logits
        """
        probs, y_true = self.cache
        n = y_true.shape[0]
        
        # ¡Esta es la magia de esta combinación!
        dlogits = (probs - y_true) / n
        return dlogits


def visualizar_perdidas_regresion():
    """Compara MSE vs MAE para regresión."""
    
    # Generar datos con outlier
    np.random.seed(42)
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_pred = y_true + np.random.randn(10) * 0.5
    y_pred[5] = 15  # Añadir un outlier
    
    # Calcular pérdidas
    mse = MSE()
    mae = MAE()
    
    mse_loss = mse.forward(y_pred, y_true)
    mae_loss = mae.forward(y_pred, y_true)
    
    # Visualizar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicciones vs Real
    axes[0].scatter(range(len(y_true)), y_true, s=100, alpha=0.6, label='Real', color='blue')
    axes[0].scatter(range(len(y_pred)), y_pred, s=100, alpha=0.6, label='Predicción', color='red')
    axes[0].plot([5], [15], 'r*', markersize=20, label='Outlier')
    axes[0].set_xlabel('Muestra', fontsize=12)
    axes[0].set_ylabel('Valor', fontsize=12)
    axes[0].set_title('Datos con Outlier', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Comparación de pérdidas
    perdidas = ['MSE', 'MAE']
    valores = [mse_loss, mae_loss]
    colores = ['#FF6B6B', '#4ECDC4']
    
    bars = axes[1].bar(perdidas, valores, color=colores, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Valor de Pérdida', fontsize=12)
    axes[1].set_title('Comparación MSE vs MAE', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, val in zip(bars, valores):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('perdidas_regresion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=" * 60)
    print("COMPARACIÓN MSE vs MAE CON OUTLIER")
    print("=" * 60)
    print(f"MSE: {mse_loss:.4f}")
    print(f"MAE: {mae_loss:.4f}")
    print("\n⚠️ Observación:")
    print("MSE es mucho más sensible al outlier debido al término cuadrático.")
    print("MAE es más robusta a outliers.")
    print("=" * 60)


def visualizar_binary_crossentropy():
    """Visualiza cómo Binary Cross-Entropy penaliza predicciones."""
    
    # Rango de probabilidades predichas
    y_pred = np.linspace(0.01, 0.99, 100)
    
    # Calcular pérdidas para clase positiva (y=1) y negativa (y=0)
    bce = BinaryCrossEntropy()
    
    losses_positive = []
    losses_negative = []
    
    for pred in y_pred:
        # Clase real = 1
        loss_pos = bce.forward(np.array([pred]), np.array([1]))
        losses_positive.append(loss_pos)
        
        # Clase real = 0
        loss_neg = bce.forward(np.array([pred]), np.array([0]))
        losses_negative.append(loss_neg)
    
    # Visualizar
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_pred, losses_positive, 'b-', linewidth=2, label='Clase Real = 1')
    plt.plot(y_pred, losses_negative, 'r-', linewidth=2, label='Clase Real = 0')
    plt.xlabel('Probabilidad Predicha', fontsize=12)
    plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
    plt.title('Pérdida vs Predicción', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 5)
    
    # Casos específicos
    plt.subplot(1, 2, 2)
    casos = [
        ('Correcto\n(y=1, ŷ=0.9)', 0.9, 1, 'green'),
        ('Incorrecto\n(y=1, ŷ=0.1)', 0.1, 1, 'red'),
        ('Correcto\n(y=0, ŷ=0.1)', 0.1, 0, 'green'),
        ('Incorrecto\n(y=0, ŷ=0.9)', 0.9, 0, 'red'),
    ]
    
    nombres = []
    perdidas_casos = []
    colores = []
    
    for nombre, pred, true, color in casos:
        loss = bce.forward(np.array([pred]), np.array([true]))
        nombres.append(nombre)
        perdidas_casos.append(loss)
        colores.append(color)
    
    bars = plt.bar(range(len(nombres)), perdidas_casos, color=colores, alpha=0.6, edgecolor='black')
    plt.xticks(range(len(nombres)), nombres, fontsize=9)
    plt.ylabel('Pérdida', fontsize=12)
    plt.title('Ejemplos de Pérdida', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for i, (bar, val) in enumerate(zip(bars, perdidas_casos)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('binary_crossentropy.png', dpi=300, bbox_inches='tight')
    plt.show()


def ejemplo_gradient_descent():
    """Demuestra gradient descent en acción."""
    
    print("\n" + "=" * 60)
    print("GRADIENT DESCENT EN ACCIÓN")
    print("=" * 60)
    
    # Problema simple: minimizar f(x) = (x - 3)²
    # Mínimo en x = 3
    
    def f(x):
        return (x - 3) ** 2
    
    def df(x):
        return 2 * (x - 3)
    
    # Diferentes learning rates
    learning_rates = [0.01, 0.1, 0.5]
    
    plt.figure(figsize=(15, 5))
    
    for idx, lr in enumerate(learning_rates):
        x = np.linspace(-2, 8, 1000)
        y = f(x)
        
        # Gradient descent
        x_current = 0.0
        history = [x_current]
        
        for _ in range(20):
            gradient = df(x_current)
            x_current = x_current - lr * gradient
            history.append(x_current)
        
        # Visualizar
        plt.subplot(1, 3, idx + 1)
        plt.plot(x, y, 'b-', linewidth=2, alpha=0.3, label='f(x)')
        plt.plot(history, [f(xi) for xi in history], 'ro-', markersize=6, linewidth=1, label='GD')
        plt.plot(3, 0, 'g*', markersize=20, label='Mínimo')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.title(f'Learning Rate = {lr}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.ylim(-1, 20)
        
        print(f"\nLearning Rate = {lr}")
        print(f"  Posición inicial: x = 0.0")
        print(f"  Posición final: x = {history[-1]:.4f}")
        print(f"  Iteraciones para converger: {len(history)}")
    
    plt.tight_layout()
    plt.savefig('gradient_descent.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=" * 60)


def ejemplo_overfitting():
    """Demuestra el concepto de overfitting."""
    
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN DE OVERFITTING")
    print("=" * 60)
    
    # Generar datos
    np.random.seed(42)
    X_train = np.linspace(0, 10, 20)
    y_train = 2 * X_train + 1 + np.random.randn(20) * 2
    
    X_test = np.linspace(0, 10, 100)
    y_test = 2 * X_test + 1 + np.random.randn(100) * 2
    
    # Ajustar polinomios de diferentes grados
    degrees = [1, 3, 15]
    
    plt.figure(figsize=(15, 5))
    
    for idx, degree in enumerate(degrees):
        # Ajustar polinomio
        coeffs = np.polyfit(X_train, y_train, degree)
        poly = np.poly1d(coeffs)
        
        # Predecir
        y_train_pred = poly(X_train)
        y_test_pred = poly(X_test)
        
        # Calcular MSE
        mse_train = np.mean((y_train_pred - y_train) ** 2)
        mse_test = np.mean((y_test_pred - y_test) ** 2)
        
        # Visualizar
        plt.subplot(1, 3, idx + 1)
        plt.scatter(X_train, y_train, s=50, alpha=0.6, label='Train', color='blue')
        plt.plot(X_test, y_test_pred, 'r-', linewidth=2, label='Modelo')
        plt.xlabel('X', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title(f'Polinomio Grado {degree}\nTrain MSE={mse_train:.2f}, Test MSE={mse_test:.2f}', 
                 fontsize=12, fontweight='bold')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.ylim(-10, 35)
        
        modelo_tipo = "Bien ajustado" if degree == 1 else ("Overfitting" if degree == 15 else "Intermedio")
        print(f"\nGrado {degree} ({modelo_tipo}):")
        print(f"  MSE Entrenamiento: {mse_train:.4f}")
        print(f"  MSE Test: {mse_test:.4f}")
    
    plt.tight_layout()
    plt.savefig('overfitting_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n⚠️ Conclusión:")
    print("Modelo muy complejo (grado 15) → Overfitting")
    print("Modelo simple (grado 1) → Generaliza mejor")
    print("=" * 60)


if __name__ == "__main__":
    print("Laboratorio 04: Funciones de Pérdida y Optimización")
    print("=" * 60)
    
    # 1. Comparar MSE vs MAE
    print("\n1. Comparando MSE vs MAE con outliers...")
    visualizar_perdidas_regresion()
    
    # 2. Binary Cross-Entropy
    print("\n2. Visualizando Binary Cross-Entropy...")
    visualizar_binary_crossentropy()
    
    # 3. Gradient Descent
    print("\n3. Demostrando Gradient Descent...")
    ejemplo_gradient_descent()
    
    # 4. Overfitting
    print("\n4. Demostrando Overfitting...")
    ejemplo_overfitting()
    
    print("\n✓ Laboratorio completado exitosamente!")
