"""
Lab 02: Primera Red Neuronal
=============================

Este script implementa una red neuronal multicapa desde cero.
"""

import numpy as np
import matplotlib.pyplot as plt


class CapaDensa:
    """Capa densa (fully connected) de una red neuronal."""
    
    def __init__(self, n_entradas, n_neuronas):
        """
        Inicializa una capa densa.
        
        Args:
            n_entradas: número de entradas/características
            n_neuronas: número de neuronas en esta capa
        """
        # Inicialización de pesos con valores pequeños aleatorios
        self.pesos = np.random.randn(n_entradas, n_neuronas) * 0.01
        self.biases = np.zeros(n_neuronas)
        
        # Guardaremos entradas y salidas para uso posterior
        self.entradas = None
        self.salida = None
    
    def forward(self, entradas):
        """
        Forward pass (propagación hacia adelante).
        
        Args:
            entradas: datos de entrada (batch_size, n_entradas)
        
        Returns:
            salida de la capa (batch_size, n_neuronas)
        """
        self.entradas = entradas
        self.salida = np.dot(entradas, self.pesos) + self.biases
        return self.salida
    
    def __repr__(self):
        return f"CapaDensa(entradas={self.pesos.shape[0]}, neuronas={self.pesos.shape[1]})"


class RedNeuronal:
    """Red neuronal multicapa."""
    
    def __init__(self, arquitectura):
        """
        Inicializa la red neuronal.
        
        Args:
            arquitectura: lista con el número de neuronas por capa
                         Ejemplo: [784, 128, 64, 10]
                         - 784: tamaño de entrada
                         - 128, 64: capas ocultas
                         - 10: capa de salida
        """
        self.arquitectura = arquitectura
        self.capas = []
        
        # Crear capas
        for i in range(len(arquitectura) - 1):
            capa = CapaDensa(arquitectura[i], arquitectura[i + 1])
            self.capas.append(capa)
        
        self.num_capas = len(self.capas)
    
    def forward(self, X):
        """
        Forward propagation a través de toda la red.
        
        Args:
            X: datos de entrada (batch_size, n_features)
        
        Returns:
            salida de la red (batch_size, n_output)
        """
        activacion = X
        
        for capa in self.capas:
            activacion = capa.forward(activacion)
        
        return activacion
    
    def contar_parametros(self):
        """Cuenta el número total de parámetros en la red."""
        total = 0
        detalles = []
        
        for i, capa in enumerate(self.capas):
            params_pesos = capa.pesos.size
            params_biases = capa.biases.size
            params_capa = params_pesos + params_biases
            total += params_capa
            
            detalles.append({
                'capa': i + 1,
                'pesos': params_pesos,
                'biases': params_biases,
                'total': params_capa
            })
        
        return total, detalles
    
    def resumen(self):
        """Imprime un resumen de la arquitectura de la red."""
        print("=" * 70)
        print("RESUMEN DE LA RED NEURONAL")
        print("=" * 70)
        print(f"Arquitectura: {self.arquitectura}")
        print(f"Número de capas: {self.num_capas}")
        print()
        
        total_params, detalles = self.contar_parametros()
        
        print(f"{'Capa':<10} {'Shape Pesos':<20} {'Parámetros':<15}")
        print("-" * 70)
        
        for i, capa in enumerate(self.capas):
            info = detalles[i]
            shape = f"({capa.pesos.shape[0]}, {capa.pesos.shape[1]})"
            print(f"Capa {i+1:<5} {shape:<20} {info['total']:<15,}")
        
        print("-" * 70)
        print(f"{'TOTAL':<31} {total_params:,}")
        print("=" * 70)
    
    def __repr__(self):
        return f"RedNeuronal(arquitectura={self.arquitectura})"


def visualizar_activaciones(red, X, titulo="Activaciones de la Red"):
    """
    Visualiza las activaciones de cada capa para una muestra.
    
    Args:
        red: instancia de RedNeuronal
        X: muestra de entrada (debe ser 1D o una sola muestra)
        titulo: título para el gráfico
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Forward pass y guardar activaciones
    activaciones = [X[0]]  # Comenzamos con la entrada
    activacion = X
    
    for capa in red.capas:
        activacion = capa.forward(activacion)
        activaciones.append(activacion[0])
    
    # Visualización
    num_capas = len(activaciones)
    fig, axes = plt.subplots(1, num_capas, figsize=(3 * num_capas, 3))
    
    if num_capas == 1:
        axes = [axes]
    
    # Graficamos cada capa
    nombres = ['Entrada'] + [f'Capa {i+1}' for i in range(len(red.capas))]
    
    for i, (ax, act, nombre) in enumerate(zip(axes, activaciones, nombres)):
        x = np.arange(len(act))
        ax.bar(x, act, alpha=0.7)
        ax.set_title(nombre)
        ax.set_xlabel('Neurona' if i > 0 else 'Característica')
        ax.set_ylabel('Activación')
        ax.grid(True, alpha=0.3)
        
        # Mostrar dimensión
        ax.text(0.5, 0.95, f'dim: {len(act)}', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(titulo, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def ejemplo_red_simple():
    """Ejemplo básico de una red neuronal."""
    print("\n" + "=" * 70)
    print("EJEMPLO 1: Red Neuronal Simple")
    print("=" * 70 + "\n")
    
    # Crear una red pequeña
    red = RedNeuronal([4, 8, 3])
    red.resumen()
    
    # Datos de prueba
    X = np.array([
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ])
    
    print(f"\nEntrada shape: {X.shape}")
    salida = red.forward(X)
    print(f"Salida shape: {salida.shape}")
    print(f"\nSalida:\n{salida}")


def ejemplo_red_mnist():
    """Ejemplo de red para MNIST."""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Red para Clasificación MNIST")
    print("=" * 70 + "\n")
    
    # Arquitectura típica para MNIST
    red_mnist = RedNeuronal([784, 128, 64, 10])
    red_mnist.resumen()
    
    # Simular imágenes
    batch_size = 32
    imagenes = np.random.randn(batch_size, 784) * 0.1
    
    print(f"\nProcesando {batch_size} imágenes...")
    predicciones = red_mnist.forward(imagenes)
    
    print(f"Predicciones shape: {predicciones.shape}")
    print(f"\nPrimera predicción (scores por clase):")
    print(predicciones[0])


def ejemplo_comparacion_arquitecturas():
    """Compara diferentes arquitecturas."""
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Comparación de Arquitecturas")
    print("=" * 70 + "\n")
    
    arquitecturas = [
        [100, 50, 10],           # Red pequeña
        [100, 200, 100, 10],     # Red mediana
        [100, 64, 64, 64, 10],   # Red profunda
    ]
    
    for arq in arquitecturas:
        red = RedNeuronal(arq)
        total_params, _ = red.contar_parametros()
        print(f"Arquitectura: {arq}")
        print(f"Parámetros totales: {total_params:,}")
        print()


def ejemplo_visualizacion():
    """Visualiza activaciones de una red."""
    print("\n" + "=" * 70)
    print("EJEMPLO 4: Visualización de Activaciones")
    print("=" * 70 + "\n")
    
    np.random.seed(42)
    
    # Red pequeña para visualizar
    red = RedNeuronal([10, 8, 6, 4])
    
    # Muestra de entrada
    X = np.random.randn(10)
    
    print("Generando visualización...")
    visualizar_activaciones(red, X, "Flujo de Activaciones")


def ejemplo_red_profunda_vs_ancha():
    """Compara red profunda vs ancha con similar número de parámetros."""
    print("\n" + "=" * 70)
    print("EJEMPLO 5: Red Profunda vs Red Ancha")
    print("=" * 70 + "\n")
    
    # Red profunda
    red_profunda = RedNeuronal([100, 30, 25, 20, 15, 10])
    params_profunda, _ = red_profunda.contar_parametros()
    
    # Red ancha
    red_ancha = RedNeuronal([100, 120, 10])
    params_ancha, _ = red_ancha.contar_parametros()
    
    print("RED PROFUNDA:")
    print(f"Arquitectura: {red_profunda.arquitectura}")
    print(f"Parámetros: {params_profunda:,}\n")
    
    print("RED ANCHA:")
    print(f"Arquitectura: {red_ancha.arquitectura}")
    print(f"Parámetros: {params_ancha:,}\n")
    
    print(f"Diferencia: {abs(params_profunda - params_ancha):,} parámetros")
    
    # Probar con los mismos datos
    X = np.random.randn(10, 100)
    
    salida_profunda = red_profunda.forward(X)
    salida_ancha = red_ancha.forward(X)
    
    print(f"\nAmbas producen salidas shape: {salida_profunda.shape}")


def main():
    """Ejecuta todos los ejemplos."""
    print("\n" + "=" * 70)
    print("LAB 02: PRIMERA RED NEURONAL")
    print("=" * 70)
    
    ejemplo_red_simple()
    ejemplo_red_mnist()
    ejemplo_comparacion_arquitecturas()
    ejemplo_visualizacion()
    ejemplo_red_profunda_vs_ancha()
    
    print("\n" + "=" * 70)
    print("¡Todos los ejemplos completados!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
