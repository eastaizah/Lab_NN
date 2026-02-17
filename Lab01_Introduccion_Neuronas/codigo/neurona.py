"""
Lab 01: Implementación de una Neurona Simple
=============================================

Este script demuestra cómo implementar una neurona artificial desde cero.
"""

import numpy as np


def neurona_simple(inputs, weights, bias):
    """
    Calcula la salida de una neurona simple.
    
    Args:
        inputs: lista o array de valores de entrada
        weights: lista o array de pesos
        bias: valor de bias
    
    Returns:
        output: salida de la neurona
    """
    output = 0.0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]
    output += bias
    return output


def neurona_numpy(inputs, weights, bias):
    """
    Calcula la salida de una neurona usando NumPy.
    
    Args:
        inputs: array de valores de entrada
        weights: array de pesos
        bias: valor de bias
    
    Returns:
        output: salida de la neurona
    """
    return np.dot(inputs, weights) + bias


class Neurona:
    """Clase que representa una neurona artificial."""
    
    def __init__(self, n_inputs):
        """
        Inicializa la neurona con pesos y bias aleatorios.
        
        Args:
            n_inputs: número de entradas que recibe la neurona
        """
        # Inicializamos pesos con valores pequeños aleatorios
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
    
    def forward(self, inputs):
        """
        Calcula la salida de la neurona (forward pass).
        
        Args:
            inputs: array de valores de entrada
        
        Returns:
            output: salida de la neurona
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output
    
    def __repr__(self):
        return f"Neurona(n_inputs={len(self.weights)})"


class CapaNeuronal:
    """Clase que representa una capa de neuronas."""
    
    def __init__(self, n_inputs, n_neurons):
        """
        Inicializa la capa con pesos y biases aleatorios.
        
        Args:
            n_inputs: número de entradas que recibe cada neurona
            n_neurons: número de neuronas en la capa
        """
        # Pesos: matriz de n_neurons x n_inputs
        # Cada fila representa los pesos de una neurona
        self.weights = np.random.randn(n_neurons, n_inputs) * 0.01
        
        # Biases: un valor por neurona
        self.biases = np.zeros(n_neurons)
        
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
    
    def forward(self, inputs):
        """
        Calcula la salida de la capa (forward pass).
        
        Args:
            inputs: array de valores de entrada (puede ser un batch)
                   Shape: (n_samples, n_inputs) o (n_inputs,)
        
        Returns:
            outputs: salidas de todas las neuronas
                    Shape: (n_samples, n_neurons) o (n_neurons,)
        """
        self.inputs = inputs
        # inputs @ weights.T + biases
        # Si inputs es 1D: (n_inputs,) @ (n_neurons, n_inputs).T = (n_neurons,)
        # Si inputs es 2D: (n_samples, n_inputs) @ (n_neurons, n_inputs).T = (n_samples, n_neurons)
        self.output = np.dot(inputs, self.weights.T) + self.biases
        return self.output
    
    def __repr__(self):
        return f"CapaNeuronal(n_inputs={self.n_inputs}, n_neurons={self.n_neurons})"


def ejemplo_neurona_simple():
    """Ejemplo básico de una neurona."""
    print("=" * 50)
    print("Ejemplo 1: Neurona Simple")
    print("=" * 50)
    
    inputs = [1.0, 2.0, 3.0]
    weights = [0.2, 0.8, -0.5]
    bias = 2.0
    
    output = neurona_simple(inputs, weights, bias)
    print(f"Entradas: {inputs}")
    print(f"Pesos: {weights}")
    print(f"Bias: {bias}")
    print(f"Salida: {output}")
    print()


def ejemplo_neurona_numpy():
    """Ejemplo de neurona usando NumPy."""
    print("=" * 50)
    print("Ejemplo 2: Neurona con NumPy")
    print("=" * 50)
    
    inputs = np.array([1.0, 2.0, 3.0, 2.5])
    weights = np.array([0.2, 0.8, -0.5, 1.0])
    bias = 2.0
    
    output = neurona_numpy(inputs, weights, bias)
    print(f"Entradas: {inputs}")
    print(f"Pesos: {weights}")
    print(f"Bias: {bias}")
    print(f"Salida: {output}")
    print()


def ejemplo_clase_neurona():
    """Ejemplo usando la clase Neurona."""
    print("=" * 50)
    print("Ejemplo 3: Clase Neurona")
    print("=" * 50)
    
    # Fijamos la semilla para reproducibilidad
    np.random.seed(0)
    
    neurona = Neurona(n_inputs=4)
    print(f"Neurona creada: {neurona}")
    print(f"Pesos iniciales: {neurona.weights}")
    print(f"Bias inicial: {neurona.bias}")
    
    inputs = np.array([1.0, 2.0, 3.0, 2.5])
    output = neurona.forward(inputs)
    print(f"\nEntradas: {inputs}")
    print(f"Salida: {output}")
    print()


def ejemplo_capa_neuronal():
    """Ejemplo usando la clase CapaNeuronal."""
    print("=" * 50)
    print("Ejemplo 4: Capa de Neuronas")
    print("=" * 50)
    
    # Fijamos la semilla para reproducibilidad
    np.random.seed(0)
    
    # Creamos una capa con 3 neuronas, cada una con 4 entradas
    capa = CapaNeuronal(n_inputs=4, n_neurons=3)
    print(f"Capa creada: {capa}")
    print(f"Shape de pesos: {capa.weights.shape}")
    print(f"Shape de biases: {capa.biases.shape}")
    
    # Procesamos una sola muestra
    inputs = np.array([1.0, 2.0, 3.0, 2.5])
    outputs = capa.forward(inputs)
    print(f"\nProcesando una muestra:")
    print(f"Entradas shape: {inputs.shape}")
    print(f"Salidas shape: {outputs.shape}")
    print(f"Salidas: {outputs}")
    print()


def ejemplo_batch_processing():
    """Ejemplo de procesamiento por lotes."""
    print("=" * 50)
    print("Ejemplo 5: Procesamiento en Batch")
    print("=" * 50)
    
    # Fijamos la semilla para reproducibilidad
    np.random.seed(0)
    
    # Creamos una capa
    capa = CapaNeuronal(n_inputs=4, n_neurons=3)
    
    # Creamos un batch de 5 muestras
    inputs = np.array([
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
        [0.5, 1.0, 1.5, 2.0],
        [3.0, -1.0, 0.5, 1.5]
    ])
    
    outputs = capa.forward(inputs)
    
    print(f"Procesando batch de {len(inputs)} muestras:")
    print(f"Entradas shape: {inputs.shape}")
    print(f"Salidas shape: {outputs.shape}")
    print(f"\nSalidas:\n{outputs}")
    print()


def main():
    """Función principal que ejecuta todos los ejemplos."""
    print("\n" + "=" * 50)
    print("LAB 01: INTRODUCCIÓN A LAS NEURONAS")
    print("=" * 50 + "\n")
    
    ejemplo_neurona_simple()
    ejemplo_neurona_numpy()
    ejemplo_clase_neurona()
    ejemplo_capa_neuronal()
    ejemplo_batch_processing()
    
    print("=" * 50)
    print("¡Todos los ejemplos completados!")
    print("=" * 50)


if __name__ == "__main__":
    main()
