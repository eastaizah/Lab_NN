"""
Implementación de Redes Neuronales Convolucionales (CNN) desde Cero
Laboratorio 09: Redes Neuronales Convolucionales
==================================================

Este módulo implementa:
- Operación de convolución 2D desde cero
- Capas convolucionales con forward y backward pass
- Capas de pooling (max y average)
- CNN completa combinando capas convolucionales, pooling y densas
- Filtros comunes (detección de bordes, blur, etc.)
- Ejemplo de entrenamiento con PyTorch en MNIST
- Funciones de visualización
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Literal

# Importar PyTorch de manera opcional
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Advertencia: PyTorch no está disponible. "
          "Las funciones de entrenamiento estarán deshabilitadas.")


# =============================================================================
# PARTE 1: CONVOLUCIÓN 2D DESDE CERO
# =============================================================================

def convolve2d(image: np.ndarray, kernel: np.ndarray, 
               stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Implementa la operación de convolución 2D desde cero usando NumPy.
    
    Args:
        image: Imagen de entrada de forma (H, W) o (C, H, W)
        kernel: Filtro/kernel de forma (K_H, K_W) o (C, K_H, K_W)
        stride: Paso de desplazamiento del kernel
        padding: Cantidad de padding a añadir alrededor de la imagen
    
    Returns:
        Resultado de la convolución de forma (H_out, W_out)
    
    Dimensiones de salida:
        H_out = (H + 2*padding - K_H) // stride + 1
        W_out = (W + 2*padding - K_W) // stride + 1
    """
    # Manejar imágenes 2D (escala de grises)
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
        kernel = kernel[np.newaxis, :, :]
    
    C, H, W = image.shape
    C_k, K_H, K_W = kernel.shape
    
    assert C == C_k, f"El número de canales debe coincidir: imagen={C}, kernel={C_k}"
    
    # Aplicar padding si es necesario
    if padding > 0:
        image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), 
                      mode='constant', constant_values=0)
        H, W = H + 2*padding, W + 2*padding
    
    # Calcular dimensiones de salida
    H_out = (H - K_H) // stride + 1
    W_out = (W - K_W) // stride + 1
    
    # Inicializar salida
    output = np.zeros((H_out, W_out))
    
    # Realizar convolución
    for i in range(H_out):
        for j in range(W_out):
            # Extraer región de la imagen
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + K_H
            w_end = w_start + K_W
            
            region = image[:, h_start:h_end, w_start:w_end]
            
            # Aplicar kernel y sumar sobre todos los canales
            output[i, j] = np.sum(region * kernel)
    
    return output


# =============================================================================
# PARTE 2: FILTROS COMUNES
# =============================================================================

class FiltrosComunes:
    """Colección de filtros/kernels comunes para procesamiento de imágenes."""
    
    @staticmethod
    def sobel_horizontal():
        """Detector de bordes horizontal (Sobel)."""
        return np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])
    
    @staticmethod
    def sobel_vertical():
        """Detector de bordes vertical (Sobel)."""
        return np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
    
    @staticmethod
    def laplaciano():
        """Detector de bordes Laplaciano (detecta bordes en todas direcciones)."""
        return np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ])
    
    @staticmethod
    def sharpen():
        """Filtro de nitidez (sharpening)."""
        return np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
    
    @staticmethod
    def blur_promedio(size: int = 3):
        """Filtro de desenfoque por promedio (box blur)."""
        return np.ones((size, size)) / (size * size)
    
    @staticmethod
    def blur_gaussiano():
        """Filtro de desenfoque Gaussiano (aproximado)."""
        return np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]) / 16.0
    
    @staticmethod
    def emboss():
        """Filtro de relieve (emboss)."""
        return np.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]
        ])
    
    @staticmethod
    def identidad():
        """Filtro identidad (no modifica la imagen)."""
        return np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])


# =============================================================================
# PARTE 3: CAPA CONVOLUCIONAL CON BACKPROPAGATION
# =============================================================================

class CapaConvolucional:
    """
    Capa convolucional con soporte para forward y backward pass.
    
    Esta implementación incluye:
    - Múltiples filtros/kernels
    - Stride configurable
    - Padding configurable
    - Cálculo de gradientes para backpropagation
    """
    
    def __init__(self, num_filtros: int, tamano_filtro: int, 
                 num_canales_entrada: int, stride: int = 1, 
                 padding: int = 0, learning_rate: float = 0.01):
        """
        Inicializa la capa convolucional.
        
        Args:
            num_filtros: Número de filtros/kernels
            tamano_filtro: Tamaño del filtro (asume cuadrado)
            num_canales_entrada: Número de canales de entrada
            stride: Paso de desplazamiento
            padding: Cantidad de padding
            learning_rate: Tasa de aprendizaje para actualización de pesos
        """
        self.num_filtros = num_filtros
        self.tamano_filtro = tamano_filtro
        self.num_canales_entrada = num_canales_entrada
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate
        
        # Inicializar filtros con distribución normal pequeña (He initialization)
        self.filtros = np.random.randn(
            num_filtros, num_canales_entrada, tamano_filtro, tamano_filtro
        ) * np.sqrt(2.0 / (num_canales_entrada * tamano_filtro * tamano_filtro))
        
        # Inicializar biases a cero
        self.biases = np.zeros(num_filtros)
        
        # Variables para almacenar durante forward pass
        self.entrada = None
        self.salida = None
    
    def forward(self, entrada: np.ndarray) -> np.ndarray:
        """
        Forward pass de la capa convolucional.
        
        Args:
            entrada: Tensor de entrada de forma (batch_size, C, H, W)
        
        Returns:
            Salida de forma (batch_size, num_filtros, H_out, W_out)
        """
        self.entrada = entrada
        batch_size, C, H, W = entrada.shape
        
        # Calcular dimensiones de salida
        H_out = (H + 2*self.padding - self.tamano_filtro) // self.stride + 1
        W_out = (W + 2*self.padding - self.tamano_filtro) // self.stride + 1
        
        # Inicializar salida
        salida = np.zeros((batch_size, self.num_filtros, H_out, W_out))
        
        # Aplicar padding si es necesario
        if self.padding > 0:
            entrada_padded = np.pad(
                entrada, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            entrada_padded = entrada
        
        # Realizar convolución para cada imagen en el batch
        for b in range(batch_size):
            for f in range(self.num_filtros):
                # Convolución del filtro f con la imagen b
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.tamano_filtro
                        w_end = w_start + self.tamano_filtro
                        
                        region = entrada_padded[b, :, h_start:h_end, w_start:w_end]
                        salida[b, f, i, j] = np.sum(region * self.filtros[f]) + self.biases[f]
        
        self.salida = salida
        return salida
    
    def backward(self, d_salida: np.ndarray) -> np.ndarray:
        """
        Backward pass de la capa convolucional.
        
        Args:
            d_salida: Gradiente de la pérdida respecto a la salida
                     de forma (batch_size, num_filtros, H_out, W_out)
        
        Returns:
            Gradiente respecto a la entrada de forma (batch_size, C, H, W)
        """
        batch_size, C, H, W = self.entrada.shape
        _, _, H_out, W_out = d_salida.shape
        
        # Inicializar gradientes
        d_filtros = np.zeros_like(self.filtros)
        d_biases = np.zeros_like(self.biases)
        d_entrada = np.zeros_like(self.entrada)
        
        # Aplicar padding a la entrada si es necesario
        if self.padding > 0:
            entrada_padded = np.pad(
                self.entrada,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
            d_entrada_padded = np.zeros_like(entrada_padded)
        else:
            entrada_padded = self.entrada
            d_entrada_padded = d_entrada
        
        # Calcular gradientes
        for b in range(batch_size):
            for f in range(self.num_filtros):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.tamano_filtro
                        w_end = w_start + self.tamano_filtro
                        
                        # Gradiente respecto a los filtros
                        region = entrada_padded[b, :, h_start:h_end, w_start:w_end]
                        d_filtros[f] += region * d_salida[b, f, i, j]
                        
                        # Gradiente respecto a la entrada
                        d_entrada_padded[b, :, h_start:h_end, w_start:w_end] += \
                            self.filtros[f] * d_salida[b, f, i, j]
                
                # Gradiente respecto a los biases
                d_biases[f] += np.sum(d_salida[:, f, :, :])
        
        # Promediar gradientes sobre el batch
        d_filtros /= batch_size
        d_biases /= batch_size
        
        # Remover padding del gradiente de entrada si es necesario
        if self.padding > 0:
            d_entrada = d_entrada_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_entrada = d_entrada_padded
        
        # Actualizar parámetros
        self.filtros -= self.learning_rate * d_filtros
        self.biases -= self.learning_rate * d_biases
        
        return d_entrada


# =============================================================================
# PARTE 4: CAPA DE POOLING
# =============================================================================

class CapaPooling:
    """
    Capa de pooling con soporte para max pooling y average pooling.
    
    El pooling reduce las dimensiones espaciales de la entrada,
    manteniendo la información más importante y reduciendo el costo computacional.
    """
    
    def __init__(self, tamano_pool: int = 2, stride: int = 2, 
                 tipo: Literal['max', 'avg'] = 'max'):
        """
        Inicializa la capa de pooling.
        
        Args:
            tamano_pool: Tamaño de la ventana de pooling
            stride: Paso de desplazamiento
            tipo: Tipo de pooling ('max' o 'avg')
        """
        self.tamano_pool = tamano_pool
        self.stride = stride
        self.tipo = tipo
        
        # Variables para almacenar durante forward pass
        self.entrada = None
        self.salida = None
        self.max_indices = None  # Para max pooling backprop
    
    def forward(self, entrada: np.ndarray) -> np.ndarray:
        """
        Forward pass de la capa de pooling.
        
        Args:
            entrada: Tensor de entrada de forma (batch_size, C, H, W)
        
        Returns:
            Salida de forma (batch_size, C, H_out, W_out)
        """
        self.entrada = entrada
        batch_size, C, H, W = entrada.shape
        
        # Calcular dimensiones de salida
        H_out = (H - self.tamano_pool) // self.stride + 1
        W_out = (W - self.tamano_pool) // self.stride + 1
        
        # Inicializar salida
        salida = np.zeros((batch_size, C, H_out, W_out))
        
        if self.tipo == 'max':
            # Guardar índices de los máximos para backprop
            self.max_indices = np.zeros((batch_size, C, H_out, W_out, 2), dtype=int)
        
        # Realizar pooling
        for b in range(batch_size):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.tamano_pool
                        w_end = w_start + self.tamano_pool
                        
                        region = entrada[b, c, h_start:h_end, w_start:w_end]
                        
                        if self.tipo == 'max':
                            # Max pooling
                            salida[b, c, i, j] = np.max(region)
                            
                            # Guardar índices del máximo
                            max_idx = np.unravel_index(np.argmax(region), region.shape)
                            self.max_indices[b, c, i, j] = [
                                h_start + max_idx[0], 
                                w_start + max_idx[1]
                            ]
                        else:
                            # Average pooling
                            salida[b, c, i, j] = np.mean(region)
        
        self.salida = salida
        return salida
    
    def backward(self, d_salida: np.ndarray) -> np.ndarray:
        """
        Backward pass de la capa de pooling.
        
        Args:
            d_salida: Gradiente de la pérdida respecto a la salida
                     de forma (batch_size, C, H_out, W_out)
        
        Returns:
            Gradiente respecto a la entrada de forma (batch_size, C, H, W)
        """
        batch_size, C, H, W = self.entrada.shape
        _, _, H_out, W_out = d_salida.shape
        
        # Inicializar gradiente de entrada
        d_entrada = np.zeros_like(self.entrada)
        
        if self.tipo == 'max':
            # Max pooling backward: el gradiente solo fluye a través del máximo
            for b in range(batch_size):
                for c in range(C):
                    for i in range(H_out):
                        for j in range(W_out):
                            h_max, w_max = self.max_indices[b, c, i, j]
                            d_entrada[b, c, h_max, w_max] += d_salida[b, c, i, j]
        else:
            # Average pooling backward: el gradiente se distribuye uniformemente
            for b in range(batch_size):
                for c in range(C):
                    for i in range(H_out):
                        for j in range(W_out):
                            h_start = i * self.stride
                            w_start = j * self.stride
                            h_end = h_start + self.tamano_pool
                            w_end = w_start + self.tamano_pool
                            
                            # Distribuir el gradiente uniformemente
                            valor_promedio = d_salida[b, c, i, j] / (self.tamano_pool ** 2)
                            d_entrada[b, c, h_start:h_end, w_start:w_end] += valor_promedio
        
        return d_entrada


# =============================================================================
# PARTE 5: RED CNN COMPLETA
# =============================================================================

class CNN:
    """
    Red Neuronal Convolucional completa que combina capas convolucionales,
    pooling y densas.
    
    Arquitectura típica: CONV -> POOL -> CONV -> POOL -> FLATTEN -> FC -> FC
    """
    
    def __init__(self):
        """Inicializa la CNN con una arquitectura predefinida."""
        self.capas = []
        self.arquitectura = []
    
    def agregar_conv(self, num_filtros: int, tamano_filtro: int,
                     num_canales_entrada: int, stride: int = 1,
                     padding: int = 0, learning_rate: float = 0.01):
        """Agrega una capa convolucional a la red."""
        capa = CapaConvolucional(
            num_filtros, tamano_filtro, num_canales_entrada,
            stride, padding, learning_rate
        )
        self.capas.append(capa)
        self.arquitectura.append(f"Conv({num_filtros}x{tamano_filtro}x{tamano_filtro})")
        return self
    
    def agregar_pooling(self, tamano_pool: int = 2, stride: int = 2,
                       tipo: Literal['max', 'avg'] = 'max'):
        """Agrega una capa de pooling a la red."""
        capa = CapaPooling(tamano_pool, stride, tipo)
        self.capas.append(capa)
        self.arquitectura.append(f"{tipo.capitalize()}Pool({tamano_pool}x{tamano_pool})")
        return self
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass a través de toda la red.
        
        Args:
            X: Entrada de forma (batch_size, C, H, W)
        
        Returns:
            Salida de la red
        """
        activacion = X
        
        for capa in self.capas:
            activacion = capa.forward(activacion)
        
        return activacion
    
    def backward(self, d_salida: np.ndarray) -> np.ndarray:
        """
        Backward pass a través de toda la red.
        
        Args:
            d_salida: Gradiente de la pérdida respecto a la salida
        
        Returns:
            Gradiente respecto a la entrada
        """
        d_actual = d_salida
        
        # Propagar hacia atrás a través de todas las capas
        for capa in reversed(self.capas):
            d_actual = capa.backward(d_actual)
        
        return d_actual
    
    def resumen(self):
        """Imprime un resumen de la arquitectura de la red."""
        print("=" * 60)
        print("ARQUITECTURA DE LA CNN")
        print("=" * 60)
        for i, nombre in enumerate(self.arquitectura, 1):
            print(f"Capa {i}: {nombre}")
        print("=" * 60)


# =============================================================================
# PARTE 6: CNN CON PYTORCH (EJEMPLO PRÁCTICO CON MNIST)
# =============================================================================

class CNNPyTorch(nn.Module if PYTORCH_AVAILABLE else object):
    """
    Red Neuronal Convolucional implementada con PyTorch.
    
    Arquitectura para MNIST:
    - Conv2d: 1 canal entrada -> 32 canales salida, kernel 3x3
    - ReLU
    - MaxPool2d: 2x2
    - Conv2d: 32 canales -> 64 canales, kernel 3x3
    - ReLU
    - MaxPool2d: 2x2
    - Flatten
    - Linear: -> 128 neuronas
    - ReLU
    - Dropout: 0.5
    - Linear: 128 -> 10 (clases)
    """
    
    def __init__(self):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch no está disponible. Instale PyTorch para usar esta funcionalidad.")
        super(CNNPyTorch, self).__init__()
        
        # Primera capa convolucional
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Imágenes en escala de grises
            out_channels=32,    # 32 filtros
            kernel_size=3,      # Kernel 3x3
            padding=1           # Mantener dimensiones
        )
        
        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Capas densas
        # Después de 2 poolings de 2x2, imágenes de 28x28 -> 7x7
        # 64 canales * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.5)
        
        # Activación
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass."""
        # Primera capa conv + activación + pooling
        x = self.pool(self.relu(self.conv1(x)))
        
        # Segunda capa conv + activación + pooling
        x = self.pool(self.relu(self.conv2(x)))
        
        # Aplanar para capas densas
        x = x.view(-1, 64 * 7 * 7)
        
        # Capas densas
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def entrenar_cnn_pytorch(epochs: int = 5, batch_size: int = 64, 
                         learning_rate: float = 0.001,
                         device: str = None) -> Tuple[CNNPyTorch, List[float], List[float]]:
    """
    Entrena una CNN en el dataset MNIST usando PyTorch.
    
    Args:
        epochs: Número de épocas de entrenamiento
        batch_size: Tamaño del batch
        learning_rate: Tasa de aprendizaje
        device: Dispositivo ('cuda' o 'cpu'). Si es None, detecta automáticamente
    
    Returns:
        Tupla con (modelo entrenado, pérdidas de entrenamiento, accuracy de prueba)
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch no está disponible. Instale PyTorch para usar esta funcionalidad.")
    
    # Detectar dispositivo
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Usando dispositivo: {device}")
    
    # Transformaciones para los datos
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Media y std de MNIST
    ])
    
    # Cargar datos
    print("Cargando dataset MNIST...")
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear modelo
    modelo = CNNPyTorch().to(device)
    
    # Definir función de pérdida y optimizador
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=learning_rate)
    
    # Listas para guardar métricas
    perdidas_entrenamiento = []
    accuracy_prueba = []
    
    print("\nIniciando entrenamiento...")
    print("=" * 70)
    
    # Ciclo de entrenamiento
    for epoch in range(epochs):
        # Modo entrenamiento
        modelo.train()
        perdida_epoch = 0.0
        
        for batch_idx, (datos, etiquetas) in enumerate(train_loader):
            # Mover datos al dispositivo
            datos, etiquetas = datos.to(device), etiquetas.to(device)
            
            # Forward pass
            salidas = modelo(datos)
            perdida = criterio(salidas, etiquetas)
            
            # Backward pass y optimización
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            
            perdida_epoch += perdida.item()
            
            # Mostrar progreso cada 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Batch [{batch_idx}/{len(train_loader)}], '
                      f'Pérdida: {perdida.item():.4f}')
        
        # Calcular pérdida promedio de la época
        perdida_promedio = perdida_epoch / len(train_loader)
        perdidas_entrenamiento.append(perdida_promedio)
        
        # Evaluar en conjunto de prueba
        modelo.eval()
        correctos = 0
        total = 0
        
        with torch.no_grad():
            for datos, etiquetas in test_loader:
                datos, etiquetas = datos.to(device), etiquetas.to(device)
                salidas = modelo(datos)
                _, predicciones = torch.max(salidas.data, 1)
                total += etiquetas.size(0)
                correctos += (predicciones == etiquetas).sum().item()
        
        acc = 100 * correctos / total
        accuracy_prueba.append(acc)
        
        print(f'\nÉpoca [{epoch+1}/{epochs}] completada:')
        print(f'  Pérdida promedio: {perdida_promedio:.4f}')
        print(f'  Accuracy en prueba: {acc:.2f}%')
        print("=" * 70)
    
    print("\n¡Entrenamiento completado!")
    
    return modelo, perdidas_entrenamiento, accuracy_prueba


# =============================================================================
# PARTE 7: FUNCIONES DE VISUALIZACIÓN
# =============================================================================

def visualizar_filtros(filtros: np.ndarray, titulo: str = "Filtros CNN",
                       filas: int = 4, columnas: int = 8):
    """
    Visualiza los filtros de una capa convolucional.
    
    Args:
        filtros: Array de filtros de forma (num_filtros, C, H, W)
        titulo: Título de la figura
        filas: Número de filas en la visualización
        columnas: Número de columnas en la visualización
    """
    num_filtros = min(filtros.shape[0], filas * columnas)
    
    fig, axes = plt.subplots(filas, columnas, figsize=(columnas*1.5, filas*1.5))
    fig.suptitle(titulo, fontsize=16)
    
    for i in range(filas):
        for j in range(columnas):
            idx = i * columnas + j
            ax = axes[i, j] if filas > 1 else axes[j]
            
            if idx < num_filtros:
                # Si tiene múltiples canales, mostrar el promedio
                if filtros.shape[1] > 1:
                    filtro = np.mean(filtros[idx], axis=0)
                else:
                    filtro = filtros[idx, 0]
                
                ax.imshow(filtro, cmap='gray')
                ax.axis('off')
                ax.set_title(f'F{idx}', fontsize=8)
            else:
                ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualizar_mapas_activacion(mapas: np.ndarray, 
                                titulo: str = "Mapas de Activación",
                                max_mapas: int = 16):
    """
    Visualiza los mapas de activación de una capa convolucional.
    
    Args:
        mapas: Mapas de activación de forma (num_mapas, H, W)
        titulo: Título de la figura
        max_mapas: Número máximo de mapas a mostrar
    """
    num_mapas = min(mapas.shape[0], max_mapas)
    filas = int(np.ceil(np.sqrt(num_mapas)))
    columnas = int(np.ceil(num_mapas / filas))
    
    fig, axes = plt.subplots(filas, columnas, figsize=(columnas*2, filas*2))
    fig.suptitle(titulo, fontsize=16)
    
    for idx in range(filas * columnas):
        fila = idx // columnas
        columna = idx % columnas
        
        if filas > 1:
            ax = axes[fila, columna]
        else:
            ax = axes[columna] if columnas > 1 else axes
        
        if idx < num_mapas:
            ax.imshow(mapas[idx], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Mapa {idx}', fontsize=8)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def aplicar_filtro_demo(imagen: np.ndarray, nombre_filtro: str = 'sobel_horizontal'):
    """
    Aplica un filtro común a una imagen y visualiza el resultado.
    
    Args:
        imagen: Imagen de entrada (escala de grises)
        nombre_filtro: Nombre del filtro a aplicar
    """
    # Obtener filtro
    filtros_disponibles = {
        'sobel_horizontal': FiltrosComunes.sobel_horizontal(),
        'sobel_vertical': FiltrosComunes.sobel_vertical(),
        'laplaciano': FiltrosComunes.laplaciano(),
        'sharpen': FiltrosComunes.sharpen(),
        'blur': FiltrosComunes.blur_promedio(),
        'gaussiano': FiltrosComunes.blur_gaussiano(),
        'emboss': FiltrosComunes.emboss(),
        'identidad': FiltrosComunes.identidad()
    }
    
    if nombre_filtro not in filtros_disponibles:
        raise ValueError(f"Filtro '{nombre_filtro}' no disponible. "
                        f"Opciones: {list(filtros_disponibles.keys())}")
    
    filtro = filtros_disponibles[nombre_filtro]
    
    # Aplicar convolución
    resultado = convolve2d(imagen, filtro, padding=1)
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(imagen, cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    axes[1].imshow(filtro, cmap='gray')
    axes[1].set_title(f'Filtro: {nombre_filtro}')
    axes[1].axis('off')
    
    axes[2].imshow(resultado, cmap='gray')
    axes[2].set_title('Resultado de la Convolución')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig, resultado


def visualizar_entrenamiento(perdidas: List[float], accuracy: List[float]):
    """
    Visualiza las curvas de pérdida y accuracy durante el entrenamiento.
    
    Args:
        perdidas: Lista de pérdidas por época
        accuracy: Lista de accuracy por época
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfica de pérdida
    ax1.plot(range(1, len(perdidas)+1), perdidas, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Pérdida', fontsize=12)
    ax1.set_title('Pérdida durante el Entrenamiento', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Gráfica de accuracy
    ax2.plot(range(1, len(accuracy)+1), accuracy, 'g-', linewidth=2, marker='s')
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy en Conjunto de Prueba', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualizar_predicciones(modelo: CNNPyTorch, dataset, num_imagenes: int = 10,
                           device: str = 'cpu'):
    """
    Visualiza predicciones del modelo en imágenes aleatorias.
    
    Args:
        modelo: Modelo CNN entrenado
        dataset: Dataset de prueba
        num_imagenes: Número de imágenes a visualizar
        device: Dispositivo de cómputo
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch no está disponible. Instale PyTorch para usar esta funcionalidad.")
    
    modelo.eval()
    
    # Seleccionar imágenes aleatorias
    indices = np.random.choice(len(dataset), num_imagenes, replace=False)
    
    filas = 2
    columnas = (num_imagenes + 1) // 2
    fig, axes = plt.subplots(filas, columnas, figsize=(columnas*2.5, filas*3))
    
    with torch.no_grad():
        for idx, img_idx in enumerate(indices):
            imagen, etiqueta_real = dataset[img_idx]
            
            # Predecir
            imagen_input = imagen.unsqueeze(0).to(device)
            salida = modelo(imagen_input)
            _, prediccion = torch.max(salida, 1)
            
            # Obtener probabilidades
            probabilidades = torch.nn.functional.softmax(salida, dim=1)
            confianza = probabilidades[0, prediccion].item() * 100
            
            # Visualizar
            fila = idx // columnas
            columna = idx % columnas
            ax = axes[fila, columna] if filas > 1 else axes[columna]
            
            # Mostrar imagen
            ax.imshow(imagen.squeeze(), cmap='gray')
            
            # Título con predicción
            color = 'green' if prediccion.item() == etiqueta_real else 'red'
            ax.set_title(f'Real: {etiqueta_real} | Pred: {prediccion.item()}\n'
                        f'Confianza: {confianza:.1f}%',
                        color=color, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    return fig


# =============================================================================
# EJEMPLOS DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LABORATORIO 09: REDES NEURONALES CONVOLUCIONALES (CNN)")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # EJEMPLO 1: Convolución 2D básica con filtros comunes
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("EJEMPLO 1: Aplicación de Filtros Comunes")
    print("="*70)
    
    # Crear una imagen de prueba simple
    imagen_prueba = np.random.rand(28, 28)
    
    # Aplicar diferentes filtros
    filtros_demo = ['sobel_horizontal', 'sobel_vertical', 'blur', 'sharpen']
    
    print("\nAplicando filtros a imagen de prueba...")
    for filtro in filtros_demo:
        print(f"  - Aplicando filtro: {filtro}")
        try:
            fig, resultado = aplicar_filtro_demo(imagen_prueba, filtro)
            plt.savefig(f'filtro_{filtro}_demo.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    ✓ Guardado como: filtro_{filtro}_demo.png")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # -------------------------------------------------------------------------
    # EJEMPLO 2: CNN desde cero con NumPy
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("EJEMPLO 2: CNN desde Cero (NumPy)")
    print("="*70)
    
    # Crear una CNN simple
    cnn_numpy = CNN()
    cnn_numpy.agregar_conv(
        num_filtros=8, 
        tamano_filtro=3, 
        num_canales_entrada=1, 
        padding=1
    )
    cnn_numpy.agregar_pooling(tamano_pool=2, tipo='max')
    cnn_numpy.agregar_conv(
        num_filtros=16, 
        tamano_filtro=3, 
        num_canales_entrada=8, 
        padding=1
    )
    cnn_numpy.agregar_pooling(tamano_pool=2, tipo='max')
    
    print("\nArquitectura de la CNN:")
    cnn_numpy.resumen()
    
    # Probar forward pass
    print("\nProbando forward pass...")
    entrada_prueba = np.random.randn(2, 1, 28, 28)  # 2 imágenes 28x28
    print(f"  Forma de entrada: {entrada_prueba.shape}")
    
    salida = cnn_numpy.forward(entrada_prueba)
    print(f"  Forma de salida: {salida.shape}")
    print("  ✓ Forward pass exitoso")
    
    # Visualizar filtros de la primera capa
    print("\nVisualizando filtros de la primera capa...")
    primera_capa = cnn_numpy.capas[0]
    if isinstance(primera_capa, CapaConvolucional):
        fig = visualizar_filtros(
            primera_capa.filtros, 
            titulo="Filtros de la Primera Capa Convolucional",
            filas=2, columnas=4
        )
        plt.savefig('filtros_primera_capa.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        print("  ✓ Guardado como: filtros_primera_capa.png")
    
    # -------------------------------------------------------------------------
    # EJEMPLO 3: Entrenamiento con PyTorch en MNIST
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("EJEMPLO 3: Entrenamiento con PyTorch (MNIST)")
    print("="*70)
    
    if not PYTORCH_AVAILABLE:
        print("\n⚠ PyTorch no está disponible. Saltando este ejemplo.")
        print("  Instale PyTorch y torchvision para usar esta funcionalidad:")
        print("  pip install torch torchvision")
    else:
        import sys
        # Soportar modo no-interactivo para pruebas automáticas
        if '--entrenar' in sys.argv:
            respuesta = 's'
        elif '--no-entrenar' in sys.argv:
            respuesta = 'n'
        else:
            respuesta = input("\n¿Desea entrenar la CNN en MNIST? (s/n): ").lower()
    
        if respuesta == 's':
            try:
                # Entrenar modelo
                modelo, perdidas, accuracy = entrenar_cnn_pytorch(
                    epochs=3,
                    batch_size=64,
                    learning_rate=0.001
                )
                
                # Visualizar curvas de entrenamiento
                print("\nGenerando gráficas de entrenamiento...")
                fig = visualizar_entrenamiento(perdidas, accuracy)
                plt.savefig('entrenamiento_cnn_mnist.png', dpi=100, bbox_inches='tight')
                plt.close(fig)
                print("  ✓ Guardado como: entrenamiento_cnn_mnist.png")
                
                # Visualizar predicciones
                print("\nGenerando visualización de predicciones...")
                test_dataset = datasets.MNIST(
                    root='./data', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
                )
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                fig = visualizar_predicciones(modelo, test_dataset, num_imagenes=10, device=device)
                plt.savefig('predicciones_cnn_mnist.png', dpi=100, bbox_inches='tight')
                plt.close(fig)
                print("  ✓ Guardado como: predicciones_cnn_mnist.png")
                
                # Guardar modelo
                print("\nGuardando modelo entrenado...")
                torch.save(modelo.state_dict(), 'cnn_mnist_modelo.pth')
                print("  ✓ Guardado como: cnn_mnist_modelo.pth")
                
                print("\n¡Entrenamiento y evaluación completados!")
                print(f"Accuracy final: {accuracy[-1]:.2f}%")
                
            except Exception as e:
                print(f"\n✗ Error durante el entrenamiento: {e}")
                print("  Continuando con el resto de ejemplos...")
        else:
            print("  Saltando entrenamiento.")
    
    # -------------------------------------------------------------------------
    # EJEMPLO 4: Demostración de Pooling
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("EJEMPLO 4: Demostración de Max Pooling vs Average Pooling")
    print("="*70)
    
    # Crear una imagen de prueba
    imagen_pool = np.random.rand(1, 1, 8, 8)
    
    print(f"\nImagen de entrada: {imagen_pool.shape}")
    
    # Max pooling
    max_pool = CapaPooling(tamano_pool=2, stride=2, tipo='max')
    salida_max = max_pool.forward(imagen_pool)
    print(f"Salida Max Pooling: {salida_max.shape}")
    
    # Average pooling
    avg_pool = CapaPooling(tamano_pool=2, stride=2, tipo='avg')
    salida_avg = avg_pool.forward(imagen_pool)
    print(f"Salida Average Pooling: {salida_avg.shape}")
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(imagen_pool[0, 0], cmap='viridis')
    axes[0].set_title('Imagen Original (8x8)')
    axes[0].axis('off')
    
    axes[1].imshow(salida_max[0, 0], cmap='viridis')
    axes[1].set_title('Max Pooling (4x4)')
    axes[1].axis('off')
    
    axes[2].imshow(salida_avg[0, 0], cmap='viridis')
    axes[2].set_title('Average Pooling (4x4)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('pooling_comparacion.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Guardado como: pooling_comparacion.png")
    
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TODOS LOS EJEMPLOS COMPLETADOS")
    print("="*70)
    print("\nArchivos generados:")
    print("  - filtro_*_demo.png: Aplicación de diferentes filtros")
    print("  - filtros_primera_capa.png: Visualización de filtros CNN")
    print("  - pooling_comparacion.png: Comparación de tipos de pooling")
    if PYTORCH_AVAILABLE:
        print("\nSi entrenó el modelo:")
        print("  - entrenamiento_cnn_mnist.png: Curvas de entrenamiento")
        print("  - predicciones_cnn_mnist.png: Predicciones del modelo")
        print("  - cnn_mnist_modelo.pth: Modelo entrenado")
    print("\n" + "="*70)
