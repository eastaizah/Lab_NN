# Guía de Laboratorio: Redes Neuronales Convolucionales (CNN)

## 📋 Información del Laboratorio

**Título:** Redes Neuronales Convolucionales — Visión Computacional con Deep Learning  
**Código:** Lab 10  
**Duración:** 3-4 horas  
**Nivel:** Avanzado  

---

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Explicar por qué las redes densas son ineficientes para imágenes y cómo las CNNs resuelven ese problema
2. Implementar la operación de convolución 2D desde cero usando NumPy
3. Aplicar filtros clásicos de visión computacional (bordes, blur, Sobel)
4. Calcular dimensiones de salida usando la fórmula: `(Input - Kernel + 2·Padding) / Stride + 1`
5. Implementar capas de Max Pooling y Average Pooling desde cero
6. Construir una arquitectura CNN completa con PyTorch
7. Entrenar una CNN en el dataset MNIST y alcanzar >98% de precisión
8. Visualizar filtros aprendidos y feature maps de activación
9. Comparar cuantitativamente el rendimiento de CNN vs red densa
10. Entender el concepto de campo receptivo y jerarquía de características
11. Describir las innovaciones de LeNet, AlexNet, VGG y ResNet
12. Implementar skip connections al estilo ResNet
13. Aplicar Transfer Learning con modelos pre-entrenados
14. Utilizar técnicas de regularización específicas para CNNs (Dropout, Batch Normalization, Data Augmentation)

---

## 📚 Prerrequisitos

### Conocimientos

- Python intermedio-avanzado y NumPy (Labs 01–09)
- Redes neuronales densas y backpropagation (Labs 02, 05)
- Funciones de activación, pérdida y optimizadores (Labs 03, 04, 06)
- PyTorch básico: tensores, autograd, `nn.Module` (Lab 08)
- Álgebra lineal: multiplicación de matrices, suma elemento a elemento

### Software

- Python 3.8+
- PyTorch 1.10+ (`pip install torch torchvision`)
- NumPy, Matplotlib, Scipy
- Jupyter Notebook (opcional pero recomendado)

```bash
pip install torch torchvision numpy matplotlib scipy pillow
```

### Material de Lectura

Antes de comenzar, lee:
- `teoria.md` — Fundamentos teóricos completos de CNNs
- `README.md` — Estructura y recursos del laboratorio
- [CS231n Lecture Notes — CNNs](http://cs231n.github.io/convolutional-networks/)
- LeCun et al. (1998): *Gradient-Based Learning Applied to Document Recognition*

---

## 📖 Introducción

### El Problema con las Redes Densas para Imágenes

En los laboratorios anteriores construiste redes neuronales completamente conectadas (dense networks). Funcionan bien para datos tabulares, pero presentan serios problemas cuando se aplican a imágenes:

**Problema 1 — Explosión de parámetros:**
```
Imagen 224×224 RGB = 224 × 224 × 3 = 150,528 píxeles
Primera capa densa con 1,000 neuronas:
  150,528 × 1,000 = 150,528,000 parámetros ← ¡sólo en la primera capa!
```

**Problema 2 — Ignorar estructura espacial:**  
Una red densa trata el píxel en la esquina superior izquierda y el del centro como totalmente independientes. Sin embargo, los píxeles cercanos están correlacionados: forman bordes, texturas y formas.

**Problema 3 — No hay invariancia:**  
Si el mismo objeto aparece desplazado 5 píxeles a la derecha, la red densa lo trata como una entrada completamente diferente.

### La Solución: Redes Neuronales Convolucionales

Las CNNs resuelven los tres problemas anteriores mediante tres principios:

| Principio | Descripción | Beneficio |
|---|---|---|
| **Conectividad local** | Cada neurona se conecta sólo a una pequeña región de la entrada | Menos parámetros |
| **Compartición de pesos** | El mismo filtro se aplica a toda la imagen | Invariancia a traslación |
| **Jerarquía de características** | Capas tempranas: bordes → medias: texturas → tardías: objetos | Representaciones ricas |

### Motivación Histórica

- **1998 — LeNet-5** (Yann LeCun): Primera CNN exitosa, reconocía dígitos manuscritos con >99% de precisión.
- **2012 — AlexNet** (Krizhevsky, Hinton): Ganó ImageNet con 15.3% de error, 10 puntos por debajo del segundo lugar. Marcó el inicio del auge del Deep Learning.
- **2014 — VGGNet**: Simplificó el diseño usando sólo kernels 3×3, llegando a 16-19 capas.
- **2015 — ResNet**: Introdujo skip connections y permitió entrenar redes de +100 capas.
- **Hoy**: CNNs en todos lados — diagnóstico médico, vehículos autónomos, reconocimiento facial, arte generativo.

### Aplicaciones Prácticas

- 🏥 Diagnóstico médico: detección de tumores en radiografías y tomografías
- 🚗 Vehículos autónomos: detección de señales, peatones y carriles
- 📱 Filtros de cámara: efectos de arte y realidad aumentada
- 🔍 Búsqueda visual: encontrar imágenes similares
- 🏭 Control de calidad industrial: detección de defectos
- 🌍 Análisis de imágenes satelitales: mapas, deforestación, etc.

---

## 🤔 Preguntas de Reflexión Iniciales

Antes de comenzar a programar, reflexiona sobre las siguientes preguntas:

1. **¿Por qué dos píxeles vecinos en una imagen suelen tener valores similares?** ¿Cómo aprovecha la convolución esta propiedad?

2. **Si tienes una imagen de 28×28 y aplicas un filtro de 5×5 con stride=1 y sin padding, ¿cuál es el tamaño de la salida?** ¿Cuántos píxeles "pierdes" en cada borde?

3. **¿Por qué el Max Pooling 2×2 reduce el tamaño espacial a la mitad?** ¿Qué información se pierde y qué se conserva?

4. **Una CNN entrenada para reconocer gatos aprende filtros de bordes en las primeras capas.** ¿Por qué crees que eso es así? ¿Qué aprenderán las últimas capas?

5. **Skip connections en ResNet suman la entrada con la salida de un bloque.** ¿Por qué esto ayuda al flujo de gradientes durante el entrenamiento?

6. **¿Cuál es la diferencia entre Transfer Learning y entrenar desde cero?** ¿Cuándo usarías cada estrategia?

7. **La misma operación de convolución que detecta bordes en imágenes se usa en audio y texto.** ¿Qué características detectaría en esos dominios?

---

## 🔬 Parte 1: Operaciones de Convolución (45 min)

### 1.1 Convolución 2D desde Cero

La convolución es el corazón de las CNNs. Antes de usar frameworks, es fundamental entender la operación matemáticamente.

**Definición matemática:**
```
Output[i, j] = Σ_m Σ_n Input[i+m, j+n] × Kernel[m, n]
```

```python
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# IMPLEMENTACIÓN DE CONVOLUCIÓN 2D DESDE CERO
# ============================================================

def convolve2d_manual(input_matrix, kernel, stride=1, padding=0):
    """
    Implementación de convolución 2D sin librerías de deep learning.
    
    Args:
        input_matrix: np.array de forma (H, W)
        kernel:       np.array de forma (Kh, Kw)
        stride:       paso del desplazamiento
        padding:      relleno de ceros en los bordes
    
    Returns:
        output: np.array de forma (Oh, Ow)
    """
    H, W = input_matrix.shape
    Kh, Kw = kernel.shape
    
    # Aplicar padding si es necesario
    if padding > 0:
        input_padded = np.pad(input_matrix, padding, mode='constant', constant_values=0)
    else:
        input_padded = input_matrix
    
    # Calcular tamaño de salida
    Oh = (H - Kh + 2 * padding) // stride + 1
    Ow = (W - Kw + 2 * padding) // stride + 1
    
    output = np.zeros((Oh, Ow))
    
    for i in range(Oh):
        for j in range(Ow):
            # Extraer ventana de la entrada
            region = input_padded[i*stride : i*stride + Kh,
                                  j*stride : j*stride + Kw]
            # Producto elemento a elemento y suma
            output[i, j] = np.sum(region * kernel)
    
    return output


# --- Ejemplo con una imagen simple ---
imagen = np.array([
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5]
], dtype=float)

kernel_bordes = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1]
], dtype=float)

resultado = convolve2d_manual(imagen, kernel_bordes)
print("Imagen original (5×5):")
print(imagen)
print("\nKernel de bordes verticales (3×3):")
print(kernel_bordes)
print("\nResultado de la convolución (3×3):")
print(resultado)

# --- Verificar fórmula de dimensiones ---
H, W = 5, 5
Kh, Kw = 3, 3
stride, padding = 1, 0

Oh = (H - Kh + 2 * padding) // stride + 1
Ow = (W - Kw + 2 * padding) // stride + 1
print(f"\nFórmula: ({H} - {Kh} + 2×{padding}) / {stride} + 1 = {Oh}")
print(f"Salida esperada: {Oh}×{Ow}")
print(f"Salida obtenida: {resultado.shape[0]}×{resultado.shape[1]}")
```

**Salida esperada:**
```
Resultado de la convolución (3×3):
[[-6. -6. -6.]
 [-6. -6. -6.]
 [-6. -6. -6.]]
Fórmula: (5 - 3 + 2×0) / 1 + 1 = 3
Salida esperada: 3×3
```

> 💡 **¿Por qué los valores son negativos?** El kernel detecta transiciones de oscuro a claro (izquierda → derecha). Un valor negativo indica borde de claro a oscuro.

---

### 1.2 Filtros Clásicos

La visión computacional clásica define filtros a mano. Las CNNs los **aprenden automáticamente**, pero entender los filtros clásicos da intuición sobre lo que la red descubre.

```python
# ============================================================
# FILTROS CLÁSICOS DE VISIÓN COMPUTACIONAL
# ============================================================

from scipy.ndimage import convolve
from PIL import Image

# Crear imagen sintética de prueba (gradiente + bordes)
def crear_imagen_prueba(size=64):
    img = np.zeros((size, size), dtype=float)
    # Cuadrado blanco en el centro
    img[16:48, 16:48] = 255.0
    # Gradiente horizontal
    for col in range(size):
        img[:, col] += col * (255 / size) * 0.3
    return np.clip(img, 0, 255)

imagen_prueba = crear_imagen_prueba(64)

# --- Definición de filtros clásicos ---
filtros = {
    "Bordes Verticales": np.array([
        [-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1]
    ], dtype=float),
    
    "Bordes Horizontales": np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=float),
    
    "Sobel X": np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=float),
    
    "Sobel Y": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=float),
    
    "Desenfoque (Blur)": np.ones((3, 3), dtype=float) / 9,
    
    "Laplaciano": np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ], dtype=float),
    
    "Realce de Nitidez": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=float),
}

# --- Aplicar y visualizar cada filtro ---
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes[0, 0].imshow(imagen_prueba, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title("Imagen Original", fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

for idx, (nombre, filtro) in enumerate(filtros.items()):
    row, col = (idx + 1) // 4, (idx + 1) % 4
    resultado = convolve(imagen_prueba, filtro)
    resultado_vis = np.abs(resultado)
    axes[row, col].imshow(resultado_vis, cmap='gray')
    axes[row, col].set_title(nombre, fontsize=10)
    axes[row, col].axis('off')
    
    # Mostrar el filtro como texto pequeño
    filtro_str = '\n'.join([' '.join([f'{v:+.1f}' for v in row]) for row in filtro])
    axes[row, col].text(0.02, 0.02, filtro_str, transform=axes[row, col].transAxes,
                        fontsize=6, color='white', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.suptitle("Filtros Clásicos de Visión Computacional", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('filtros_clasicos.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ Figura guardada: filtros_clasicos.png")

# --- Gradiente de Sobel (magnitud) ---
sobel_x = convolve(imagen_prueba, filtros["Sobel X"])
sobel_y = convolve(imagen_prueba, filtros["Sobel Y"])
magnitud = np.sqrt(sobel_x**2 + sobel_y**2)
direccion = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

print(f"\nMagnitud Sobel — min: {magnitud.min():.1f}, max: {magnitud.max():.1f}")
print(f"Dirección Sobel — min: {direccion.min():.1f}°, max: {direccion.max():.1f}°")
```

---

### 1.3 Padding y Stride

Estos dos hiperparámetros controlan el tamaño de salida y qué información se preserva.

```python
# ============================================================
# PADDING Y STRIDE — ANÁLISIS DETALLADO
# ============================================================

def calcular_dimensiones_salida(H, W, Kh, Kw, stride, padding):
    """Calcula dimensiones de salida y las imprime con detalle."""
    Oh = (H - Kh + 2 * padding) // stride + 1
    Ow = (W - Kw + 2 * padding) // stride + 1
    return Oh, Ow

print("=" * 60)
print("TABLA DE DIMENSIONES DE SALIDA")
print("=" * 60)
print(f"{'Config':<35} {'Entrada':<12} {'Salida'}")
print("-" * 60)

configs = [
    # (H, W, Kh, Kw, stride, padding, descripción)
    (32, 32, 3, 3, 1, 0, "Valid (sin padding)"),
    (32, 32, 3, 3, 1, 1, "Same (padding=1)"),
    (32, 32, 3, 3, 2, 0, "Stride=2 (sin padding)"),
    (32, 32, 3, 3, 2, 1, "Stride=2 (padding=1)"),
    (28, 28, 5, 5, 1, 0, "Kernel 5×5 Valid"),
    (28, 28, 5, 5, 1, 2, "Kernel 5×5 Same"),
    (64, 64, 7, 7, 2, 3, "AlexNet Conv1 style"),
    (14, 14, 3, 3, 1, 1, "VGG style"),
]

for H, W, Kh, Kw, s, p, desc in configs:
    Oh, Ow = calcular_dimensiones_salida(H, W, Kh, Kw, s, p)
    print(f"{desc:<35} {H}×{W}{'':5} → {Oh}×{Ow}")

print()
print("TIPOS DE PADDING:")
print("  - 'valid' (p=0): sin relleno, salida más pequeña")
print("  - 'same'  (p=(K-1)/2): mantiene tamaño con stride=1")
print()

# --- Demostración visual de stride ---
print("EFECTO DEL STRIDE EN UNA IMAGEN 6×6 CON KERNEL 2×2:")
print()
imagen_6x6 = np.arange(1, 37).reshape(6, 6)
print("Imagen:")
print(imagen_6x6)

for stride_val in [1, 2, 3]:
    resultado = convolve2d_manual(imagen_6x6.astype(float),
                                  np.ones((2, 2)) / 4,
                                  stride=stride_val, padding=0)
    Oh = (6 - 2 + 0) // stride_val + 1
    print(f"\nStride={stride_val} → salida {Oh}×{Oh}:")
    print(resultado.astype(int))

# --- Padding 'same' exacto ---
def padding_para_same(kernel_size):
    """Calcula el padding necesario para mantener tamaño (stride=1)."""
    return (kernel_size - 1) // 2

print("\nPADDING NECESARIO PARA 'SAME' (stride=1):")
for k in [1, 3, 5, 7, 9]:
    p = padding_para_same(k)
    print(f"  Kernel {k}×{k} → padding = {p}")
```

---

## 🔬 Parte 2: Capas CNN (45 min)

### 2.1 Capa Convolucional

Una capa convolucional aprende sus filtros durante el entrenamiento. Aquí implementamos la estructura completa con múltiples filtros y canales.

```python
# ============================================================
# CAPA CONVOLUCIONAL COMPLETA
# ============================================================

class CapaConvolucional:
    """
    Capa convolucional con N filtros, soporte para múltiples canales.
    Implementación educativa con NumPy.
    """
    
    def __init__(self, num_filtros, kernel_size, num_canales=1,
                 stride=1, padding=0, seed=42):
        """
        Args:
            num_filtros:  Número de filtros (feature maps de salida)
            kernel_size:  Tamaño del kernel (kernel_size × kernel_size)
            num_canales:  Canales de entrada (1=gris, 3=RGB)
            stride:       Paso del filtro
            padding:      Relleno de ceros
            seed:         Semilla para reproducibilidad
        """
        self.num_filtros = num_filtros
        self.kernel_size = kernel_size
        self.num_canales = num_canales
        self.stride = stride
        self.padding = padding
        
        # Inicialización He (recomendada para ReLU)
        rng = np.random.default_rng(seed)
        escala = np.sqrt(2.0 / (kernel_size * kernel_size * num_canales))
        self.pesos = rng.normal(0, escala, 
                                (num_filtros, num_canales, kernel_size, kernel_size))
        self.bias = np.zeros(num_filtros)
        
    def convolve_canal(self, entrada, kernel, stride, padding):
        """Convolución de una entrada 2D con un kernel 2D."""
        H, W = entrada.shape
        K = kernel.shape[0]
        
        if padding > 0:
            entrada = np.pad(entrada, padding, mode='constant')
        
        Oh = (H - K + 2 * padding) // stride + 1
        Ow = (W - K + 2 * padding) // stride + 1
        salida = np.zeros((Oh, Ow))
        
        for i in range(Oh):
            for j in range(Ow):
                region = entrada[i*stride:i*stride+K, j*stride:j*stride+K]
                salida[i, j] = np.sum(region * kernel)
        return salida
    
    def forward(self, entrada):
        """
        Forward pass de la capa convolucional.
        
        Args:
            entrada: np.array de forma (C, H, W) o (H, W) para 1 canal
        Returns:
            salida: np.array de forma (num_filtros, Oh, Ow)
        """
        if entrada.ndim == 2:
            entrada = entrada[np.newaxis, :]   # (1, H, W)
        
        C, H, W = entrada.shape
        K = self.kernel_size
        p = self.padding
        s = self.stride
        
        Oh = (H - K + 2 * p) // s + 1
        Ow = (W - K + 2 * p) // s + 1
        
        salida = np.zeros((self.num_filtros, Oh, Ow))
        
        for f in range(self.num_filtros):
            mapa = np.zeros((Oh, Ow))
            for c in range(C):
                mapa += self.convolve_canal(entrada[c], self.pesos[f, c],
                                            self.stride, self.padding)
            salida[f] = mapa + self.bias[f]
        
        return salida
    
    def info(self):
        """Muestra información de la capa."""
        total_params = self.pesos.size + self.bias.size
        print(f"CapaConvolucional:")
        print(f"  Filtros:     {self.num_filtros}")
        print(f"  Kernel:      {self.kernel_size}×{self.kernel_size}")
        print(f"  Canales in:  {self.num_canales}")
        print(f"  Stride:      {self.stride}")
        print(f"  Padding:     {self.padding}")
        print(f"  Parámetros:  {total_params:,}")


# --- Demostración ---
np.random.seed(42)

# Imagen de entrada: 3 canales (RGB), 32×32
imagen_rgb = np.random.randn(3, 32, 32)

# Capa conv: 16 filtros de 3×3
capa_conv = CapaConvolucional(num_filtros=16, kernel_size=3,
                              num_canales=3, stride=1, padding=1)
capa_conv.info()

salida_conv = capa_conv.forward(imagen_rgb)
print(f"\nEntrada:  {imagen_rgb.shape}  (C, H, W)")
print(f"Salida:   {salida_conv.shape}  (Filtros, Oh, Ow)")

# Verificar dimensiones
Oh_esperado = (32 - 3 + 2*1) // 1 + 1
print(f"Oh esperado: {Oh_esperado}")

# Estadísticas de la salida
print(f"\nEstadísticas de la salida:")
print(f"  Media:  {salida_conv.mean():.4f}")
print(f"  Std:    {salida_conv.std():.4f}")
print(f"  Min:    {salida_conv.min():.4f}")
print(f"  Max:    {salida_conv.max():.4f}")
```

---

### 2.2 Pooling (Max y Average)

El pooling reduce el tamaño espacial manteniendo las características más importantes.

```python
# ============================================================
# CAPAS DE POOLING — MAX Y AVERAGE
# ============================================================

class CapaPooling:
    """
    Capa de pooling con soporte para Max y Average pooling.
    """
    
    def __init__(self, pool_size=2, stride=None, modo='max'):
        """
        Args:
            pool_size: Tamaño de la ventana de pooling
            stride:    Paso (por defecto = pool_size)
            modo:      'max' o 'average'
        """
        self.pool_size = pool_size
        self.stride = stride if stride else pool_size
        self.modo = modo
    
    def _pool_2d(self, entrada):
        """Aplica pooling a una entrada 2D."""
        H, W = entrada.shape
        P = self.pool_size
        S = self.stride
        
        Oh = (H - P) // S + 1
        Ow = (W - P) // S + 1
        salida = np.zeros((Oh, Ow))
        
        for i in range(Oh):
            for j in range(Ow):
                ventana = entrada[i*S:i*S+P, j*S:j*S+P]
                if self.modo == 'max':
                    salida[i, j] = np.max(ventana)
                else:
                    salida[i, j] = np.mean(ventana)
        return salida
    
    def forward(self, entrada):
        """
        Forward pass.
        Args:
            entrada: np.array (C, H, W)
        Returns:
            salida:  np.array (C, Oh, Ow)
        """
        if entrada.ndim == 2:
            return self._pool_2d(entrada)
        
        C = entrada.shape[0]
        salida = [self._pool_2d(entrada[c]) for c in range(C)]
        return np.array(salida)
    
    def info(self):
        print(f"CapaPooling({self.modo}):")
        print(f"  Pool size: {self.pool_size}×{self.pool_size}")
        print(f"  Stride:    {self.stride}")
        print(f"  Parámetros: 0 (sin pesos aprendibles)")


# --- Demostración de Max vs Average Pooling ---
mapa_features = np.array([
    [ 1,  3,  2,  4],
    [ 5,  6,  7,  8],
    [ 9,  2,  1,  3],
    [ 4,  5, 10, 11]
], dtype=float)

max_pool = CapaPooling(pool_size=2, modo='max')
avg_pool = CapaPooling(pool_size=2, modo='average')

resultado_max = max_pool.forward(mapa_features)
resultado_avg = avg_pool.forward(mapa_features)

print("Mapa de features (4×4):")
print(mapa_features)
print(f"\nMax Pooling 2×2 (resultado {resultado_max.shape}):")
print(resultado_max)
print(f"\nAverage Pooling 2×2 (resultado {resultado_avg.shape}):")
print(resultado_avg)

# --- Aplicar pooling a la salida de la capa conv anterior ---
max_pool_capa = CapaPooling(pool_size=2, modo='max')
max_pool_capa.info()

salida_pool = max_pool_capa.forward(salida_conv)
print(f"\nDespués de Max Pooling 2×2:")
print(f"  Antes del pooling: {salida_conv.shape}")
print(f"  Después del pooling: {salida_pool.shape}")
print(f"  Reducción: {salida_conv.shape[1]}×{salida_conv.shape[2]} → "
      f"{salida_pool.shape[1]}×{salida_pool.shape[2]}")

# --- Comparación de propiedades ---
print("\n" + "=" * 50)
print("COMPARACIÓN MAX vs AVERAGE POOLING")
print("=" * 50)
print(f"{'Propiedad':<30} {'Max':<15} {'Average'}")
print("-" * 50)
propiedades = [
    ("Parámetros aprendibles", "0", "0"),
    ("Preserva características fuertes", "✅ Sí", "❌ Suaviza"),
    ("Invariancia a traslación", "✅ Alta", "✅ Moderada"),
    ("Uso típico", "Redes gen.", "GAP final"),
    ("Diferenciable en máximo", "❌ No siempre", "✅ Sí"),
]
for prop, max_val, avg_val in propiedades:
    print(f"{prop:<30} {max_val:<15} {avg_val}")
```

---

### 2.3 Flatten y Capas Densas

Después de las capas convolucionales, se transforma el tensor 3D en un vector 1D para las capas totalmente conectadas.

```python
# ============================================================
# FLATTEN, RELU Y CAPAS DENSAS
# ============================================================

def relu(x):
    """Función de activación ReLU."""
    return np.maximum(0, x)

def softmax(x):
    """Función softmax estable numéricamente."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Flatten:
    """Convierte tensor 3D en vector 1D."""
    
    def __init__(self):
        self.forma_entrada = None
    
    def forward(self, entrada):
        self.forma_entrada = entrada.shape
        return entrada.flatten()
    
    def info(self, forma_entrada=None):
        if forma_entrada:
            C, H, W = forma_entrada
            n = C * H * W
            print(f"Flatten: {C}×{H}×{W} → {n} (vector)")
        else:
            print("Flatten: 3D → 1D")


class CapaDensa:
    """Capa totalmente conectada con activación opcional."""
    
    def __init__(self, n_entrada, n_salida, activacion=None, seed=42):
        rng = np.random.default_rng(seed)
        escala = np.sqrt(2.0 / n_entrada)  # Inicialización He
        self.W = rng.normal(0, escala, (n_salida, n_entrada))
        self.b = np.zeros(n_salida)
        self.activacion = activacion
    
    def forward(self, x):
        z = self.W @ x + self.b
        if self.activacion == 'relu':
            return relu(z)
        elif self.activacion == 'softmax':
            return softmax(z)
        return z
    
    def info(self):
        params = self.W.size + self.b.size
        print(f"CapaDensa: {self.W.shape[1]} → {self.W.shape[0]} "
              f"| act={self.activacion} | params={params:,}")


# --- Pipeline completo: Conv → Pool → Flatten → Dense ---
print("PIPELINE CNN COMPLETO")
print("=" * 50)

# Datos de entrada (imagen gris 28×28)
entrada = np.random.randn(1, 28, 28)
print(f"Entrada:     {entrada.shape}")

# Capa Conv 1: 8 filtros 3×3, padding=1
conv1 = CapaConvolucional(num_filtros=8, kernel_size=3,
                          num_canales=1, padding=1)
x = relu(conv1.forward(entrada))
print(f"Conv1+ReLU:  {x.shape}  (8 filtros × 28×28)")

# Max Pooling 2×2
pool1 = CapaPooling(pool_size=2, modo='max')
x = pool1.forward(x)
print(f"MaxPool:     {x.shape}  (8 × 14×14)")

# Capa Conv 2: 16 filtros 3×3, padding=1
conv2 = CapaConvolucional(num_filtros=16, kernel_size=3,
                          num_canales=8, padding=1)
x = relu(conv2.forward(x))
print(f"Conv2+ReLU:  {x.shape}  (16 filtros × 14×14)")

# Max Pooling 2×2
pool2 = CapaPooling(pool_size=2, modo='max')
x = pool2.forward(x)
print(f"MaxPool:     {x.shape}  (16 × 7×7)")

# Flatten
flatten = Flatten()
x_flat = flatten.forward(x)
print(f"Flatten:     {x_flat.shape}  ({16*7*7} elementos)")

# Capas densas
dense1 = CapaDensa(16*7*7, 128, activacion='relu')
x = dense1.forward(x_flat)
print(f"Dense+ReLU:  {x.shape}")

dense2 = CapaDensa(128, 10, activacion='softmax')
x = dense2.forward(x)
print(f"Dense+Softmax: {x.shape}")

print(f"\nPredicciones (probabilidades):")
print(x.round(4))
print(f"Clase predicha: {np.argmax(x)}")
print(f"Suma de probs:  {x.sum():.4f}")
```

---

## 🔬 Parte 3: Arquitectura CNN Completa (60 min)

### 3.1 Arquitectura CNN con PyTorch

Ahora construimos una CNN real usando PyTorch, aprovechando GPU, autograd y optimizadores modernos.

```python
# ============================================================
# CNN COMPLETA CON PYTORCH
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

# ---- Definición de la arquitectura ----
class CNN_MNIST(nn.Module):
    """
    CNN para clasificación de dígitos MNIST.
    
    Arquitectura:
      INPUT (1×28×28)
      CONV1 (32 filtros 3×3, padding=1) → BN → ReLU → (32×28×28)
      MaxPool 2×2 → (32×14×14)
      CONV2 (64 filtros 3×3, padding=1) → BN → ReLU → (64×14×14)
      MaxPool 2×2 → (64×7×7)
      CONV3 (128 filtros 3×3, padding=1) → BN → ReLU → (128×7×7)
      Flatten → (128×7×7 = 6272)
      FC1 (256) → ReLU → Dropout(0.5)
      FC2 (10) → Output
    """
    
    def __init__(self, num_clases=10, dropout=0.5):
        super(CNN_MNIST, self).__init__()
        
        # Bloque convolucional 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque convolucional 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque convolucional 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Capas clasificadoras
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_clases)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


# ---- Instanciar y revisar el modelo ----
modelo = CNN_MNIST(num_clases=10).to(device)
print("\nArquitectura del modelo:")
print(modelo)

# Contar parámetros
total_params = sum(p.numel() for p in modelo.parameters())
trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(f"\nParámetros totales:       {total_params:,}")
print(f"Parámetros entrenables:   {trainable_params:,}")

# Probar con una imagen de ejemplo
imagen_test = torch.randn(1, 1, 28, 28).to(device)
with torch.no_grad():
    salida_test = modelo(imagen_test)
print(f"\nForma de salida (1 imagen): {salida_test.shape}")
print(f"Logits: {salida_test.detach().cpu().numpy().round(3)}")
```

---

### 3.2 Entrenamiento en MNIST

```python
# ============================================================
# ENTRENAMIENTO EN MNIST
# ============================================================

def entrenar_una_epoca(modelo, loader, optimizer, criterion, device):
    modelo.train()
    total_loss = 0
    total_correcto = 0
    total_muestras = 0
    
    for batch_idx, (datos, etiquetas) in enumerate(loader):
        datos, etiquetas = datos.to(device), etiquetas.to(device)
        
        optimizer.zero_grad()
        salida = modelo(datos)
        loss = criterion(salida, etiquetas)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(datos)
        predicciones = salida.argmax(dim=1)
        total_correcto += (predicciones == etiquetas).sum().item()
        total_muestras += len(datos)
    
    return total_loss / total_muestras, total_correcto / total_muestras


def evaluar(modelo, loader, criterion, device):
    modelo.eval()
    total_loss = 0
    total_correcto = 0
    total_muestras = 0
    
    with torch.no_grad():
        for datos, etiquetas in loader:
            datos, etiquetas = datos.to(device), etiquetas.to(device)
            salida = modelo(datos)
            loss = criterion(salida, etiquetas)
            
            total_loss += loss.item() * len(datos)
            predicciones = salida.argmax(dim=1)
            total_correcto += (predicciones == etiquetas).sum().item()
            total_muestras += len(datos)
    
    return total_loss / total_muestras, total_correcto / total_muestras


# --- Preparar datos ---
transform_train = transforms.Compose([
    transforms.RandomRotation(10),          # Augmentación: rotación aleatoria ±10°
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Traslación aleatoria
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Media y std de MNIST
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=2)

print(f"Datos de entrenamiento: {len(train_dataset):,} imágenes")
print(f"Datos de prueba:        {len(test_dataset):,} imágenes")
print(f"Batches por época:      {len(train_loader)}")

# --- Configurar entrenamiento ---
modelo = CNN_MNIST(num_clases=10, dropout=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# --- Ciclo de entrenamiento ---
NUM_EPOCHS = 10
historial = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n" + "="*65)
print(f"{'Época':<8} {'LR':<10} {'Train Loss':<12} {'Train Acc':<12} {'Val Acc'}")
print("="*65)

mejor_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = entrenar_una_epoca(modelo, train_loader, optimizer, criterion, device)
    val_loss, val_acc     = evaluar(modelo, test_loader, criterion, device)
    scheduler.step()
    
    historial['train_loss'].append(train_loss)
    historial['train_acc'].append(train_acc)
    historial['val_loss'].append(val_loss)
    historial['val_acc'].append(val_acc)
    
    if val_acc > mejor_val_acc:
        mejor_val_acc = val_acc
        torch.save(modelo.state_dict(), 'mejor_cnn_mnist.pth')
        marca = "⭐"
    else:
        marca = ""
    
    lr_actual = scheduler.get_last_lr()[0]
    print(f"{epoch:<8} {lr_actual:<10.6f} {train_loss:<12.4f} {train_acc:<12.4f} {val_acc:.4f} {marca}")

print("="*65)
print(f"\n✅ Mejor precisión de validación: {mejor_val_acc:.4f} ({mejor_val_acc*100:.2f}%)")

# --- Graficar curvas de entrenamiento ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pérdida
axes[0].plot(historial['train_loss'], 'b-o', label='Entrenamiento', markersize=4)
axes[0].plot(historial['val_loss'], 'r-o', label='Validación', markersize=4)
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Pérdida (Cross-Entropy)')
axes[0].set_title('Curva de Pérdida')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Precisión
axes[1].plot([a*100 for a in historial['train_acc']], 'b-o', label='Entrenamiento', markersize=4)
axes[1].plot([a*100 for a in historial['val_acc']], 'r-o', label='Validación', markersize=4)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Precisión (%)')
axes[1].set_title('Curva de Precisión')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([90, 100])

plt.tight_layout()
plt.savefig('curvas_entrenamiento_cnn.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ Figura guardada: curvas_entrenamiento_cnn.png")
```

---

### 3.3 Visualización de Filtros y Feature Maps

Una de las características más poderosas de las CNNs es que sus filtros son interpretables. Visualizarlos revela qué aprende la red.

```python
# ============================================================
# VISUALIZACIÓN DE FILTROS Y FEATURE MAPS
# ============================================================

# Cargar el mejor modelo
modelo.load_state_dict(torch.load('mejor_cnn_mnist.pth', map_location=device))
modelo.eval()

# ---- 1. Visualizar filtros de la primera capa convolucional ----
filtros_conv1 = modelo.conv_block1[0].weight.data.cpu().numpy()
print(f"Filtros Conv1: {filtros_conv1.shape}")  # (32, 1, 3, 3)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for idx, ax in enumerate(axes.flat):
    if idx < 32:
        filtro = filtros_conv1[idx, 0]
        im = ax.imshow(filtro, cmap='RdBu_r',
                       vmin=-filtro.abs().max(),
                       vmax=filtro.abs().max())
        ax.set_title(f'F{idx+1}', fontsize=8)
    ax.axis('off')

plt.suptitle('Filtros Aprendidos — Capa Conv1 (32 filtros 3×3)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('filtros_conv1.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ Figura guardada: filtros_conv1.png")

# ---- 2. Visualizar Feature Maps de activación ----
# Registrar activaciones con hooks
activaciones = {}

def obtener_activacion(nombre):
    def hook(model, input, output):
        activaciones[nombre] = output.detach()
    return hook

# Registrar hooks en cada bloque conv
hook1 = modelo.conv_block1[2].register_forward_hook(obtener_activacion('relu1'))
hook2 = modelo.conv_block2[2].register_forward_hook(obtener_activacion('relu2'))
hook3 = modelo.conv_block3[2].register_forward_hook(obtener_activacion('relu3'))

# Pasar una imagen de prueba
imagen_muestra = test_dataset[0][0].unsqueeze(0).to(device)
etiqueta_muestra = test_dataset[0][1]

with torch.no_grad():
    pred = modelo(imagen_muestra)

clase_pred = pred.argmax().item()
confianza = F.softmax(pred, dim=1).max().item()

# Eliminar hooks
hook1.remove(); hook2.remove(); hook3.remove()

# Visualizar feature maps de cada bloque
for nombre_bloque, num_mostrar in [('relu1', 32), ('relu2', 32), ('relu3', 16)]:
    maps = activaciones[nombre_bloque][0].cpu().numpy()
    n_cols = 8
    n_rows = num_mostrar // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2))
    for idx, ax in enumerate(axes.flat):
        if idx < num_mostrar:
            ax.imshow(maps[idx], cmap='viridis')
            ax.set_title(f'Ch{idx+1}', fontsize=7)
        ax.axis('off')
    
    titulo = (f"Feature Maps — {nombre_bloque.upper()}\n"
              f"Imagen: dígito={etiqueta_muestra} | "
              f"Pred={clase_pred} | Confianza={confianza:.2%}")
    plt.suptitle(titulo, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'feature_maps_{nombre_bloque}.png', dpi=90, bbox_inches='tight')
    plt.show()
    print(f"✅ Figura guardada: feature_maps_{nombre_bloque}.png")

# ---- 3. Imagen original + predicciones por clase ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Imagen original
img_display = test_dataset[0][0].squeeze().numpy()
axes[0].imshow(img_display, cmap='gray')
axes[0].set_title(f'Imagen original (dígito: {etiqueta_muestra})', fontsize=12)
axes[0].axis('off')

# Probabilidades por clase
probabilidades = F.softmax(pred, dim=1)[0].cpu().numpy()
colores = ['green' if i == clase_pred else 'steelblue' for i in range(10)]
axes[1].bar(range(10), probabilidades * 100, color=colores)
axes[1].set_xlabel('Clase (dígito)')
axes[1].set_ylabel('Probabilidad (%)')
axes[1].set_title(f'Predicción: {clase_pred} ({confianza:.2%} confianza)', fontsize=12)
axes[1].set_xticks(range(10))
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('prediccion_cnn.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

## 🔬 Parte 4: Arquitecturas Avanzadas (30 min)

### 4.1 Skip Connections al Estilo ResNet

ResNet resolvió el problema de la degradación en redes muy profundas al aprender el **residuo** en lugar de la transformación completa.

```python
# ============================================================
# SKIP CONNECTIONS — BLOQUES RESIDUALES (ResNet)
# ============================================================

class BloqueResidual(nn.Module):
    """
    Bloque residual básico de ResNet.
    
    Arquitectura:
      x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
      └──────────────────────────────────┘
              (skip connection)
    """
    
    def __init__(self, canales, stride=1):
        super(BloqueResidual, self).__init__()
        
        self.bloque = nn.Sequential(
            nn.Conv2d(canales, canales, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(canales),
            nn.ReLU(inplace=True),
            nn.Conv2d(canales, canales, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(canales)
        )
        
        # Proyección si el stride cambia las dimensiones
        self.proyeccion = None
        if stride != 1:
            self.proyeccion = nn.Sequential(
                nn.Conv2d(canales, canales, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(canales)
            )
    
    def forward(self, x):
        identidad = x
        salida = self.bloque(x)
        
        if self.proyeccion:
            identidad = self.proyeccion(x)
        
        return F.relu(salida + identidad)  # ← Suma residual


class BloqueResidualCuello(nn.Module):
    """
    Bloque bottleneck de ResNet-50/101/152.
    Reduce parámetros con conv 1×1 antes y después de la conv 3×3.
    
    Arquitectura:
      x → Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+x) → ReLU
    """
    
    expansion = 4
    
    def __init__(self, canales_entrada, canales_cuello, stride=1):
        super(BloqueResidualCuello, self).__init__()
        canales_salida = canales_cuello * self.expansion
        
        self.bloque = nn.Sequential(
            # 1×1: reducir canales
            nn.Conv2d(canales_entrada, canales_cuello, kernel_size=1, bias=False),
            nn.BatchNorm2d(canales_cuello),
            nn.ReLU(inplace=True),
            # 3×3: convolución principal
            nn.Conv2d(canales_cuello, canales_cuello, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(canales_cuello),
            nn.ReLU(inplace=True),
            # 1×1: expandir canales
            nn.Conv2d(canales_cuello, canales_salida, kernel_size=1, bias=False),
            nn.BatchNorm2d(canales_salida)
        )
        
        self.proyeccion = nn.Sequential(
            nn.Conv2d(canales_entrada, canales_salida, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(canales_salida)
        )
    
    def forward(self, x):
        return F.relu(self.bloque(x) + self.proyeccion(x))


# --- Probar bloques residuales ---
batch_x = torch.randn(4, 64, 28, 28)  # Batch de 4 imágenes

bloque_res = BloqueResidual(canales=64)
salida_res = bloque_res(batch_x)
print(f"BloqueResidual: {batch_x.shape} → {salida_res.shape}")

bloque_cuello = BloqueResidualCuello(canales_entrada=64, canales_cuello=16)
salida_cuello = bloque_cuello(batch_x)
print(f"BloqueBottleneck: {batch_x.shape} → {salida_cuello.shape}")

# --- Comparar gradientes: con y sin skip connections ---
print("\n" + "="*55)
print("ANÁLISIS DE GRADIENTES: Con vs Sin Skip Connections")
print("="*55)

class RedProfundaSinSkip(nn.Module):
    def __init__(self, n_bloques=8):
        super().__init__()
        self.capas = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 64), nn.ReLU())
            for _ in range(n_bloques)
        ])
        self.salida = nn.Linear(64, 10)
    
    def forward(self, x):
        for capa in self.capas:
            x = capa(x)
        return self.salida(x)


class RedProfundaConSkip(nn.Module):
    def __init__(self, n_bloques=8):
        super().__init__()
        self.capas = nn.ModuleList([
            nn.Linear(64, 64) for _ in range(n_bloques)
        ])
        self.salida = nn.Linear(64, 10)
    
    def forward(self, x):
        for capa in self.capas:
            x = F.relu(capa(x)) + x   # ← Skip connection
        return self.salida(x)


x_test = torch.randn(32, 64)
y_test = torch.randint(0, 10, (32,))
criterion = nn.CrossEntropyLoss()

for nombre, red in [("Sin Skip", RedProfundaSinSkip(8)),
                    ("Con Skip", RedProfundaConSkip(8))]:
    red.zero_grad()
    loss = criterion(red(x_test), y_test)
    loss.backward()
    
    normas = [p.grad.norm().item() for p in red.parameters() if p.grad is not None]
    print(f"\n{nombre} connections:")
    print(f"  Norma del gradiente (primera capa): {normas[0]:.6f}")
    print(f"  Norma del gradiente (última capa):  {normas[-1]:.6f}")
    print(f"  Ratio última/primera: {normas[-1]/max(normas[0], 1e-10):.2f}")
```

---

### 4.2 Transfer Learning con PyTorch

```python
# ============================================================
# TRANSFER LEARNING — ESTRATEGIAS
# ============================================================

from torchvision import models

print("="*55)
print("TRANSFER LEARNING — ESTRATEGIAS PRINCIPALES")
print("="*55)

# --- Estrategia 1: Feature Extraction (congelar todo menos la cabeza) ---
def crear_modelo_feature_extraction(num_clases, backbone='resnet18'):
    """
    Usa una red pre-entrenada como extractor de características.
    Solo entrena la última capa lineal.
    Ideal cuando tienes muy pocos datos de entrenamiento.
    """
    if backbone == 'resnet18':
        modelo = models.resnet18(pretrained=False)  # En prod: pretrained=True
        # Congelar todos los parámetros
        for param in modelo.parameters():
            param.requires_grad = False
        # Reemplazar sólo la cabeza de clasificación
        in_features = modelo.fc.in_features
        modelo.fc = nn.Linear(in_features, num_clases)
    return modelo


# --- Estrategia 2: Fine-Tuning (descongelar algunas capas) ---
def crear_modelo_fine_tuning(num_clases, capas_a_entrenar=2):
    """
    Fine-tuning: descongela sólo las últimas N capas del backbone.
    Ideal con datos moderados y cuando el dominio es similar a ImageNet.
    """
    modelo = models.resnet18(pretrained=False)  # En prod: pretrained=True
    
    # Congelar todo primero
    for param in modelo.parameters():
        param.requires_grad = False
    
    # Descongelar las últimas capas
    capas_resnet = [modelo.layer1, modelo.layer2, modelo.layer3, modelo.layer4]
    for capa in capas_resnet[-capas_a_entrenar:]:
        for param in capa.parameters():
            param.requires_grad = True
    
    # Reemplazar la cabeza
    in_features = modelo.fc.in_features
    modelo.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_clases)
    )
    
    return modelo


# Demostración
modelo_fe = crear_modelo_feature_extraction(num_clases=10)
modelo_ft = crear_modelo_fine_tuning(num_clases=10, capas_a_entrenar=2)

def contar_params(modelo):
    total = sum(p.numel() for p in modelo.parameters())
    entrenable = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, entrenable

total_fe, train_fe = contar_params(modelo_fe)
total_ft, train_ft = contar_params(modelo_ft)

print(f"\nFeature Extraction:")
print(f"  Parámetros totales:     {total_fe:,}")
print(f"  Parámetros entrenables: {train_fe:,}")
print(f"  Porcentaje entrenado:   {train_fe/total_fe*100:.1f}%")

print(f"\nFine-Tuning (2 últimas capas):")
print(f"  Parámetros totales:     {total_ft:,}")
print(f"  Parámetros entrenables: {train_ft:,}")
print(f"  Porcentaje entrenado:   {train_ft/total_ft*100:.1f}%")

print("\nGUÍA DE SELECCIÓN DE ESTRATEGIA:")
print("  Pocos datos + dominio similar  → Feature Extraction")
print("  Datos moderados + similar       → Fine-Tuning parcial")
print("  Muchos datos + diferente        → Entrenar desde cero")
print("  Muchos datos + similar          → Fine-Tuning completo")
```

---

## 📊 Análisis de Rendimiento

### Comparación CNN vs Red Densa en MNIST

```python
# ============================================================
# BENCHMARK: CNN vs RED DENSA
# ============================================================

# ---- Red densa equivalente ----
class RedDensa_MNIST(nn.Module):
    """
    Red totalmente conectada para MNIST.
    Misma cantidad aproximada de parámetros que CNN_MNIST.
    """
    def __init__(self, dropout=0.5):
        super(RedDensa_MNIST, self).__init__()
        self.red = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.red(x)


def entrenar_y_evaluar(modelo, train_loader, test_loader,
                       num_epochs=5, lr=0.001, device='cpu'):
    """Entrena un modelo y retorna métricas finales."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        modelo.train()
        for datos, etiquetas in train_loader:
            datos, etiquetas = datos.to(device), etiquetas.to(device)
            optimizer.zero_grad()
            loss = criterion(modelo(datos), etiquetas)
            loss.backward()
            optimizer.step()
    
    # Evaluación final
    modelo.eval()
    correcto = 0
    total = 0
    with torch.no_grad():
        for datos, etiquetas in test_loader:
            datos, etiquetas = datos.to(device), etiquetas.to(device)
            preds = modelo(datos).argmax(dim=1)
            correcto += (preds == etiquetas).sum().item()
            total += len(etiquetas)
    
    return correcto / total


import time

resultados = {}
modelos_comparar = {
    'CNN': CNN_MNIST(num_clases=10, dropout=0.3),
    'Red Densa': RedDensa_MNIST(dropout=0.3)
}

print("="*65)
print(f"{'Modelo':<15} {'Parámetros':<15} {'Tiempo (s)':<15} {'Precisión (%)'}")
print("="*65)

for nombre, model in modelos_comparar.items():
    model = model.to(device)
    total_p, _ = contar_params(model)
    
    inicio = time.time()
    acc = entrenar_y_evaluar(model, train_loader, test_loader,
                             num_epochs=5, device=device)
    tiempo = time.time() - inicio
    
    resultados[nombre] = {'params': total_p, 'tiempo': tiempo, 'acc': acc}
    print(f"{nombre:<15} {total_p:<15,} {tiempo:<15.1f} {acc*100:.2f}")

print("="*65)

# --- Gráfica de comparación ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

nombres = list(resultados.keys())
params_vals = [resultados[n]['params'] / 1000 for n in nombres]  # en miles
acc_vals = [resultados[n]['acc'] * 100 for n in nombres]
tiempo_vals = [resultados[n]['tiempo'] for n in nombres]

colores = ['#2196F3', '#FF5722']

axes[0].bar(nombres, params_vals, color=colores)
axes[0].set_ylabel('Parámetros (miles)')
axes[0].set_title('Parámetros del Modelo')
for i, v in enumerate(params_vals):
    axes[0].text(i, v + 0.5, f'{v:.0f}K', ha='center', fontweight='bold')

axes[1].bar(nombres, acc_vals, color=colores)
axes[1].set_ylabel('Precisión (%)')
axes[1].set_title('Precisión en Test')
axes[1].set_ylim([95, 100])
for i, v in enumerate(acc_vals):
    axes[1].text(i, v + 0.05, f'{v:.2f}%', ha='center', fontweight='bold')

axes[2].bar(nombres, tiempo_vals, color=colores)
axes[2].set_ylabel('Tiempo (segundos)')
axes[2].set_title('Tiempo de Entrenamiento (5 épocas)')
for i, v in enumerate(tiempo_vals):
    axes[2].text(i, v + 0.5, f'{v:.0f}s', ha='center', fontweight='bold')

plt.suptitle('CNN vs Red Densa — Comparación en MNIST', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('benchmark_cnn_vs_densa.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ Figura guardada: benchmark_cnn_vs_densa.png")

# --- Análisis del campo receptivo ---
print("\n" + "="*55)
print("ANÁLISIS DEL CAMPO RECEPTIVO")
print("="*55)

def calcular_campo_receptivo(num_capas, kernel_size=3, stride=1):
    """
    Calcula el campo receptivo acumulado capa por capa.
    RF_l = RF_(l-1) + (kernel_size - 1) * stride_acumulado
    """
    rf = 1
    stride_acum = 1
    print(f"\nKernel={kernel_size}×{kernel_size}, Stride={stride}")
    print(f"{'Capa':<8} {'RF':<15} {'Stride acum.'}")
    print("-" * 35)
    print(f"{'Input':<8} {rf}×{rf}{'':10} {stride_acum}")
    
    for capa in range(1, num_capas + 1):
        rf += (kernel_size - 1) * stride_acum
        stride_acum *= stride
        print(f"{'Conv '+str(capa):<8} {rf}×{rf}{'':10} {stride_acum}")
    return rf

rf_final = calcular_campo_receptivo(num_capas=6, kernel_size=3, stride=1)
print(f"\nDespués de 6 capas conv 3×3: campo receptivo = {rf_final}×{rf_final}")
print("(Cada neurona 've' una región de 13×13 píxeles de la imagen original)")
```

---

## 🎯 EJERCICIOS PROPUESTOS

### Nivel Básico

**Ejercicio B1: Convolución Manual con Filtro Personalizado**  
Implementa la función `detectar_bordes_diagonales(imagen)` que aplique dos kernels personalizados para detectar bordes en dirección diagonal (+45° y -45°). Retorna la magnitud combinada de ambas respuestas.

```python
# Plantilla de inicio
def detectar_bordes_diagonales(imagen):
    """
    Detecta bordes diagonales en una imagen en escala de grises.
    
    Kernel diagonal 1 (+45°):      Kernel diagonal 2 (-45°):
    [ 0  1  2]                     [ 2  1  0]
    [-1  0  1]                     [ 1  0 -1]
    [-2 -1  0]                     [ 0 -1 -2]
    
    Returns:
        magnitud: imagen con la magnitud de bordes diagonales
    """
    kernel_diag1 = # TODO: define el kernel +45°
    kernel_diag2 = # TODO: define el kernel -45°
    
    resp1 = convolve2d_manual(imagen, kernel_diag1)
    resp2 = convolve2d_manual(imagen, kernel_diag2)
    
    return # TODO: combina las respuestas
```

**Ejercicio B2: Cálculo de Parámetros**  
Dado el siguiente stack de capas, calcula manualmente el número exacto de parámetros de cada capa y el total. Verifica tu respuesta con PyTorch.

```
Input: 3×224×224 (imagen RGB)
Conv1: 64 filtros, 7×7, stride=2, padding=3
Conv2: 128 filtros, 3×3, stride=1, padding=1
Conv3: 256 filtros, 3×3, stride=1, padding=1
FC:    1000 clases
```

**Ejercicio B3: Convolución Vectorizada**  
Reimplementa `convolve2d_manual` usando `numpy.lib.stride_tricks` para evitar los bucles `for` y hacer la operación completamente vectorizada. Mide la diferencia de velocidad con `time.time()`.

---

### Nivel Intermedio

**Ejercicio I1: Clasificación CIFAR-10**  
El dataset CIFAR-10 contiene imágenes a color (3×32×32) de 10 categorías. Diseña y entrena una CNN desde cero que alcance >80% de precisión en el conjunto de prueba.

```python
# Requisitos mínimos de la arquitectura:
# - Al menos 4 capas convolucionales
# - Batch Normalization después de cada conv
# - Data Augmentation: flip horizontal, crop aleatorio, jitter de color
# - Dropout antes de la capa de clasificación final
# - Scheduler de learning rate

from torchvision import datasets

transform_cifar_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# TODO: Define tu arquitectura CNN para CIFAR-10
class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        # Tu implementación aquí
        pass
    
    def forward(self, x):
        pass
```

**Ejercicio I2: Visualización de Saliency Maps**  
Implementa Gradient-based Saliency Maps para visualizar qué regiones de una imagen son más importantes para la predicción de una CNN.

```python
def calcular_saliency_map(modelo, imagen_tensor, clase_objetivo):
    """
    Calcula el saliency map basado en gradientes.
    
    El saliency map indica qué píxeles, si se modifican levemente,
    más afectarían la probabilidad de la clase objetivo.
    
    Returns:
        saliency: np.array (H, W) — importancia de cada píxel
    """
    imagen_tensor = imagen_tensor.unsqueeze(0).requires_grad_(True)
    
    modelo.zero_grad()
    salida = modelo(imagen_tensor)
    
    # Backprop respecto a la clase objetivo
    salida[0, clase_objetivo].backward()
    
    # El gradiente respecto a la imagen es el saliency
    saliency = # TODO: extrae el gradiente absoluto
    saliency = # TODO: reduce a 2D tomando el máximo sobre los canales
    
    return saliency
```

**Ejercicio I3: Implementar Global Average Pooling (GAP)**  
Modifica la clase `CNN_MNIST` para reemplazar las capas densas con Global Average Pooling seguido de una única capa lineal. Compara el número de parámetros y la precisión resultante.

---

### Nivel Avanzado

**Ejercicio A1: Arquitectura ResNet-20 para CIFAR-10**  
Implementa completamente la arquitectura ResNet-20 original (He et al., 2016) para CIFAR-10 e intenta alcanzar >90% de precisión.

```python
class ResNet20_CIFAR10(nn.Module):
    """
    ResNet-20 para CIFAR-10 siguiendo el paper original.
    
    Estructura:
    - 1 capa conv inicial 3×3 con 16 filtros
    - 3 bloques de 3 capas residuales cada uno:
      * Bloque 1: 16 filtros, stride=1
      * Bloque 2: 32 filtros, stride=2
      * Bloque 3: 64 filtros, stride=2
    - Global Average Pooling
    - FC → 10 clases
    
    Total: 6×3 + 2 = 20 capas aprendibles
    """
    def __init__(self):
        super().__init__()
        # TODO: Implementar la arquitectura completa
        pass
```

**Ejercicio A2: Depthwise Separable Convolutions (MobileNet)**  
Implementa el bloque de convolución separable en profundidad (Depthwise Separable Convolution) al estilo MobileNet y mide la reducción en parámetros y FLOPs comparado con una conv estándar equivalente.

```python
class BloqueDepthwiseSeparable(nn.Module):
    """
    Convolución separable en profundidad:
    
    Convolución estándar: H×W×Cin×Cout×K×K parámetros
    Separable:
      1. Depthwise: H×W×Cin con kernel K×K por canal → Cin×K×K parámetros
      2. Pointwise: 1×1 conv para combinar canales → Cin×Cout parámetros
    
    Reducción: ~K² veces menos parámetros
    """
    def __init__(self, c_in, c_out, kernel_size=3, stride=1):
        super().__init__()
        # TODO: implementar
        pass
```

**Ejercicio A3: Detección de Objetos con CNN (Proyecto)**  
Usando una CNN pre-entrenada como backbone, implementa un detector de objetos simple tipo YOLO-tiny para detectar objetos en el dataset Pascal VOC mini (subset de 5 clases). El modelo debe predecir bounding boxes y etiquetas de clase.

Requisitos:
- Dataset: Pascal VOC 2012 (5 clases: persona, gato, perro, coche, avión)
- Backbone: ResNet-18 pre-entrenado
- Cabeza de detección: 3×3 conv + FC con salida `(S×S×(5+C))` para grid `S=7`
- Función de pérdida: MSE para coordenadas + Cross-Entropy para clases
- mAP@0.5 > 0.30

---

## 📝 Entregables

### Código Fuente

| Archivo | Descripción |
|---|---|
| `convoluciones_numpy.py` | Implementación de convolución 2D y filtros clásicos |
| `cnn_pytorch.py` | Arquitectura CNN completa con PyTorch |
| `entrenamiento_mnist.py` | Pipeline de entrenamiento en MNIST |
| `visualizaciones.py` | Filtros, feature maps, saliency maps |
| `benchmark.py` | Comparación CNN vs red densa |
| `ejercicio_*.py` | Archivos de solución de ejercicios |

### Modelos Guardados

- `mejor_cnn_mnist.pth` — Pesos del mejor modelo CNN en MNIST
- `cnn_cifar10.pth` — Modelo entrenado en CIFAR-10 (Ejercicio I1)

### Documentación

- Notebook Jupyter documentado con resultados de cada experimento
- Gráficas de curvas de entrenamiento con comentarios de análisis
- Tabla comparativa de arquitecturas CNN analizadas

### Reporte Final

El reporte debe incluir (2-4 páginas, formato libre):

1. **Experimentos realizados**: descripción breve de cada parte del laboratorio
2. **Resultados cuantitativos**: tablas con métricas (precisión, pérdida, parámetros)
3. **Análisis de filtros**: ¿qué detectan los filtros aprendidos en la primera capa?
4. **Comparación CNN vs Densa**: ¿cuándo vale la pena usar CNN?
5. **Reflexión final**: qué aprendiste y qué te sorprendió más

---

## 🎯 Criterios de Evaluación (CDIO)

### Concebir (25%) — Comprensión Conceptual

**Objetivo:** Demostrar comprensión profunda de los principios que hacen a las CNNs superiores para datos visuales.

✅ **Evidencias esperadas:**
- Explica por qué una CNN necesita muchos menos parámetros que una red densa equivalente
- Describe correctamente la jerarquía de características (bordes → texturas → objetos)
- Justifica la elección de hiperparámetros: kernel size, número de filtros, stride, padding
- Explica el problema de la degradación y cómo ResNet lo resuelve
- Relaciona Transfer Learning con el concepto de jerarquía de características

### Diseñar (25%) — Diseño de Arquitecturas

**Objetivo:** Diseñar arquitecturas CNN apropiadas para distintas tareas y datasets.

✅ **Evidencias esperadas:**
- Diseña una CNN para CIFAR-10 con justificación explícita de cada capa
- Aplica la regla de duplicar filtros al reducir dimensionalidad espacial
- Incorpora Batch Normalization y Dropout en lugares apropiados
- Selecciona la estrategia correcta de Transfer Learning según el tamaño del dataset
- Calcula manualmente el número de parámetros de su arquitectura

### Implementar (30%) — Implementación y Entrenamiento

**Objetivo:** Implementar y entrenar CNNs funcionales con código limpio y correcto.

✅ **Evidencias esperadas:**
- Convolución 2D implementada correctamente desde cero (verificada contra SciPy)
- CNN en PyTorch entrena sin errores y converge en MNIST (>98%)
- Implementa Data Augmentation con al menos 3 transformaciones
- Usa callbacks: guardado del mejor modelo, scheduler de LR
- Visualizaciones de filtros y feature maps generadas correctamente

### Operar (20%) — Análisis y Operación

**Objetivo:** Interpretar resultados, diagnosticar problemas y mejorar modelos.

✅ **Evidencias esperadas:**
- Interpreta las curvas de pérdida y precisión (underfitting, overfitting, convergencia)
- Identifica filtros aprendidos con interpretación visual razonable
- Compara CNN vs red densa con justificación basada en datos
- Diagnostica y resuelve al menos un problema durante el entrenamiento (overfitting, NaN loss, etc.)
- Propone mejoras con fundamento técnico

---

## 📋 Rúbrica de Evaluación

| Criterio | Excelente (100%) | Bueno (75%) | Aceptable (50%) | Insuficiente (25%) |
|---|---|---|---|---|
| **Convolución desde cero** (15 pts) | Implementada sin errores, vectorizable, verificada contra SciPy; maneja padding/stride correctamente | Funciona correctamente con pequeños casos pero sin validación rigurosa | Implementada con algún error menor; resultado correcto en casos simples | No implementada o con errores conceptuales graves |
| **Filtros clásicos** (10 pts) | Implementa ≥5 filtros, visualización clara, explica la interpretación visual de cada uno | ≥3 filtros implementados con visualización | 2 filtros implementados | Solo 1 filtro o sin visualización |
| **Fórmulas de dimensiones** (10 pts) | Calcula correctamente todas las configuraciones de la tabla; justifica cada paso | Calcula correctamente la mayoría con algún error puntual | Aplica la fórmula con errores frecuentes | No aplica la fórmula correctamente |
| **CNN PyTorch** (20 pts) | Arquitectura bien documentada, entrena >98% en MNIST, usa BN, Dropout y scheduler | >97% en MNIST con arquitectura razonable | >95% en MNIST | No entrena o <95% |
| **Visualizaciones** (10 pts) | Filtros y feature maps de las 3 capas visualizados e interpretados | 2 capas visualizadas | Solo 1 capa o sin interpretación | Sin visualizaciones |
| **Skip Connections** (10 pts) | BloqueResidual implementado correctamente, análisis de gradientes realizado | Implementado correctamente sin análisis | Implementado con errores menores | No implementado |
| **Transfer Learning** (10 pts) | Ambas estrategias implementadas, comparación cuantitativa, guía de selección justificada | Una estrategia correcta | Código de transfer learning sin evaluación | No implementado |
| **Benchmark CNN vs Densa** (10 pts) | Tabla completa con métricas, gráfica comparativa, análisis escrito de ventajas/desventajas | Comparación numérica sin análisis | Solo precisión comparada | Sin comparación |
| **Reporte Final** (5 pts) | Reporte completo, análisis profundo, conclusiones basadas en evidencia | Reporte con la mayoría de secciones | Reporte incompleto pero con reflexión genuina | Sin reporte o copia de la guía |

**Escala:**
- 90-100: Sobresaliente
- 75-89: Bueno  
- 60-74: Aceptable
- <60: Necesita refuerzo

---

## 📚 Referencias Adicionales

### Papers Fundamentales

1. **LeCun et al. (1998)** — *Gradient-Based Learning Applied to Document Recognition*  
   El paper original de LeNet-5. Disponible en: [http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

2. **Krizhevsky, Sutskever & Hinton (2012)** — *ImageNet Classification with Deep CNNs (AlexNet)*  
   El paper que inició el auge del Deep Learning moderno.

3. **Simonyan & Zisserman (2015)** — *Very Deep CNNs for Large-Scale Image Recognition (VGGNet)*  
   Disponible en: [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)

4. **He et al. (2016)** — *Deep Residual Learning for Image Recognition (ResNet)*  
   Disponible en: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

5. **Howard et al. (2017)** — *MobileNets: Efficient CNNs for Mobile Vision Applications*  
   Disponible en: [https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)

### Tutoriales y Cursos

6. **CS231n Stanford** — *Convolutional Neural Networks for Visual Recognition*  
   [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/) — El mejor curso sobre CNNs

7. **CNN Explainer** — Visualización interactiva de operaciones CNN  
   [https://poloclub.github.io/cnn-explainer/](https://poloclub.github.io/cnn-explainer/)

8. **PyTorch Tutorials — Training a Classifier**  
   [https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

9. **Distill.pub — Feature Visualization**  
   [https://distill.pub/2017/feature-visualization/](https://distill.pub/2017/feature-visualization/) — Cómo visualizar lo que aprenden las CNNs

10. **FastAI Practical Deep Learning**  
    [https://course.fast.ai/](https://course.fast.ai/) — Enfoque práctico

### Libros

11. **Goodfellow, Bengio & Courville** — *Deep Learning* (2016), Capítulo 9: Convolutional Networks  
    Disponible gratuitamente en: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

12. **Chollet** — *Deep Learning with Python* (2nd ed., 2021), Manning

13. **Zhang et al.** — *Dive into Deep Learning*  
    [https://d2l.ai/](https://d2l.ai/) — Libro interactivo con código ejecutable

### Herramientas y Repositorios

14. **torchvision.models** — Implementaciones oficiales de arquitecturas CNN  
    [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)

15. **timm (PyTorch Image Models)** — Biblioteca con +600 modelos pre-entrenados  
    `pip install timm` | [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

16. **Netron** — Visualizador de arquitecturas de redes neuronales  
    [https://netron.app/](https://netron.app/)

---

## 🎓 Notas Finales

### Lo Que Debes Recordar

Las CNNs revolucionaron la visión computacional por tres principios simples pero poderosos:

```
1. CONECTIVIDAD LOCAL
   Cada neurona ve sólo una pequeña región de la entrada.
   → Aprovecha que los píxeles cercanos están correlacionados.

2. COMPARTICIÓN DE PESOS
   El mismo filtro se aplica a toda la imagen.
   → 150M parámetros (densa) vs 60K parámetros (CNN) para 224×224.

3. JERARQUÍA DE CARACTERÍSTICAS
   Capas apiladas construyen representaciones cada vez más abstractas.
   → Capa 1: bordes → Capa 5: orejas de gato → Capa 10: gatos completos.
```

### Conceptos Clave por Recapitular

| Concepto | Fórmula/Definición |
|---|---|
| Dimensión salida | `(H - K + 2P) / S + 1` |
| Parámetros conv | `(Kh × Kw × Cin + 1) × Nfiltros` |
| Padding "same" | `P = (K - 1) / 2` (con stride=1) |
| Campo receptivo | Se duplica aproximadamente con cada capa conv 3×3 |
| Skip connection | `F(x) = H(x) - x` → Aprende el residuo |
| Global Avg Pool | `C×H×W → C` (sin parámetros) |

### Relación con Laboratorios Anteriores

Este laboratorio integra conceptos de todos los laboratorios previos:

- **Labs 01-02**: Neuronas y redes → ahora organizadas en convoluciones
- **Lab 03**: ReLU, la activación más usada en CNNs
- **Lab 04**: Cross-Entropy loss → función de pérdida estándar para clasificación
- **Lab 05**: Backpropagation → ahora a través de capas convolucionales
- **Lab 06**: Entrenamiento, SGD, Adam → aplicados aquí con scheduler
- **Lab 07**: Métricas → precisión, confusión matrix, Precision/Recall
- **Lab 08**: PyTorch → framework usado en todo este lab
- **Lab 09**: IA Generativa → CNNs son el backbone de GANs y Diffusion Models

### Próximos Pasos

Después de dominar las CNNs, el siguiente laboratorio explora:

> 👉 **[Lab 11: Redes Neuronales Recurrentes y LSTM](../Lab11_Redes_Neuronales_Recurrentes_LSTM/)**
> 
> Aprenderás arquitecturas diseñadas para **datos secuenciales**: texto, audio, series de tiempo y video. Las RNNs y LSTMs procesan la dimensión temporal de la misma manera que las CNNs procesan la dimensión espacial.

---

## ✅ Checklist de Verificación

Antes de entregar, verifica que has completado todo:

### Parte 1 — Convolución
- [ ] Implementé `convolve2d_manual` con padding y stride funcionando correctamente
- [ ] Apliqué al menos 4 filtros clásicos y generé visualizaciones
- [ ] Verifiqué la fórmula de dimensiones en todas las configuraciones de la tabla
- [ ] Entiendo la diferencia entre padding "valid" y "same"

### Parte 2 — Capas CNN
- [ ] Implementé `CapaConvolucional` con múltiples filtros y canales
- [ ] Implementé `CapaPooling` para Max y Average pooling
- [ ] Construí el pipeline completo: Conv → Pool → Flatten → Dense desde cero
- [ ] Puedo calcular parámetros manualmente para cualquier configuración

### Parte 3 — CNN con PyTorch
- [ ] Definí `CNN_MNIST` con al menos 3 bloques convolucionales
- [ ] El modelo entrena correctamente y alcanza >97% en MNIST
- [ ] Guardé el mejor modelo con `torch.save`
- [ ] Visualicé filtros de la primera capa convolucional
- [ ] Visualicé feature maps de activación usando hooks

### Parte 4 — Arquitecturas Avanzadas
- [ ] Implementé `BloqueResidual` con skip connection funcional
- [ ] Analicé la diferencia en gradientes con y sin skip connections
- [ ] Entiendo las dos estrategias de Transfer Learning
- [ ] Sé cuándo usar Feature Extraction vs Fine-Tuning

### Benchmark y Análisis
- [ ] Comparé CNN vs red densa en MNIST (precisión, parámetros, tiempo)
- [ ] Calculé el campo receptivo para al menos 3 profundidades
- [ ] Generé y guardé todas las figuras (filtros, feature maps, curvas)
- [ ] Escribí el reporte final con análisis y conclusiones

### Ejercicios (mínimo 2 por nivel)
- [ ] Ejercicio Básico 1: Filtros diagonales ✓
- [ ] Ejercicio Básico 2: Cálculo de parámetros ✓
- [ ] Ejercicio Intermedio 1: CNN en CIFAR-10 ✓
- [ ] Ejercicio Avanzado: uno a elección ✓

### Calidad del Código
- [ ] Código comentado y organizado en funciones/clases
- [ ] Variables con nombres descriptivos en español o inglés (consistente)
- [ ] Sin código de debug o celdas con errores en el notebook final
- [ ] Todas las figuras tienen título, etiquetas de ejes y leyenda

---

*Guía desarrollada para el curso de Redes Neuronales — Lab 10: CNNs*  
*Conecta hacia atrás con Labs 01-09 y hacia adelante con Lab 11 (RNNs/LSTM) y Lab 12 (Transformers)*
