# Teoría: Redes Neuronales Convolucionales (CNN)

## 1. Introducción

Las Redes Neuronales Convolucionales (CNN o ConvNets) son una clase especializada de redes neuronales diseñadas específicamente para procesar datos con estructura de cuadrícula, como imágenes.

### ¿Por qué CNNs para Imágenes?

**Problema con Redes Densas:**
- Una imagen 224x224 RGB tiene 224 × 224 × 3 = 150,528 píxeles
- Primera capa densa con 1000 neuronas: 150,528 × 1000 = 150 millones de parámetros
- Extremadamente costoso computacionalmente
- Propensa al overfitting
- Ignora estructura espacial de la imagen

**Solución: CNNs**
- Aprovechan estructura espacial local
- Comparten parámetros (mismo filtro en toda la imagen)
- Invariancia a traslación
- Jerarquía de características: bordes → formas → objetos

## 2. Operación de Convolución

### 2.1 Convolución 1D

La convolución es una operación matemática entre dos funciones. En señales discretas:

```
(f * g)[n] = Σ f[m] · g[n - m]
```

**Ejemplo práctico:**
- Input: [3, 4, 5, 6, 7]
- Kernel: [1, 0, -1]
- Output: aplica el kernel deslizándolo sobre el input

### 2.2 Convolución 2D (Imágenes)

En imágenes, trabajamos con convolución 2D:

```python
# Input: matriz H × W
# Kernel: matriz K_h × K_w
# Output: (H - K_h + 1) × (W - K_w + 1)

Output[i,j] = Σ Σ Input[i+m, j+n] * Kernel[m, n]
              m n
```

**Visualización:**
```
Input (5×5):        Kernel (3×3):      Output (3×3):
[1 2 3 4 5]         [1  0 -1]          [...]
[1 2 3 4 5]         [1  0 -1]
[1 2 3 4 5]         [1  0 -1]
[1 2 3 4 5]
[1 2 3 4 5]
```

### 2.3 Filtros Clásicos

**Detector de Bordes Verticales:**
```
[-1  0  1]
[-1  0  1]
[-1  0  1]
```

**Detector de Bordes Horizontales:**
```
[-1 -1 -1]
[ 0  0  0]
[ 1  1  1]
```

**Desenfoque (Blur):**
```
[1/9  1/9  1/9]
[1/9  1/9  1/9]
[1/9  1/9  1/9]
```

**Detección de Esquinas (Sobel):**
```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

## 3. Componentes de una CNN

### 3.1 Capa Convolucional

**Parámetros:**
- **Número de filtros**: Cuántos feature maps genera
- **Tamaño del kernel**: Típicamente 3×3, 5×5, 7×7
- **Stride**: Paso del desplazamiento (1, 2, ...)
- **Padding**: Relleno de bordes (valid, same)

**Cálculo de dimensiones:**
```
Output_height = (Input_height - Kernel_height + 2*Padding) / Stride + 1
Output_width = (Input_width - Kernel_width + 2*Padding) / Stride + 1
Output_channels = Number_of_filters
```

**Ejemplo:**
```python
Input: 32×32×3 (imagen RGB)
Conv2D: 64 filtros, kernel 5×5, stride=1, padding=0
Output: 28×28×64

Cálculo: (32 - 5 + 0) / 1 + 1 = 28
```

**Número de parámetros:**
```
Params = (K_h × K_w × Input_channels + 1) × Num_filters

Ejemplo: (5 × 5 × 3 + 1) × 64 = 4,864 parámetros
```

### 3.2 Padding

**Valid (sin padding):**
- No agrega bordes
- Output es más pequeño que input
- Se pierden píxeles de los bordes

**Same (con padding):**
- Agrega bordes de ceros
- Output tiene mismo tamaño que input (con stride=1)
- Preserva información de bordes

```python
# Para mantener tamaño con stride=1:
Padding = (Kernel_size - 1) / 2

# Ejemplo con kernel 3×3:
Padding = (3 - 1) / 2 = 1
```

### 3.3 Stride

- **Stride = 1**: Mueve filtro 1 píxel a la vez (más overlap)
- **Stride = 2**: Mueve filtro 2 píxeles (menos overlap, reduce tamaño)
- **Stride > 1**: Alternativa a pooling para reducir dimensionalidad

### 3.4 Capas de Pooling

**Propósito:**
1. Reducir dimensionalidad espacial
2. Reducir parámetros y computación
3. Proveer invariancia a pequeñas traslaciones
4. Controlar overfitting

**Max Pooling:**
```python
# Toma el valor máximo en cada ventana
Input (4×4):         Output (2×2) con 2×2 pool:
[1  3  2  4]         [6  8]
[5  6  7  8]   →     [9  11]
[9  2  1  3]
[4  5  10 11]
```

**Average Pooling:**
```python
# Promedia valores en cada ventana
Input (4×4):         Output (2×2) con 2×2 pool:
[1  3  2  4]         [3.75  5.25]
[5  6  7  8]   →     [5.0   6.25]
[9  2  1  3]
[4  5  10 11]
```

**Características:**
- Reduce tamaño espacial pero no número de canales
- No tiene parámetros entrenables
- Típicamente 2×2 con stride=2

### 3.5 Global Average Pooling (GAP)

- Promedia toda la feature map a un solo valor
- Convierte feature map H×W×C en vector de tamaño C
- Reemplaza capas densas finales
- Menos parámetros, menos overfitting

```python
Input: 7×7×512
Global Average Pooling
Output: 1×1×512 = 512
```

## 4. Arquitectura de una CNN

### 4.1 Estructura Típica

```
Input Image
    ↓
[CONV → ReLU → POOL] × N
    ↓
[CONV → ReLU → POOL] × M
    ↓
Flatten
    ↓
[FC → ReLU] × K
    ↓
FC → Softmax
    ↓
Output (Classes)
```

### 4.2 Jerarquía de Características

**Capas Tempranas (cerca del input):**
- Detectan características simples
- Bordes, colores, texturas
- Campo receptivo pequeño

**Capas Medias:**
- Combinan características simples
- Formas, patrones
- Campo receptivo mediano

**Capas Profundas:**
- Características de alto nivel
- Partes de objetos, objetos completos
- Campo receptivo grande

### 4.3 Campo Receptivo (Receptive Field)

El campo receptivo de una neurona es la región de la entrada que afecta su valor.

**Cálculo:**
```python
# Capa 1: kernel 3×3 → receptive field = 3×3
# Capa 2: kernel 3×3 → receptive field = 5×5
# Capa 3: kernel 3×3 → receptive field = 7×7

# Fórmula general:
RF_l = RF_(l-1) + (kernel_size - 1) * Π(stride anterior)
```

## 5. Arquitecturas CNN Famosas

### 5.1 LeNet-5 (1998) - Yann LeCun

**Arquitectura:**
```
INPUT → CONV1 → POOL1 → CONV2 → POOL2 → FC1 → FC2 → OUTPUT
32×32  →  28×28 → 14×14 → 10×10 →  5×5  → 120 → 84  →  10
```

**Características:**
- Primera CNN exitosa
- Reconocimiento de dígitos (MNIST)
- Usaba Tanh en lugar de ReLU

### 5.2 AlexNet (2012) - Krizhevsky, Sutskever, Hinton

**Arquitectura:**
```
227×227×3 → CONV1(96) → POOL → CONV2(256) → POOL → 
CONV3(384) → CONV4(384) → CONV5(256) → POOL → FC(4096) → FC(4096) → FC(1000)
```

**Innovaciones:**
- ReLU activations (6× más rápido que tanh)
- Dropout para regularización
- Data augmentation
- GPU training
- Ganó ImageNet 2012 (top-5 error: 15.3%)

### 5.3 VGG (2014) - Visual Geometry Group, Oxford

**Características:**
- Usa solo conv 3×3 y pool 2×2
- Arquitectura muy profunda: VGG-16 (16 capas), VGG-19 (19 capas)
- Simple pero muy efectiva
- Muchos parámetros (~138M en VGG-16)

**VGG-16 Arquitectura:**
```
64 → 64 → POOL → 
128 → 128 → POOL → 
256 → 256 → 256 → POOL → 
512 → 512 → 512 → POOL → 
512 → 512 → 512 → POOL → 
FC(4096) → FC(4096) → FC(1000)
```

### 5.4 ResNet (2015) - Microsoft Research

**Innovación Principal: Skip Connections**
```python
# Bloque residual
x → [CONV → ReLU → CONV] → (+) → ReLU
 ↓_________________________↑
        (skip connection)
```

**Ventajas:**
- Permite entrenar redes muy profundas (50, 101, 152 capas)
- Soluciona problema de degradación
- Gradientes fluyen directamente por skip connections
- Ganó ImageNet 2015 (3.57% top-5 error)

**Fórmula:**
```
F(x) = H(x) - x
Output = F(x) + x = H(x)

Donde:
- x: entrada del bloque
- H(x): salida deseada
- F(x): residuo que la red debe aprender
```

### 5.5 Inception / GoogLeNet (2014)

**Innovación: Módulo Inception**
- Aplica múltiples filtros en paralelo (1×1, 3×3, 5×5, pooling)
- Concatena resultados
- Reduce parámetros con convoluciones 1×1

### 5.6 MobileNet (2017)

**Innovación: Depthwise Separable Convolutions**
- Separa convolución espacial y por canales
- Mucho más eficiente (menos parámetros y cómputo)
- Ideal para dispositivos móviles

## 6. Técnicas Importantes

### 6.1 Batch Normalization

Normaliza activaciones entre batches:
```python
# Para cada feature map:
y = γ * (x - μ) / σ + β

Donde:
- μ, σ: media y desviación estándar del batch
- γ, β: parámetros aprendibles
```

**Beneficios:**
- Acelera entrenamiento
- Permite learning rates más altos
- Reduce dependencia de inicialización
- Regularización (ligero efecto de dropout)

### 6.2 Data Augmentation

Aumenta tamaño del dataset con transformaciones:
- Rotación, traslación, escala
- Flip horizontal/vertical
- Cambios de brillo, contraste
- Recortes aleatorios (random crops)
- Mezcla (mixup, cutmix)

### 6.3 Transfer Learning

Usa red pre-entrenada en ImageNet:

**Estrategia 1: Feature Extraction**
- Congela capas convolucionales
- Re-entrena solo capas finales
- Usa cuando tienes pocos datos

**Estrategia 2: Fine-Tuning**
- Descongela algunas capas finales
- Re-entrena con learning rate bajo
- Usa cuando tienes datos moderados

### 6.4 Convoluciones 1×1

**Propósitos:**
1. Cambiar número de canales (dimensionalidad)
2. Reducir parámetros antes de conv grandes
3. Agregar no-linealidad extra

```python
Input: 28×28×192
Conv 1×1 con 64 filtros
Output: 28×28×64

# Reduce de 192 a 64 canales
# Parámetros: 192 × 64 = 12,288
```

## 7. Aplicaciones Avanzadas

### 7.1 Clasificación de Imágenes
- Reconocer categorías de objetos
- Estado del arte: >95% top-5 en ImageNet

### 7.2 Detección de Objetos
- Localizar y clasificar múltiples objetos
- Arquitecturas: YOLO, Faster R-CNN, SSD
- Salida: bounding boxes + clases

### 7.3 Segmentación Semántica
- Clasificar cada píxel
- Arquitecturas: U-Net, SegNet, DeepLab
- Aplicaciones: conducción autónoma, medicina

### 7.4 Segmentación de Instancias
- Detectar y segmentar cada instancia
- Arquitectura: Mask R-CNN
- Segmentación a nivel de objeto individual

### 7.5 Face Recognition
- Verificación: ¿Son la misma persona?
- Identificación: ¿Quién es esta persona?
- Arquitecturas: FaceNet, DeepFace

### 7.6 Neural Style Transfer
- Aplicar estilo artístico a foto
- Preservar contenido, cambiar estilo
- Usado en apps de filtros artísticos

### 7.7 Diagnóstico Médico
- Detección de tumores en radiografías
- Clasificación de lesiones dermatológicas
- Segmentación de órganos en MRI/CT

## 8. Consideraciones Prácticas

### 8.1 Diseño de Arquitectura

**Reglas generales:**
1. Aumenta profundidad gradualmente
2. Duplica filtros cuando reduces tamaño espacial
3. Usa padding para mantener información de bordes
4. Batch normalization después de cada conv
5. ReLU como activación estándar

**Progresión común:**
```
32×32×3  →  32×32×64  →  16×16×128  →  8×8×256  →  4×4×512
         (conv+BN)    (pool)       (pool)       (pool)
```

### 8.2 Regularización

1. **Dropout**: Típicamente 0.5 en capas FC
2. **Weight Decay**: L2 regularization, λ=1e-4
3. **Data Augmentation**: Crítico para imágenes
4. **Batch Normalization**: Regularización implícita

### 8.3 Optimización

**Learning Rate Schedule:**
- Empezar con LR alto (ej: 0.1)
- Reducir cuando plateau (×0.1)
- O usar cosine annealing, step decay

**Optimizadores recomendados:**
- SGD + Momentum (0.9)
- Adam (α=0.001, β1=0.9, β2=0.999)

### 8.4 Problemas Comunes

**Overfitting:**
- Más data augmentation
- Más dropout
- Reducir capacidad del modelo

**Underfitting:**
- Modelo más profundo/ancho
- Entrenar más epochs
- Reducir regularización

**Convergencia lenta:**
- Batch normalization
- Learning rate más alto
- Mejor inicialización (Xavier, He)

## 9. Matemáticas de Backpropagation en CNN

### 9.1 Gradiente de Convolución

Para capa convolucional:
```python
# Forward:
output = input * kernel

# Backward:
∂L/∂input = ∂L/∂output * kernel_rotated_180
∂L/∂kernel = input * ∂L/∂output
```

### 9.2 Gradiente de Max Pooling

```python
# Forward: guarda índices del máximo
max_idx = argmax(window)

# Backward: gradiente va solo a posición del máximo
∂L/∂input[max_idx] = ∂L/∂output
∂L/∂input[otros] = 0
```

## 10. Resumen

**CNNs son poderosas porque:**
1. ✅ Aprovechan estructura espacial local
2. ✅ Comparten parámetros (menos overfitting)
3. ✅ Invariancia a traslación
4. ✅ Jerarquía de características
5. ✅ Escalables a imágenes grandes

**Componentes clave:**
- Convolución: detecta patrones locales
- Pooling: reduce dimensionalidad
- Stride/Padding: controla tamaño de salida
- Arquitectura profunda: jerarquía de características

**Para recordar:**
- Kernels pequeños (3×3) son preferidos
- Batch normalization es casi siempre beneficioso
- Data augmentation es crítico
- Transfer learning cuando tienes pocos datos
- ResNet y sus skip connections revolucionaron el campo

---

**¡Las CNNs son el pilar de la visión computacional moderna!** 🖼️👁️
