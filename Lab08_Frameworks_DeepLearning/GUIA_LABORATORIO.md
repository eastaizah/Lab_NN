# Guía de Laboratorio: Frameworks de Deep Learning

## 📋 Información del Laboratorio

**Título:** Fundamentos de Deep Learning - Frameworks Modernos  
**Código:** Lab 08  
**Duración:** 3-4 horas  
**Nivel:** Intermedio-Avanzado  

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender las ventajas de usar frameworks de deep learning
2. Implementar redes neuronales en PyTorch desde cero
3. Implementar redes neuronales en TensorFlow/Keras
4. Utilizar diferenciación automática (Autograd)
5. Aprovechar aceleración con GPU
6. Comparar PyTorch y TensorFlow en casos prácticos
7. Migrar código desde implementaciones NumPy a frameworks
8. Usar utilidades modernas (DataLoaders, Optimizers, etc.)
9. Entrenar modelos de manera eficiente
10. Guardar, cargar y desplegar modelos

## 📚 Prerrequisitos

### Conocimientos

- Python avanzado (POO, decoradores)
- NumPy sólido (todos los labs anteriores)
- Redes neuronales, backpropagation, entrenamiento
- Conceptos de GPU computing (básicos)

### Software

- Python 3.8+
- PyTorch 1.9+ (`pip install torch torchvision`)
- TensorFlow 2.6+ (`pip install tensorflow`)
- NumPy, Matplotlib
- CUDA (opcional, para GPU)

### Material de Lectura

Antes de comenzar, lee:
- `teoria.md` - Comparación de frameworks
- `README.md` - Recursos y estructura
- Documentación oficial de PyTorch y TensorFlow

## 📖 Introducción

### Del Código Manual a los Frameworks

Felicidades! Has llegado lejos implementando todo desde cero:
- ✓ Neuronas y capas (Lab 01)
- ✓ Funciones de activación (Lab 03)
- ✓ Funciones de pérdida (Lab 04)
- ✓ Backpropagation (Lab 05)
- ✓ Entrenamiento completo (Lab 06)
- ✓ Evaluación rigurosa (Lab 07)

**Ahora es tiempo de usar las herramientas profesionales.**

### ¿Por Qué Usar Frameworks?

**Sin frameworks (lo que has hecho):**
```python
# Implementar forward pass
z = np.dot(W, x) + b
a = sigmoid(z)

# Implementar backward pass
dz = a - y
dW = np.dot(dz, x.T)
db = np.sum(dz)

# Actualizar manualmente
W -= learning_rate * dW
b -= learning_rate * db
```

**Con frameworks:**
```python
# PyTorch hace todo automáticamente
output = model(x)
loss = criterion(output, y)
loss.backward()  # ¡Gradientes automáticos!
optimizer.step()  # ¡Actualización automática!
```

### Ventajas Principales

**1. Autograd (Diferenciación Automática)**
- No más backpropagation manual
- Sin errores en derivadas
- Soporta operaciones complejas

**2. Optimización de Performance**
- Operaciones optimizadas en C++/CUDA
- Paralelización automática
- 10-100x más rápido que NumPy

**3. GPU Acceleration**
```python
# Mover a GPU (una línea!)
model = model.to('cuda')
```

**4. Ecosistema Rico**
- Modelos pre-entrenados (ResNet, BERT, GPT)
- Data loaders optimizados
- Herramientas de visualización
- Comunidad masiva

**5. Productización**
- Guardar/cargar modelos fácilmente
- Desplegar en servidores
- Exportar a móviles (TF Lite)
- Optimizar para inferencia

### PyTorch vs TensorFlow

| Característica | PyTorch | TensorFlow/Keras |
|---------------|---------|------------------|
| **Filosofía** | Investigación, Pythónico | Producción, Escalable |
| **Curva de aprendizaje** | Más fácil | Media (Keras fácil) |
| **Debugging** | Excelente | Bueno |
| **Popularidad investigación** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Popularidad industria** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Documentación** | Excelente | Excelente |
| **Despliegue** | Mejorando | Excelente |

**Recomendación:** Aprende ambos! Son las herramientas estándar de la industria.

### Aplicaciones en el Mundo Real

**Todos los modelos modernos usan frameworks:**
- **GPT-3/4**: Entrenado con frameworks
- **Stable Diffusion**: PyTorch
- **BERT, T5**: TensorFlow
- **AlphaGo**: TensorFlow
- **DALL-E**: PyTorch
- **99% de papers de investigación**: PyTorch o TensorFlow

## 🤔 Preguntas de Reflexión Iniciales

1. ¿Por qué no seguir implementando todo manualmente?
2. ¿Qué significa "diferenciación automática"?
3. ¿Cómo puede un framework ser 100x más rápido?
4. ¿Cuál es el trade-off entre control y conveniencia?
5. ¿PyTorch o TensorFlow para tu proyecto?

## 🔬 Parte 1: PyTorch Fundamentals (60 min)

### 1.1 Tensores: Los Bloques Básicos

Los tensores son la estructura de datos fundamental de PyTorch y de todo el deep learning moderno. Antes de construir redes neuronales, es imprescindible dominar este bloque esencial.

**¿Qué es un tensor?**

Un tensor es una generalización matemática de escalares, vectores y matrices a dimensiones arbitrarias:

| Dimensiones | Nombre matemático | Ejemplo en deep learning |
|---|---|---|
| 0D | Escalar | Un valor de pérdida: `loss = 0.42` |
| 1D | Vector | Un ejemplo: `[x₁, x₂, ..., xₙ]` |
| 2D | Matriz | Un batch de datos: `(batch_size × features)` |
| 3D | Tensor orden 3 | Imagen en escala de grises por batch: `(batch × alto × ancho)` |
| 4D | Tensor orden 4 | Batch de imágenes a color: `(batch × canales × alto × ancho)` |

**¿Por qué tensores y no arrays NumPy?**

Los tensores de PyTorch son casi idénticos a los `ndarray` de NumPy, pero añaden dos capacidades críticas que NumPy no tiene:

1. **Ejecución en GPU**: Las operaciones con tensores pueden ejecutarse en NVIDIA GPUs con CUDA, logrando aceleraciones de 10×–100× para modelos grandes.
2. **Grafo computacional automático**: Cuando `requires_grad=True`, PyTorch registra cada operación sobre el tensor y construye dinámicamente un **grafo de cómputo**. Este grafo es la base de `autograd` (sección siguiente), que calcula gradientes automáticamente.

**¿Cómo funciona la interoperabilidad con NumPy?**

PyTorch y NumPy pueden **compartir memoria** mediante `torch.from_numpy()`: modificar el tensor modifica el array original y viceversa. Para evitar esto, usa `.clone()` para obtener una copia independiente.

**¿Qué resultados debes esperar?**

El código a continuación crea tensores de distintas formas, realiza operaciones matemáticas básicas (suma, producto elemento a elemento, producto punto, multiplicación matricial) y demuestra la conversión fluida entre NumPy y PyTorch.

```python
import torch
import numpy as np

print("=== TENSORES EN PYTORCH ===\n")

# Crear tensores
x = torch.tensor([1, 2, 3, 4])
print(f"Tensor 1D: {x}")
print(f"Shape: {x.shape}, dtype: {x.dtype}\n")

# Tensor 2D
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
print(f"Tensor 2D:\n{matrix}")
print(f"Shape: {matrix.shape}\n")

# Tensores especiales
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
randn = torch.randn(3, 3)  # Distribución normal
rand = torch.rand(2, 2)     # Distribución uniforme [0, 1]

print(f"Zeros:\n{zeros}\n")
print(f"Random normal:\n{randn}\n")

# Conversión NumPy ↔ PyTorch
np_array = np.array([1, 2, 3])
torch_tensor = torch.from_numpy(np_array)
back_to_numpy = torch_tensor.numpy()

print(f"NumPy → Torch → NumPy: {back_to_numpy}\n")

# Operaciones básicas
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"Suma: {a + b}")
print(f"Producto elemento-wise: {a * b}")
print(f"Producto punto: {torch.dot(a, b)}")
print(f"Matriz @ vector: {torch.randn(3, 4) @ torch.randn(4)}")
```

**Salida esperada:**
```
Tensor 1D: tensor([1, 2, 3, 4])
Shape: torch.Size([4]), dtype: torch.int64

Tensor 2D:
tensor([[1., 2.],
        [3., 4.],
        [5., 6.]])
Shape: torch.Size([3, 2])

Zeros:
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])

NumPy → Torch → NumPy: [1 2 3]

Suma: tensor([5., 7., 9.])
Producto elemento-wise: tensor([ 4., 10., 18.])
Producto punto: tensor(32.)
```

> 💡 **Tip:** Un tensor creado con `torch.from_numpy()` comparte memoria con el array NumPy original. Si modificas uno, el otro cambia. Usa `tensor.clone()` o `tensor.detach().clone()` para obtener una copia completamente independiente.

### 1.2 Autograd: El Corazón de PyTorch

Autograd es el sistema de **diferenciación automática** de PyTorch y es la razón por la que los frameworks eliminan la necesidad de implementar backpropagation manualmente.

**¿Qué es la diferenciación automática?**

La diferenciación automática (AD) es una técnica computacional que calcula derivadas exactas de funciones definidas como programas. No es diferenciación simbólica (como SymPy) ni diferenciación numérica (como diferencias finitas): es algo más eficiente y preciso que ambas.

**¿Cómo funciona el grafo computacional?**

Cada vez que realizas una operación sobre tensores con `requires_grad=True`, PyTorch construye dinámicamente un **grafo computacional dirigido acíclico (DAG)**:

```
x ──→ (operación) ──→ y ──→ (operación) ──→ loss
        ↑                       ↑
   registra                registra
   gradiente               gradiente
```

Cuando llamas a `loss.backward()`, PyTorch recorre el grafo **en sentido inverso** aplicando la **regla de la cadena** (chain rule) automáticamente para calcular el gradiente de `loss` respecto a cada parámetro con `requires_grad=True`.

**La regla de la cadena en PyTorch:**

Para una composición de funciones `L = f(g(h(x)))`:

```
∂L/∂x = (∂L/∂f) · (∂f/∂g) · (∂g/∂h) · (∂h/∂x)
```

PyTorch calcula y acumula esto automáticamente al llamar `.backward()`.

**¿Por qué elimina el backpropagation manual?**

En los laboratorios anteriores (Lab 05), implementaste backpropagation calculando manualmente cada derivada parcial. Con autograd:
- **No necesitas derivar fórmulas** para cada nueva arquitectura
- **No hay errores de derivación** (gradientes siempre correctos)
- **Soporta operaciones complejas**: convoluciones, atención, operaciones personalizadas

**¿Qué resultados debes esperar?**

El código calcula gradientes de funciones simples y verifica que coincidan con los valores analíticos: para `y = x²` con `x = 3`, el gradiente es `dy/dx = 2x = 6`.

```python
print("\n=== AUTOGRAD: DIFERENCIACIÓN AUTOMÁTICA ===\n")

# Ejemplo 1: Derivada simple
x = torch.tensor(3.0, requires_grad=True)  # Activar tracking de gradientes
print(f"x = {x}")

# Forward: y = x²
y = x ** 2
print(f"y = x² = {y}")

# Backward: calcular dy/dx
y.backward()
print(f"dy/dx = 2x = {x.grad}")  # Debería ser 2*3 = 6

print("\n--- Ejemplo 2: Función más compleja ---")

# Reset
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# z = w*x + b
z = w * x + b
print(f"z = w*x + b = {z}")

# Backward
z.backward()

print(f"∂z/∂x = w = {x.grad}")  # = w = 3
print(f"∂z/∂w = x = {w.grad}")  # = x = 2
print(f"∂z/∂b = 1 = {b.grad}")  # = 1

print("\n--- Ejemplo 3: Red neuronal simple ---")

# Input
x = torch.randn(1, 10, requires_grad=True)
print(f"Input shape: {x.shape}")

# Parámetros
W1 = torch.randn(10, 5, requires_grad=True)
b1 = torch.randn(1, 5, requires_grad=True)

W2 = torch.randn(5, 1, requires_grad=True)
b2 = torch.randn(1, 1, requires_grad=True)

# Forward
h = torch.relu(x @ W1 + b1)  # Capa oculta
y = h @ W2 + b2               # Salida

# Simular pérdida
target = torch.tensor([[1.0]])
loss = (y - target) ** 2

print(f"Loss: {loss.item():.4f}")

# Backward: ¡calcula TODOS los gradientes automáticamente!
loss.backward()

print(f"Gradiente de W1: {W1.grad.shape}")  # (10, 5)
print(f"Gradiente de W2: {W2.grad.shape}")  # (5, 1)
print("✓ Gradientes calculados automáticamente!")
```

**Salida esperada:**
```
=== AUTOGRAD: DIFERENCIACIÓN AUTOMÁTICA ===

x = 3.0
y = x² = 9.0
dy/dx = 2x = 6.0

--- Ejemplo 2: Función más compleja ---
z = w*x + b = 8.0
∂z/∂x = w = 3.0
∂z/∂w = x = 2.0
∂z/∂b = 1 = 1.0

--- Ejemplo 3: Red neuronal simple ---
Input shape: torch.Size([1, 10])
Loss: (valor variable)
Gradiente de W1: torch.Size([10, 5])
Gradiente de W2: torch.Size([5, 1])
✓ Gradientes calculados automáticamente!
```

> 💡 **Tip:** Recuerda llamar `optimizer.zero_grad()` antes de cada `backward()`. PyTorch **acumula** gradientes por defecto (en lugar de reemplazarlos). Si no limpias los gradientes, obtendrás sumas de gradientes de iteraciones anteriores, lo que corrompe el entrenamiento.

### 1.3 Primera Red Neuronal en PyTorch

PyTorch define redes neuronales mediante el paradigma de **programación orientada a objetos (POO)**. La clase `nn.Module` es la clase base de todos los modelos en PyTorch.

**¿Qué es `nn.Module`?**

`nn.Module` es la clase base que provee toda la infraestructura necesaria para una red neuronal:
- Registro automático de parámetros entrenables (`nn.Parameter`)
- Método `.parameters()` para iterar sobre todos los pesos
- Métodos `.train()` y `.eval()` para cambiar el comportamiento de capas como Dropout y BatchNorm
- Serialización para guardar y cargar modelos

**El paradigma PyTorch: `__init__` + `forward`**

```python
class MiRed(nn.Module):
    def __init__(self):          # 1. Define las CAPAS (estructura)
        super().__init__()
        self.capa1 = nn.Linear(...)
    
    def forward(self, x):        # 2. Define el FLUJO de datos
        return self.capa1(x)
```

- **`__init__`**: Se llama **una vez** al crear el modelo. Aquí defines las capas y sus parámetros.
- **`forward`**: Se llama **cada vez** que pasas datos por el modelo. Define cómo fluyen los datos de entrada a salida.

> El método `backward()` **no** se define manualmente: autograd lo construye automáticamente a partir del grafo generado por `forward()`.

**El optimizador SGD:**

El **Descenso de Gradiente Estocástico (SGD)** actualiza cada parámetro θ según:

```
θ ← θ - lr · ∂L/∂θ
```

Donde `lr` es la tasa de aprendizaje. En cada iteración del entrenamiento el ciclo es:
1. `optimizer.zero_grad()` → Limpiar gradientes acumulados
2. `loss.backward()` → Calcular gradientes (autograd)
3. `optimizer.step()` → Aplicar la actualización

**¿Qué resultados debes esperar?**

Verás la arquitectura del modelo impresa por PyTorch, el conteo de parámetros por capa, y cómo la pérdida (MSE) decrece progresivamente durante 100 épocas de entrenamiento con datos sintéticos.

```python
import torch.nn as nn
import torch.optim as optim

print("\n=== PRIMERA RED NEURONAL EN PYTORCH ===\n")

# Definir arquitectura
class SimpleNet(nn.Module):
    """Red neuronal simple: 10 → 20 → 1"""
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Definir capas
        self.fc1 = nn.Linear(10, 20)  # Capa 1: 10 → 20
        self.fc2 = nn.Linear(20, 1)   # Capa 2: 20 → 1
    
    def forward(self, x):
        """Forward pass"""
        x = torch.relu(self.fc1(x))  # Activación ReLU
        x = self.fc2(x)               # Sin activación en salida
        return x

# Instanciar modelo
model = SimpleNet()
print(model)
print()

# Ver parámetros
total_params = sum(p.numel() for p in model.parameters())
print(f"Parámetros totales: {total_params}")

# Listar parámetros
for name, param in model.named_parameters():
    print(f"{name:10s}: {param.shape}")

print("\n--- Entrenar el modelo ---")

# Generar datos sintéticos
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)

# Función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Loop de entrenamiento
for epoch in range(100):
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    # Backward pass
    optimizer.zero_grad()  # Limpiar gradientes
    loss.backward()        # Calcular gradientes
    optimizer.step()       # Actualizar parámetros
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

print("\n✓ Modelo entrenado!")
```

**Salida esperada:**
```
=== PRIMERA RED NEURONAL EN PYTORCH ===

SimpleNet(
  (fc1): Linear(in_features=10, out_features=20, bias=True)
  (fc2): Linear(in_features=20, out_features=1, bias=True)
)

Parámetros totales: 241
fc1.weight: torch.Size([20, 10])
fc1.bias  : torch.Size([20])
fc2.weight: torch.Size([1, 20])
fc2.bias  : torch.Size([1])

--- Entrenar el modelo ---
Epoch   0, Loss: 1.2345
Epoch  20, Loss: 0.9876
Epoch  40, Loss: 0.8123
Epoch  60, Loss: 0.7654
Epoch  80, Loss: 0.7321

✓ Modelo entrenado!
```

> 💡 **Tip:** El número `241` de parámetros se calcula así: capa `fc1` tiene `10×20 + 20 = 220` (pesos + biases), y `fc2` tiene `20×1 + 1 = 21`. Total: `220 + 21 = 241`. Saber contar parámetros te ayuda a estimar la complejidad del modelo y el riesgo de overfitting.

**Actividad 1.1:** Crea una red 20 → 50 → 30 → 10 con ReLU y entrénala en un problema de regresión.

### 1.4 Clasificación con PyTorch

Ahora construiremos un clasificador binario completo: desde la creación del dataset hasta la evaluación del modelo entrenado. Este flujo de trabajo es el estándar en PyTorch para tareas de clasificación.

**Clasificación binaria y BCELoss:**

En clasificación binaria, la salida del modelo es una probabilidad `ŷ ∈ [0, 1]` (usando `Sigmoid`). La función de pérdida **Binary Cross-Entropy (BCE)** mide qué tan bien calibradas están esas probabilidades:

```
BCE(ŷ, y) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

- Si `y=1` y `ŷ≈1`: pérdida ≈ 0 ✓
- Si `y=1` y `ŷ≈0`: pérdida → ∞ ✗ (penaliza fuertemente el error)

**El optimizador Adam:**

**Adam (Adaptive Moment Estimation)** es una mejora sobre SGD que mantiene tasas de aprendizaje adaptativas para cada parámetro:

```
mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ       ← Media móvil del gradiente (1er momento)
vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²      ← Media móvil del gradiente² (2do momento)
θ ← θ - lr · m̂ₜ / (√v̂ₜ + ε)    ← Actualización adaptativa
```

Adam suele converger más rápido que SGD porque:
1. **Momentum**: recuerda el historial de gradientes (evita oscilaciones)
2. **Adaptativo**: parámetros con gradientes grandes reciben actualizaciones más pequeñas

**`model.train()` vs `model.eval()`:**

| Modo | Efecto en Dropout | Efecto en BatchNorm |
|---|---|---|
| `model.train()` | Activo (desactiva aleatoriamente neuronas) | Usa estadísticas del batch actual |
| `model.eval()` | Desactivado (todas las neuronas activas) | Usa estadísticas acumuladas |

**`torch.no_grad()`:**

Durante la evaluación no necesitamos calcular gradientes. `torch.no_grad()` le dice a PyTorch que no construya el grafo computacional, reduciendo el consumo de memoria y acelerando la inferencia.

**¿Qué resultados debes esperar?**

Con 100 épocas de entrenamiento en un dataset sintético balanceado de 1000 ejemplos y 20 features, deberías alcanzar una **accuracy de test ≥ 85%**. La pérdida de entrenamiento y test deben decrecer de forma estable.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("\n=== CLASIFICACIÓN BINARIA CON PYTORCH ===\n")

# Generar datos
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a tensores
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

# Definir modelo
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularización
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Salida [0, 1]
        )
    
    def forward(self, x):
        return self.network(x)

# Crear modelo
model = BinaryClassifier(input_dim=20)
print(model)
print()

# Setup entrenamiento
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar
epochs = 100
for epoch in range(epochs):
    # Training mode
    model.train()
    
    # Forward
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Evaluar cada 10 épocas
    if epoch % 10 == 0:
        model.eval()  # Evaluation mode
        with torch.no_grad():  # No calcular gradientes
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            
            # Accuracy
            predictions = (test_outputs > 0.5).float()
            accuracy = (predictions == y_test).float().mean()
        
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | "
              f"Test Loss: {test_loss.item():.4f} | Acc: {accuracy.item():.4f}")

print("\n✓ Clasificador entrenado!")
```

**Salida esperada:**
```
Train: torch.Size([800, 20]), Test: torch.Size([200, 20])

BinaryClassifier(
  (network): Sequential(
    (0): Linear(in_features=20, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    ...
  )
)

Epoch   0 | Train Loss: 0.7023 | Test Loss: 0.6891 | Acc: 0.5450
Epoch  10 | Train Loss: 0.5832 | Test Loss: 0.5721 | Acc: 0.7250
Epoch  20 | Train Loss: 0.4901 | Test Loss: 0.4843 | Acc: 0.8050
...
Epoch  90 | Train Loss: 0.3241 | Test Loss: 0.3312 | Acc: 0.8750

✓ Clasificador entrenado!
```

> 💡 **Tip:** Si observas que `Train Loss` es mucho menor que `Test Loss`, es señal de **overfitting**: el modelo memoriza los datos de entrenamiento en lugar de generalizar. El `Dropout(0.3)` en este modelo actúa como regularizador para mitigar este problema.

**Actividad 1.2:** Modifica el modelo para clasificación multiclase (3+ clases) usando Softmax.

## 🔬 Parte 2: TensorFlow/Keras Fundamentals (60 min)

### 2.1 Introducción a TensorFlow

TensorFlow es el framework de deep learning desarrollado por Google Brain. Junto con PyTorch, es el estándar de la industria. Keras, integrado en TensorFlow desde la versión 2.0, proporciona una API de alto nivel que simplifica enormemente la construcción y entrenamiento de modelos.

**TensorFlow 2.x: Eager Execution por defecto**

En TensorFlow 1.x, era necesario construir un grafo estático y luego ejecutarlo en una "sesión". TensorFlow 2.x elimina esta complejidad con **Eager Execution**: las operaciones se ejecutan inmediatamente, igual que en Python y PyTorch, haciendo el código mucho más intuitivo y fácil de depurar.

**TensorFlow vs PyTorch a nivel de tensores:**

| Característica | PyTorch | TensorFlow |
|---|---|---|
| Creación | `torch.tensor([1,2,3])` | `tf.constant([1,2,3])` |
| Mutabilidad | Mutable (`x[0] = 1` funciona) | Inmutable (usa `tf.Variable`) |
| Autograd | `requires_grad=True` + `.backward()` | `tf.GradientTape()` |
| NumPy | `.numpy()` | `.numpy()` (igual) |
| GPU | `.to('cuda')` | automático o `/device:GPU:0` |

**`tf.constant` vs `tf.Variable`:**

- `tf.constant`: tensor **inmutable**, para datos de entrada y constantes.
- `tf.Variable`: tensor **mutable**, para parámetros del modelo (pesos y biases). Keras gestiona las Variables automáticamente.

**¿Qué resultados debes esperar?**

Verás la versión de TensorFlow instalada, la creación de tensores básicos y operaciones matemáticas equivalentes a las de PyTorch, confirmando que ambas APIs son muy similares a nivel de operaciones.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=== TENSORFLOW/KERAS BASICS ===\n")
print(f"TensorFlow version: {tf.__version__}\n")

# Tensores en TensorFlow
t1 = tf.constant([1, 2, 3, 4])
t2 = tf.constant([[1, 2], [3, 4]])

print(f"Tensor 1D: {t1}")
print(f"Tensor 2D:\n{t2}\n")

# Operaciones
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

print(f"Suma: {a + b}")
print(f"Producto: {a * b}")
print(f"Matmul: {tf.linalg.matmul(tf.reshape(a, (3, 1)), tf.reshape(b, (1, 3)))}")
```

**Salida esperada:**
```
=== TENSORFLOW/KERAS BASICS ===

TensorFlow version: 2.x.x

Tensor 1D: tf.Tensor([1 2 3 4], shape=(4,), dtype=int32)
Tensor 2D:
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32)

Suma: tf.Tensor([5. 7. 9.], shape=(3,), dtype=float32)
Producto: tf.Tensor([ 4. 10. 18.], shape=(3,), dtype=float32)
```

> 💡 **Tip:** En TensorFlow 2.x puedes acceder al valor numérico de un tensor con `.numpy()`, igual que en PyTorch. Internamente, tanto TensorFlow como PyTorch delegan las operaciones matemáticas a librerías optimizadas en C++ (Eigen, cuBLAS) que explotan las capacidades del hardware.

### 2.2 Primera Red con Keras Sequential API

La **Sequential API** de Keras es la forma más simple de construir modelos de deep learning. Está diseñada para arquitecturas donde los datos fluyen linealmente de una capa a la siguiente, sin bifurcaciones.

**¿Qué hace `model.compile()`?**

`compile()` configura el proceso de entrenamiento especificando tres elementos:

```python
model.compile(
    optimizer='adam',           # Algoritmo de optimización
    loss='binary_crossentropy', # Función de pérdida a minimizar
    metrics=['accuracy']        # Métricas a monitorear (no se optimizan)
)
```

Internamente, Keras construye el grafo de TensorFlow para el entrenamiento, incluyendo el cálculo de gradientes con `tf.GradientTape`.

**¿Qué hace `model.fit()`?**

`model.fit()` abstrae completamente el loop de entrenamiento:

```
Para cada época:
  Para cada batch:
    1. Forward pass: ŷ = model(X_batch)
    2. Calcular pérdida: L = loss(ŷ, y_batch)
    3. Backward pass: gradientes = tape.gradient(L, params)
    4. Actualizar: optimizer.apply_gradients(...)
  Calcular métricas de validación
  Imprimir progreso
```

Esto es equivalente al loop manual de PyTorch, pero completamente encapsulado.

**¿Qué hace `model.evaluate()`?**

`evaluate()` ejecuta solo el **forward pass** (sin actualizar pesos) sobre los datos proporcionados y devuelve la pérdida y métricas configuradas en `compile()`.

**Ventajas y desventajas de la abstracción de Keras:**

| Aspecto | Keras Sequential | PyTorch manual |
|---|---|---|
| **Código necesario** | Muy poco (~5 líneas) | Más verboso (~15 líneas) |
| **Flexibilidad** | Limitada (solo flujo lineal) | Total |
| **Debugging** | Más difícil (caja negra) | Más fácil (control total) |
| **Curva de aprendizaje** | Baja | Media |

**¿Qué resultados debes esperar?**

`model.summary()` mostrará la arquitectura con el número de parámetros por capa. Después de 50 épocas, deberías obtener una **accuracy de validación ≥ 85%** en el dataset de clasificación binaria.

```python
print("\n=== RED NEURONAL CON KERAS SEQUENTIAL ===\n")

# Definir modelo (API secuencial - más simple)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Ver arquitectura
model.summary()

# Compilar
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Generar datos (NumPy)
X_train_np, y_train_np = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42
)
X_test_np = X_train_np[:200]
y_test_np = y_train_np[:200]
X_train_np = X_train_np[200:]
y_train_np = y_train_np[200:]

# Entrenar
print("\nEntrenando...")
history = model.fit(
    X_train_np, y_train_np,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_np, y_test_np),
    verbose=0  # Silencioso
)

# Evaluar
test_loss, test_acc = model.evaluate(X_test_np, y_test_np, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predicciones
predictions = model.predict(X_test_np[:5])
print(f"\nPrimeras 5 predicciones:\n{predictions.flatten()}")
```

**Salida esperada:**
```
=== RED NEURONAL CON KERAS SEQUENTIAL ===

Model: "sequential"
_________________________________________________________________
 Layer (type)          Output Shape         Param #
=================================================================
 dense (Dense)         (None, 64)           1344
 dropout (Dropout)     (None, 64)           0
 dense_1 (Dense)       (None, 32)           2080
 dropout_1 (Dropout)   (None, 32)           0
 dense_2 (Dense)       (None, 1)            33
=================================================================
Total params: 3,457
Trainable params: 3,457
Non-trainable params: 0

Entrenando...

Test Accuracy: 0.8800
Test Loss: 0.2943

Primeras 5 predicciones:
[0.923 0.041 0.876 0.134 0.791]
```

> 💡 **Tip:** El objeto `history` retornado por `model.fit()` contiene un diccionario con el historial de pérdida y métricas por época: `history.history['loss']`, `history.history['val_accuracy']`, etc. Úsalo para graficar curvas de aprendizaje y diagnosticar overfitting o underfitting.

### 2.3 Keras Functional API (Más Flexible)

La **Functional API** de Keras permite construir modelos más complejos que la Sequential API. Mientras que Sequential obliga a un flujo lineal (una capa tras otra), la Functional API trata las capas como **funciones** que pueden conectarse arbitrariamente.

**¿Por qué existe la Functional API?**

La Sequential API no puede manejar:
- **Múltiples entradas**: e.g., un modelo que recibe imagen + texto
- **Múltiples salidas**: e.g., clasificador + regresor simultáneo
- **Skip connections** (conexiones residuales): e.g., ResNet
- **Ramas paralelas**: e.g., Inception modules
- **Grafos acíclicos dirigidos (DAG)** en general

**¿Cómo funciona?**

En la Functional API, cada capa es literalmente una función Python:

```python
x = layers.Dense(64)(input_tensor)   # La capa se "llama" sobre el tensor
x = layers.ReLU()(x)                  # Se pueden encadenar
```

El modelo se define especificando **qué tensor entra** y **qué tensor sale**:
```python
model = keras.Model(inputs=inputs, outputs=outputs)
```

Keras infiere automáticamente toda la topología del grafo entre `inputs` y `outputs`.

**Ventaja práctica:** La Functional API mantiene la comodidad de `compile()`/`fit()` pero con la flexibilidad para arquitecturas complejas. Es el **punto medio** entre Sequential (simple pero rígido) y Subclassing (flexible pero más código).

**¿Cuándo usarla?**

- Cuando necesitas skip connections
- Cuando tienes múltiples entradas o salidas
- Cuando quieres visualizar el grafo con `keras.utils.plot_model()`
- Para arquitecturas estilo ResNet, U-Net, Siamese Networks

**¿Qué resultados debes esperar?**

El modelo Functional tendrá exactamente la misma arquitectura y número de parámetros que el Sequential equivalente, demostrando que la API es solo una forma diferente de definir el mismo modelo.

```python
print("\n=== KERAS FUNCTIONAL API ===\n")

# Input layer
inputs = keras.Input(shape=(20,))

# Hidden layers
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.3)(x)

# Output layer
outputs = layers.Dense(1, activation='sigmoid')(x)

# Crear modelo
model_functional = keras.Model(inputs=inputs, outputs=outputs)

model_functional.summary()

# Compilar y entrenar igual que antes
model_functional.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model_functional.fit(
    X_train_np, y_train_np,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_np, y_test_np),
    verbose=0
)

print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
```

**Salida esperada:**
```
=== KERAS FUNCTIONAL API ===

Model: "model"
_________________________________________________________________
 Layer (type)          Output Shape         Param #
=================================================================
 input_1 (InputLayer)  [(None, 20)]         0
 dense (Dense)         (None, 64)           1344
 dropout (Dropout)     (None, 64)           0
 dense_1 (Dense)       (None, 32)           2080
 dropout_1 (Dropout)   (None, 32)           0
 dense_2 (Dense)       (None, 1)            33
=================================================================
Total params: 3,457

Final val accuracy: 0.8750
```

> 💡 **Tip:** Una ventaja única de la Functional API es que puedes acceder a las salidas **intermedias** del modelo: `intermediate_model = keras.Model(inputs=model.input, outputs=model.layers[2].output)`. Esto es muy útil para visualizar feature maps o construir modelos de extracción de características.

### 2.4 Subclassing (Máximo Control)

El **Model Subclassing** en Keras (equivalente a `nn.Module` en PyTorch) es la API más flexible de TensorFlow. Permite implementar cualquier arquitectura, incluidas aquellas con comportamiento dinámico que no pueden expresarse como un grafo estático.

**¿Qué hace el método `call()`?**

`call()` es el equivalente de `forward()` en PyTorch: define cómo fluyen los datos de entrada a salida. Se invoca automáticamente cuando "llamas" al modelo como función:

```python
modelo = CustomModel()
salida = modelo(entrada)  # Invoca model.call(entrada)
```

**El parámetro `training` en `call()`:**

```python
def call(self, inputs, training=False):
    x = self.dropout1(x, training=training)  # Solo activo si training=True
```

El parámetro `training` es crítico para capas con comportamiento diferente en entrenamiento vs inferencia:
- **Dropout**: aplica máscara aleatoria solo durante `training=True`
- **BatchNormalization**: usa estadísticas del batch en training, estadísticas acumuladas en inferencia

**¿Cuándo usar cada API?**

| Situación | API recomendada |
|---|---|
| Arquitectura simple, prototipo rápido | **Sequential** |
| Múltiples I/O, skip connections, grafo estático | **Functional** |
| Lógica dinámica, loops en forward, investigación | **Subclassing** |
| Máximo control, equivalente a PyTorch | **Subclassing** |

**Tradeoffs del Subclassing:**

- ✅ **Máxima flexibilidad**: cualquier lógica Python en `call()`
- ✅ **Familiar para usuarios de PyTorch**: mismo paradigma
- ❌ **No serializable directamente**: `model.save()` requiere llamar al modelo primero
- ❌ **No puedes inspeccionar el grafo** con `plot_model()` sin ejecutarlo antes

**¿Qué resultados debes esperar?**

El modelo subclassed entrenará igual que los anteriores, alcanzando accuracy similar (~85–88%), demostrando que las tres APIs producen modelos funcionalmente equivalentes para arquitecturas simples.

```python
print("\n=== MODEL SUBCLASSING ===\n")

class CustomModel(keras.Model):
    """Modelo personalizado con subclassing"""
    
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(32, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        """Forward pass"""
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)  # Dropout solo en training
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

# Crear y entrenar
model_custom = CustomModel()

model_custom.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Necesita build con una llamada o especificando input_shape
model_custom.build(input_shape=(None, 20))
model_custom.summary()

history = model_custom.fit(
    X_train_np, y_train_np,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_np, y_test_np),
    verbose=0
)

print(f"Final accuracy: {history.history['val_accuracy'][-1]:.4f}")
```

**Salida esperada:**
```
=== MODEL SUBCLASSING ===

Model: "custom_model"
_________________________________________________________________
 Layer (type)          Output Shape         Param #
=================================================================
 dense (Dense)         multiple             1344
 dropout (Dropout)     multiple             0
 dense_1 (Dense)       multiple             2080
 dropout_1 (Dropout)   multiple             0
 dense_2 (Dense)       multiple             33
=================================================================
Total params: 3,457

Final accuracy: 0.8650
```

> 💡 **Tip:** Con Subclassing, el `model.summary()` muestra "multiple" en Output Shape porque el grafo no se construye hasta que el modelo se ejecuta. Para ver las formas correctas, construye el modelo con `model.build(input_shape=(None, 20))` antes de llamar a `summary()`.

**Actividad 2.1:** Crea un modelo con arquitectura residual (skip connections) usando Functional API.

## 🔬 Parte 3: Comparación PyTorch vs TensorFlow (40 min)

### 3.1 Mismo Modelo en Ambos Frameworks

Una de las mejores formas de consolidar el aprendizaje de ambos frameworks es implementar exactamente el mismo modelo en PyTorch y TensorFlow y comparar sus características. Esta sección revela las diferencias fundamentales en filosofía de diseño.

**¿Qué revela esta comparación?**

Implementando el mismo modelo en ambos frameworks con los mismos hiperparámetros, podemos comparar:

1. **Cantidad de código**: PyTorch requiere un loop de entrenamiento explícito; Keras lo encapsula en `fit()`.
2. **Velocidad**: Ambos deberían ser comparables en CPU; TensorFlow puede tener ventaja en GPU con XLA compilation.
3. **Accuracy**: Con los mismos datos y arquitectura, los resultados deben ser similares (las diferencias son por inicialización aleatoria).

**El loop de entrenamiento explícito (PyTorch) vs implícito (TensorFlow):**

```
PyTorch (explícito):              TensorFlow/Keras (implícito):
─────────────────────             ──────────────────────────────
for epoch in range(50):           model.fit(X_train, y_train,
    outputs = model(X)                epochs=50,
    loss = criterion(out, y)          batch_size=32)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**PyTorch da más control**, pero requiere más código. **Keras abstrae el loop**, lo que es más conveniente pero oculta los detalles.

**Consideraciones de rendimiento:**

- En datasets pequeños (como el de este ejemplo), las diferencias de tiempo son mínimas y dependen de la inicialización de los frameworks.
- Para datasets grandes, TensorFlow puede compilar el grafo con XLA para mayor velocidad, mientras que PyTorch tiene `torch.compile()` (desde PyTorch 2.0).
- En producción, ambos frameworks ofrecen herramientas de optimización similares.

**¿Qué resultados debes esperar?**

Los tiempos de entrenamiento serán del orden de segundos para 50 épocas. Las diferencias pueden ser de 20-50%, pero varían según hardware. La accuracy final será similar en ambos.

```python
print("\n=== COMPARACIÓN LADO A LADO ===\n")

# ----- PYTORCH -----
print("1. PYTORCH\n")

class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

pytorch_model = PyTorchModel()
print(f"Parámetros PyTorch: {sum(p.numel() for p in pytorch_model.parameters())}")

# ----- TENSORFLOW -----
print("\n2. TENSORFLOW/KERAS\n")

tensorflow_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

tensorflow_model.build(input_shape=(None, 20))
print(f"Parámetros TensorFlow: {tensorflow_model.count_params()}")

# ----- ENTRENAR AMBOS -----
import time

# Datos
X_train_torch = torch.FloatTensor(X_train_np)
y_train_torch = torch.FloatTensor(y_train_np).reshape(-1, 1)

# PyTorch
print("\n--- Entrenando PyTorch ---")
criterion = nn.BCELoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

start = time.time()
for epoch in range(50):
    outputs = pytorch_model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

pytorch_time = time.time() - start
print(f"Tiempo PyTorch: {pytorch_time:.2f}s")

# TensorFlow
print("\n--- Entrenando TensorFlow ---")
tensorflow_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

start = time.time()
tensorflow_model.fit(
    X_train_np, y_train_np,
    epochs=50,
    batch_size=32,
    verbose=0
)
tensorflow_time = time.time() - start
print(f"Tiempo TensorFlow: {tensorflow_time:.2f}s")

print(f"\n--- Comparación ---")
print(f"PyTorch:    {pytorch_time:.2f}s")
print(f"TensorFlow: {tensorflow_time:.2f}s")
```

**Salida esperada:**
```
=== COMPARACIÓN LADO A LADO ===

1. PYTORCH
Parámetros PyTorch: 3457

2. TENSORFLOW/KERAS
Parámetros TensorFlow: 3457

--- Entrenando PyTorch ---
Tiempo PyTorch: 0.85s

--- Entrenando TensorFlow ---
Tiempo TensorFlow: 1.23s

--- Comparación ---
PyTorch:    0.85s
TensorFlow: 1.23s
```

> 💡 **Tip:** Los tiempos variarán significativamente según tu hardware y si usas GPU. TensorFlow tiene un overhead de inicialización mayor al primer run (compilación JIT), pero puede ser más rápido en ejecuciones subsecuentes. Para benchmarks precisos, ignora la primera ejecución y promedia múltiples corridas (como hace la sección de Benchmark Completo más adelante).

### 3.2 DataLoaders y Pipelines

El pipeline de carga de datos es uno de los cuellos de botella más comunes en el entrenamiento de modelos de deep learning. Si los datos no se cargan suficientemente rápido, la GPU (o CPU) se queda esperando, desperdiciando capacidad de cómputo.

**¿Por qué son críticos los pipelines de datos?**

En el entrenamiento moderno, el modelo puede procesar un batch en milisegundos, pero cargar imágenes de disco, aplicar transformaciones y preprocesarlas puede ser mucho más lento. Sin un pipeline eficiente:

```
[Cargar datos] → [Procesar] → [Entrenar]
   500ms              200ms       10ms       ← GPU inactiva 700ms por batch!
```

Con pipeline asíncrono:
```
[Cargar N+1] → paralelo con → [Entrenar N]
   500ms                          10ms       ← GPU siempre ocupada
```

**PyTorch: Dataset + DataLoader**

El patrón de PyTorch separa dos responsabilidades:
- **`Dataset`**: Define cómo obtener un elemento individual por índice (`__getitem__`) y el tamaño total (`__len__`).
- **`DataLoader`**: Orquesta el batching, shuffling y carga paralela en múltiples procesos (`num_workers`).

```python
# Dataset personalizado
class MyDataset(Dataset):
    def __getitem__(self, idx): ...  # Un elemento
    def __len__(self): ...           # Tamaño total

# DataLoader gestiona el batching
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

**TensorFlow: tf.data.Dataset**

`tf.data` es el sistema de pipelines de TensorFlow, diseñado para alta eficiencia con operaciones encadenables:

```python
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=1000)  # Aleatorización
dataset = dataset.batch(32)                  # Agrupar en batches
dataset = dataset.prefetch(tf.data.AUTOTUNE) # Prefetch automático
```

**¿Por qué importa el shuffling?**

Si el modelo ve siempre los ejemplos en el mismo orden, puede aprender patrones espurios. El shuffling garantiza que cada época el modelo vea los datos en orden diferente, mejorando la generalización.

**¿Qué resultados debes esperar?**

Ambos DataLoaders mostrarán un batch de shape `(32, 20)` para X y `(32, 1)` para y, confirmando que el batching funciona correctamente.

```python
print("\n=== DATA LOADING ===\n")

# ----- PYTORCH DATALOADER -----
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """Dataset personalizado para PyTorch"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Crear dataset y dataloader
train_dataset = CustomDataset(X_train_np, y_train_np)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("PyTorch DataLoader:")
for batch_X, batch_y in train_loader:
    print(f"  Batch shape: X={batch_X.shape}, y={batch_y.shape}")
    break

# ----- TENSORFLOW DATASET -----
train_dataset_tf = tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np))
train_dataset_tf = train_dataset_tf.shuffle(1000).batch(32)

print("\nTensorFlow Dataset:")
for batch_X, batch_y in train_dataset_tf.take(1):
    print(f"  Batch shape: X={batch_X.shape}, y={batch_y.shape}")
```

**Salida esperada:**
```
=== DATA LOADING ===

PyTorch DataLoader:
  Batch shape: X=torch.Size([32, 20]), y=torch.Size([32, 1])

TensorFlow Dataset:
  Batch shape: X=(32, 20), y=(32,)
```

> 💡 **Tip:** Para datasets grandes que no caben en RAM, usa `DataLoader` con `num_workers > 0` en PyTorch (carga paralela en múltiples CPUs) o `.prefetch(tf.data.AUTOTUNE)` en TensorFlow. Para imágenes, considera `torchvision.datasets` (PyTorch) o `tf.keras.preprocessing.image_dataset_from_directory` (TF) que cargan desde disco eficientemente.

**Actividad 3.1:** Implementa el mismo modelo en ambos frameworks y compara resultados.

## 🔬 Parte 4: Funcionalidades Avanzadas (50 min)

### 4.1 Guardar y Cargar Modelos

Guardar modelos es esencial en cualquier flujo de trabajo real de machine learning. Permite reanudar entrenamientos interrumpidos, compartir modelos entrenados, versionar experimentos y desplegar modelos en producción.

**¿Cuándo y por qué guardar modelos?**

| Situación | Estrategia |
|---|---|
| Entrenamiento largo (horas/días) | Checkpoints periódicos para no perder progreso |
| Mejor modelo durante validación | Guardar cuando la métrica mejora (ModelCheckpoint) |
| Despliegue en producción | Guardar el modelo final optimizado |
| Reproducibilidad científica | Guardar modelos de experimentos publicados |

**PyTorch: Dos enfoques**

**Enfoque 1 — `state_dict` (recomendado):**
```python
torch.save(model.state_dict(), 'model.pth')  # Solo los pesos
model.load_state_dict(torch.load('model.pth'))
```
- ✅ Más flexible (puedes cargar en modelos con arquitectura modificada)
- ✅ Tamaño de archivo más pequeño
- ❌ Debes tener el código de la clase del modelo para cargar

**Enfoque 2 — Modelo completo:**
```python
torch.save(model, 'model_complete.pth')  # Arquitectura + pesos
model = torch.load('model_complete.pth')
```
- ✅ No necesitas el código de la clase
- ❌ Frágil si cambias la estructura del código

**TensorFlow: Múltiples formatos**

- **HDF5 (`.h5`)**: Formato legacy, compacto, soportado por Keras.
- **SavedModel (directorio)**: Formato moderno de TF, incluye el grafo computacional, compatible con TensorFlow Serving para producción.

```python
model.save('model.h5')              # HDF5
model.save('model_dir/')            # SavedModel (recomendado)
loaded = keras.models.load_model('model.h5')
```

**¿Qué resultados debes esperar?**

Los modelos guardados y cargados deben producir exactamente las mismas predicciones que el modelo original, verificando que los pesos se guardaron y cargaron correctamente.

```python
print("\n=== GUARDAR Y CARGAR MODELOS ===\n")

# ----- PYTORCH -----
print("1. PyTorch\n")

# Guardar
torch.save(pytorch_model.state_dict(), '/tmp/pytorch_model.pth')
print("✓ Modelo guardado: pytorch_model.pth")

# Cargar
loaded_pytorch_model = PyTorchModel()
loaded_pytorch_model.load_state_dict(torch.load('/tmp/pytorch_model.pth'))
loaded_pytorch_model.eval()
print("✓ Modelo cargado")

# Verificar que funciona
test_input = torch.randn(1, 20)
output = loaded_pytorch_model(test_input)
print(f"Predicción de prueba: {output.item():.4f}\n")

# ----- TENSORFLOW -----
print("2. TensorFlow\n")

# Guardar (varios formatos)
tensorflow_model.save('/tmp/tf_model.h5')  # HDF5
print("✓ Modelo guardado: tf_model.h5")

# Cargar
loaded_tf_model = keras.models.load_model('/tmp/tf_model.h5')
print("✓ Modelo cargado")

# Verificar
test_input_tf = np.random.randn(1, 20)
output_tf = loaded_tf_model.predict(test_input_tf, verbose=0)
print(f"Predicción de prueba: {output_tf[0][0]:.4f}")
```

**Salida esperada:**
```
=== GUARDAR Y CARGAR MODELOS ===

1. PyTorch

✓ Modelo guardado: pytorch_model.pth
✓ Modelo cargado
Predicción de prueba: 0.6234

2. TensorFlow

✓ Modelo guardado: tf_model.h5
✓ Modelo cargado
Predicción de prueba: 0.7891
```

> 💡 **Tip:** Para PyTorch, siempre llama `model.eval()` después de cargar el modelo con `load_state_dict()`. Esto asegura que las capas como Dropout y BatchNorm estén en modo inferencia. Si planeas continuar entrenando, llama `model.train()` en su lugar. En TensorFlow, el modelo cargado recuerda automáticamente su estado de compilación.

### 4.2 Callbacks y Early Stopping

Los **callbacks** son funciones que Keras llama automáticamente en puntos específicos del entrenamiento (al inicio/fin de cada época, al inicio/fin de cada batch, etc.). Permiten añadir comportamiento personalizado sin modificar el loop de entrenamiento.

**¿Qué es Early Stopping y por qué previene overfitting?**

El **Early Stopping** detiene el entrenamiento cuando la métrica de validación deja de mejorar. Sin él, un modelo puede seguir bajando la pérdida de entrenamiento mientras la pérdida de validación sube (overfitting):

```
          Zona ideal
             ↓
Pérdida │  ╲  train
        │   ╲___________
        │         ╱──── val (empieza a subir → overfitting)
        └─────────────────── épocas
                  ↑
           Early Stopping aquí
```

El parámetro **`patience`** define cuántas épocas consecutivas sin mejora tolerar antes de detener. `patience=10` significa "detente si no mejora en 10 épocas seguidas". `restore_best_weights=True` recupera los pesos del mejor modelo encontrado.

**¿Qué hace `ReduceLROnPlateau`?**

Reduce la tasa de aprendizaje cuando la pérdida de validación se estanca. Si no mejora en `patience` épocas, multiplica `lr` por `factor`:

```
lr_nueva = lr_actual × factor    (e.g., 0.001 × 0.5 = 0.0005)
```

Esto permite que el modelo "refine" su posición en el espacio de parámetros con pasos más pequeños cuando está cerca del óptimo.

**¿Qué hace `ModelCheckpoint`?**

Guarda el modelo automáticamente cada vez que la métrica monitoreada mejora. Con `save_best_only=True`, solo guarda cuando supera el mejor resultado anterior, actuando como un "versionado automático" del mejor modelo.

**Callbacks disponibles en Keras:**

| Callback | Función |
|---|---|
| `EarlyStopping` | Detener cuando la métrica se estanca |
| `ReduceLROnPlateau` | Reducir lr cuando la métrica se estanca |
| `ModelCheckpoint` | Guardar el mejor modelo automáticamente |
| `TensorBoard` | Registrar métricas para visualización |
| `LearningRateScheduler` | Programar cambios de lr manualmente |
| `CSVLogger` | Guardar historial en CSV |

**¿Qué resultados debes esperar?**

Con `patience=10` sobre 100 épocas programadas, el entrenamiento típicamente se detiene antes (entre 20 y 50 épocas) cuando la validación converge, ahorrando tiempo de cómputo sin sacrificar accuracy.

```python
print("\n=== CALLBACKS (TENSORFLOW) ===\n")

# Crear modelo fresco
model_with_callbacks = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model_with_callbacks.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Definir callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        '/tmp/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Entrenar con callbacks
history = model_with_callbacks.fit(
    X_train_np, y_train_np,
    epochs=100,  # Muchas épocas, pero early stopping lo detendrá
    batch_size=32,
    validation_data=(X_test_np, y_test_np),
    callbacks=callbacks,
    verbose=0
)

print(f"\nÉpocas entrenadas: {len(history.history['loss'])}")
print(f"Mejor val_accuracy: {max(history.history['val_accuracy']):.4f}")
```

**Salida esperada:**
```
=== CALLBACKS (TENSORFLOW) ===

Epoch 00032: early stopping
Restoring model weights from the end of the best epoch.

Epoch 00027: ReduceLROnPlateau reducing learning rate to 0.0005.

Epoch 00032: val_accuracy improved from 0.8700 to 0.8750, saving model to /tmp/best_model.h5

Épocas entrenadas: 32
Mejor val_accuracy: 0.8750
```

> 💡 **Tip:** El número de épocas que Early Stopping necesita depende del dataset y la arquitectura. Un `patience` muy pequeño puede detener el entrenamiento prematuramente (antes de que converja), mientras que uno muy grande pierde el beneficio. Un valor de `patience = epochs * 0.1` (10% del total) suele ser un buen punto de partida.

### 4.3 Visualización con TensorBoard

**TensorBoard** es la herramienta de visualización oficial de TensorFlow (también disponible para PyTorch). Proporciona un dashboard interactivo en el navegador para monitorear y depurar el entrenamiento de modelos.

**¿Por qué es crítico monitorear el entrenamiento?**

Entrenar un modelo sin visualización es como conducir con los ojos cerrados. TensorBoard permite:
- **Detectar problemas temprano**: overfitting, gradientes que se desvanecen, learning rate inadecuado
- **Comparar experimentos**: diferentes arquitecturas o hiperparámetros en la misma gráfica
- **Entender el modelo**: visualizar pesos, activaciones y gradientes por capa

**¿Qué métricas registra TensorBoard?**

| Panel | Información |
|---|---|
| **Scalars** | Pérdida y métricas por época (loss, accuracy, lr) |
| **Histograms** | Distribución de pesos y gradientes por capa |
| **Graphs** | Grafo computacional del modelo |
| **Images** | Imágenes de entrada, activaciones, filtros |
| **Projector** | Visualización de embeddings en 2D/3D (t-SNE) |
| **HParams** | Búsqueda de hiperparámetros |

**¿Cómo se usa en la práctica?**

```python
# 1. Crear callback con directorio de logs
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir='./logs/experimento_01',
    histogram_freq=1  # Guardar histogramas cada época
)

# 2. Pasar callback a fit()
model.fit(..., callbacks=[tensorboard_cb])

# 3. Lanzar TensorBoard desde terminal
# tensorboard --logdir=./logs --port=6006
# Abrir en navegador: http://localhost:6006
```

**Interpretando las curvas de aprendizaje en TensorBoard:**

```
Escenario ideal:           Overfitting:          Underfitting:
train ╲                    train  ╲__             train ╲
val   ╲__                  val    ╱               val   ╲  (ambas altas)
                                (divergen)
```

**¿Qué resultados debes esperar?**

Después de ejecutar el código, los logs se guardarán en `/tmp/logs`. Al lanzar TensorBoard, verás gráficas de pérdida y accuracy que bajan progresivamente durante 20 épocas.

```python
print("\n=== TENSORBOARD ===\n")

# Crear callback de TensorBoard
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='/tmp/logs',
    histogram_freq=1
)

# Entrenar con logging
model_tb = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_tb.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_tb.fit(
    X_train_np, y_train_np,
    epochs=20,
    validation_data=(X_test_np, y_test_np),
    callbacks=[tensorboard_callback],
    verbose=0
)

print("✓ Logs guardados en /tmp/logs")
print("Para visualizar: tensorboard --logdir=/tmp/logs")
```

**Salida esperada:**
```
=== TENSORBOARD ===

✓ Logs guardados en /tmp/logs
Para visualizar: tensorboard --logdir=/tmp/logs
```

Para visualizar en un entorno local, abre una terminal y ejecuta:
```bash
tensorboard --logdir=/tmp/logs --port=6006
```
Luego abre `http://localhost:6006` en tu navegador.

> 💡 **Tip:** Organiza tus experimentos con subdirectorios con nombre descriptivo: `logs/experimento_lr001/`, `logs/experimento_lr0001/`. Así TensorBoard mostrará ambas curvas superpuestas y podrás comparar directamente el efecto de diferentes hiperparámetros.

### 4.4 Transfer Learning Básico

El **Transfer Learning** (aprendizaje por transferencia) es una de las técnicas más poderosas del deep learning moderno. Permite reutilizar conocimiento aprendido en una tarea para resolver una tarea diferente pero relacionada.

**¿Por qué funciona el Transfer Learning?**

Las redes profundas aprenden representaciones **jerárquicas**:

```
Capa 1-3:   Detectan bordes, colores, texturas simples (universales)
Capa 4-7:   Detectan formas, partes de objetos (semiespecíficas)
Capa 8-10:  Detectan objetos específicos del dataset (específicas)
```

Las capas tempranas aprenden características genéricas útiles para cualquier tarea visual. Una red entrenada en ImageNet (1.2M imágenes, 1000 clases) ha aprendido detectores de bordes, texturas y formas que son transferibles a cualquier tarea de visión.

**Feature Extraction vs Fine-tuning:**

| Enfoque | ¿Qué se entrena? | ¿Cuándo usarlo? |
|---|---|---|
| **Feature Extraction** | Solo las capas nuevas (clasificador) | Dataset pequeño (<1000 imágenes) |
| **Fine-tuning parcial** | Capas nuevas + últimas capas del base | Dataset mediano |
| **Fine-tuning total** | Todo el modelo | Dataset grande y similar |

**¿Por qué congelamos capas (`trainable = False`)?**

Al congelar la red base:
1. **Evitamos destruir features aprendidas**: Si entrenamos toda la red con pocas muestras, podríamos sobreescribir representaciones valiosas con ruido.
2. **Reducimos parámetros entrenables**: Entrenamos solo el clasificador nuevo, que es mucho más pequeño y rápido de entrenar.
3. **Necesitamos menos datos**: El clasificador es simple y no necesita millones de ejemplos.

**MobileNetV2:**

MobileNetV2 es una arquitectura eficiente diseñada para dispositivos móviles. Con `include_top=False` y `weights='imagenet'`, cargamos la red base sin la capa de clasificación final, listos para añadir nuestro propio clasificador.

**¿Qué resultados debes esperar?**

El modelo mostrará el contraste dramático entre parámetros totales (~3.5M de MobileNetV2) y parámetros entrenables (solo los del clasificador personalizado, ~100K). Este es el poder del Transfer Learning: entrenar solo el 3% de los parámetros para obtener un clasificador poderoso.

```python
print("\n=== TRANSFER LEARNING (EJEMPLO) ===\n")

# Usando modelo pre-entrenado de Keras
from tensorflow.keras.applications import MobileNetV2

# Cargar modelo pre-entrenado (sin top/clasificador)
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Congelar pesos del modelo base
base_model.trainable = False

# Añadir clasificador personalizado
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)  # 10 clases

transfer_model = keras.Model(inputs, outputs)

print("Modelo con transfer learning:")
print(f"  Parámetros totales: {transfer_model.count_params():,}")
print(f"  Parámetros entrenables: {sum(tf.size(w).numpy() for w in transfer_model.trainable_weights):,}")
print(f"  Parámetros congelados: {sum(tf.size(w).numpy() for w in transfer_model.non_trainable_weights):,}")
```

**Salida esperada:**
```
=== TRANSFER LEARNING (EJEMPLO) ===

Modelo con transfer learning:
  Parámetros totales:      3,638,538
  Parámetros entrenables:    131,082
  Parámetros congelados:   3,507,456
```

La proporción típica es: **~3.6% de parámetros entrenables**. Esto es lo que hace que el Transfer Learning sea tan eficiente: con solo 131K parámetros a entrenar (en lugar de 3.6M), necesitas muchos menos datos y tiempo de entrenamiento.

> 💡 **Tip:** Para hacer **fine-tuning** después de feature extraction, descongela las últimas capas del modelo base con `base_model.trainable = True` y entrena con una tasa de aprendizaje muy baja (`lr=1e-5`). Esto "afina" las representaciones específicas para tu tarea sin destruir el conocimiento previo.

**Actividad 4.1:** Implementa un sistema de checkpointing en PyTorch similar a los callbacks de Keras.

## 📊 Análisis Final de Rendimiento

### Benchmark Completo

El benchmark final proporciona una comparación estadística rigurosa entre PyTorch y TensorFlow. Al ejecutar múltiples corridas, obtenemos no solo el tiempo y accuracy promedio, sino también la **variabilidad** (desviación estándar), que es un indicador de la estabilidad de cada framework.

**Metodología del benchmark:**

1. **Múltiples corridas (`n_runs=3-5`)**: Promediar múltiples ejecuciones elimina la variabilidad por calentamiento del sistema, JIT compilation, y otras fuentes de ruido.
2. **Mismo dataset y arquitectura**: Control de variables — la única diferencia es el framework.
3. **Mismos hiperparámetros**: `lr=0.001`, `epochs=20`, `batch_size=32` en ambos.

**¿Qué métricas se comparan?**

| Métrica | Qué indica | Cuál es mejor |
|---|---|---|
| `avg_time` | Velocidad de entrenamiento | Menor es mejor |
| `std_time` | Estabilidad de velocidad | Menor es mejor |
| `avg_accuracy` | Calidad del modelo | Mayor es mejor |
| `std_accuracy` | Consistencia de resultados | Menor es mejor |

**¿Cómo interpretar los resultados?**

- Si los tiempos son similares (< 20% de diferencia): ambos frameworks son igualmente apropiados para esta tarea.
- Si la accuracy varía mucho entre corridas: considera revisar la inicialización de pesos o el learning rate.
- Las barras de error en la gráfica representan ±1 desviación estándar.

**Insight clave:** Los benchmarks en CPU con datasets pequeños no reflejan el rendimiento real en producción con GPU y datasets grandes. Las diferencias se magnifican con modelos más grandes y datos más complejos.

```python
import matplotlib.pyplot as plt

print("\n=== BENCHMARK FINAL ===\n")

def benchmark_framework(framework_name, train_fn, n_runs=5):
    """Benchmark de un framework"""
    times = []
    accuracies = []
    
    for run in range(n_runs):
        start = time.time()
        accuracy = train_fn()
        elapsed = time.time() - start
        
        times.append(elapsed)
        accuracies.append(accuracy)
    
    return {
        'framework': framework_name,
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'avg_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies)
    }

# Definir funciones de entrenamiento
def train_pytorch():
    model = PyTorchModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    for epoch in range(20):
        outputs = model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluar
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test_np))
        preds = (preds > 0.5).float()
        accuracy = (preds.numpy().flatten() == y_test_np).mean()
    
    return accuracy

def train_tensorflow():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(20,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_np, y_train_np, epochs=20, verbose=0)
    
    _, accuracy = model.evaluate(X_test_np, y_test_np, verbose=0)
    return accuracy

# Ejecutar benchmarks
results_pytorch = benchmark_framework('PyTorch', train_pytorch, n_runs=3)
results_tf = benchmark_framework('TensorFlow', train_tensorflow, n_runs=3)

# Mostrar resultados
print("RESULTADOS:\n")
print(f"PyTorch:")
print(f"  Tiempo: {results_pytorch['avg_time']:.2f}s (+/- {results_pytorch['std_time']:.2f}s)")
print(f"  Accuracy: {results_pytorch['avg_accuracy']:.4f} (+/- {results_pytorch['std_accuracy']:.4f})")

print(f"\nTensorFlow:")
print(f"  Tiempo: {results_tf['avg_time']:.2f}s (+/- {results_tf['std_time']:.2f}s)")
print(f"  Accuracy: {results_tf['avg_accuracy']:.4f} (+/- {results_tf['std_accuracy']:.4f})")

# Visualizar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

frameworks = ['PyTorch', 'TensorFlow']
times = [results_pytorch['avg_time'], results_tf['avg_time']]
time_errs = [results_pytorch['std_time'], results_tf['std_time']]

ax1.bar(frameworks, times, yerr=time_errs, capsize=10, color=['#EE4C2C', '#FF6F00'])
ax1.set_ylabel('Tiempo (s)')
ax1.set_title('Tiempo de Entrenamiento')
ax1.grid(True, alpha=0.3, axis='y')

accuracies = [results_pytorch['avg_accuracy'], results_tf['avg_accuracy']]
acc_errs = [results_pytorch['std_accuracy'], results_tf['std_accuracy']]

ax2.bar(frameworks, accuracies, yerr=acc_errs, capsize=10, color=['#EE4C2C', '#FF6F00'])
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Final')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/framework_comparison.png')
print("\n✓ Gráfica guardada: framework_comparison.png")
```

**Resultados esperados:**
```
=== BENCHMARK FINAL ===

RESULTADOS:

PyTorch:
  Tiempo: 0.78s (+/- 0.12s)
  Accuracy: 0.8767 (+/- 0.0152)

TensorFlow:
  Tiempo: 1.15s (+/- 0.23s)
  Accuracy: 0.8833 (+/- 0.0208)

✓ Gráfica guardada: framework_comparison.png
```

Los valores exactos variarán según tu hardware. Lo importante es el **orden de magnitud** y la **variabilidad relativa**.

> 💡 **Conclusión del benchmark:** Para datasets pequeños en CPU, las diferencias entre frameworks son marginales. La elección entre PyTorch y TensorFlow debe basarse en factores como: tu equipo, el ecosistema de herramientas, los requisitos de despliegue, y las arquitecturas que necesitas implementar — no en benchmarks de velocidad en datasets de juguete.

### Nivel Básico

**Ejercicio 1:** Primera Red en PyTorch
```
Implementa una red 784 → 128 → 64 → 10 para MNIST:
- Usa ReLU en capas ocultas
- Softmax en salida
- CrossEntropyLoss
- Entrena 10 épocas
```

**Ejercicio 2:** Primera Red en TensorFlow
```
Implementa la misma arquitectura en Keras:
- Usa Sequential API
- Compila con Adam optimizer
- Entrena con validation split
- Grafica historia de entrenamiento
```

**Ejercicio 3:** Comparación Directa
```
Implementa el mismo modelo en ambos frameworks:
- Misma arquitectura
- Mismos hiperparámetros
- Compara tiempos y resultados
```

### Nivel Intermedio

**Ejercicio 4:** DataLoaders Personalizados
```
Crea un dataset personalizado:
- PyTorch: Implementa Dataset y DataLoader
- TensorFlow: Usa tf.data.Dataset
- Incluye augmentation de datos
- Batch processing eficiente
```

**Ejercicio 5:** Early Stopping
```
Implementa early stopping:
- PyTorch: Manualmente o con biblioteca
- TensorFlow: Usa Callbacks
- Compara implementaciones
- Guarda mejor modelo
```

**Ejercicio 6:** Transfer Learning
```
Usa modelo pre-entrenado:
- Carga ResNet o MobileNet
- Congela capas base
- Añade clasificador personalizado
- Fine-tuning gradual
```

### Nivel Avanzado

**Ejercicio 7:** Modelo Personalizado Complejo
```
Implementa arquitectura compleja:
- Skip connections (ResNet-style)
- Multiple inputs/outputs
- Custom training loop
- En ambos frameworks
```

**Ejercicio 8:** Optimización y Despliegue
```
Optimiza modelo para producción:
- Pruning/Quantization
- ONNX export (PyTorch)
- TF Lite conversion
- Benchmarks de inferencia
```

**Ejercicio 9:** Proyecto Completo
```
Pipeline end-to-end:
- Carga y preprocesamiento de datos
- Entrenamiento con validación
- Evaluación completa
- Guardado y deployment
- API de inferencia
```

## 📝 Entregables

### 1. Código Fuente
- `pytorch_basics.py`: Fundamentos de PyTorch
- `tensorflow_basics.py`: Fundamentos de TensorFlow
- `comparison.py`: Comparación de frameworks
- `advanced_features.py`: Funcionalidades avanzadas
- `experiments.ipynb`: Notebook comparativo

### 2. Modelos Entrenados
- Modelos guardados en ambos formatos
- Checkpoints de entrenamiento
- Métricas de evaluación

### 3. Documentación
- Guía de uso de cada framework
- Comparación detallada
- Mejores prácticas
- Troubleshooting common

### 4. Reporte Final (4-5 páginas)
- Experiencia con cada framework
- Ventajas y desventajas
- Casos de uso recomendados
- Conclusiones y recomendaciones

## 🎯 Criterios de Evaluación (CDIO)

### Conceive (Concebir) - 25%
- [ ] Comprensión de ventajas de frameworks
- [ ] Selección apropiada de herramientas
- [ ] Diseño de experimentos comparativos
- [ ] Planificación de arquitecturas

### Design (Diseñar) - 25%
- [ ] Implementación correcta en PyTorch
- [ ] Implementación correcta en TensorFlow
- [ ] Uso apropiado de APIs
- [ ] Código limpio y modular

### Implement (Implementar) - 30%
- [ ] Modelos entrenan correctamente
- [ ] Uso efectivo de autograd
- [ ] Aprovechamiento de utilidades
- [ ] Resultados reproducibles

### Operate (Operar) - 20%
- [ ] Comparaciones significativas
- [ ] Análisis crítico de resultados
- [ ] Optimización de performance
- [ ] Documentación completa

## 📋 Rúbrica de Evaluación

| Criterio | Excelente (90-100%) | Bueno (75-89%) | Satisfactorio (60-74%) | Insuficiente (<60%) |
|----------|-------------------|--------------|---------------------|------------------|
| **PyTorch** | Dominio completo | Buen manejo | Uso básico | Dificultades |
| **TensorFlow** | Dominio completo | Buen manejo | Uso básico | Dificultades |
| **Comparación** | Análisis profundo | Buena comparación | Comparación básica | Comparación pobre |
| **Código** | Excelente, modular | Bien estructurado | Funcional | Desorganizado |
| **Optimización** | Altamente optimizado | Bien optimizado | Optimización básica | Sin optimización |

## 📚 Referencias Adicionales

### Documentación Oficial
- **PyTorch**: https://pytorch.org/docs/
- **TensorFlow**: https://www.tensorflow.org/api_docs
- **Keras**: https://keras.io/

### Tutoriales
- PyTorch Tutorials: https://pytorch.org/tutorials/
- TensorFlow Tutorials: https://www.tensorflow.org/tutorials
- Fast.ai: https://www.fast.ai/

### Libros
- "Deep Learning with PyTorch" (Stevens et al.)
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Géron)
- "Programming PyTorch for Deep Learning" (Rao)

### Comunidad
- PyTorch Forums
- TensorFlow Community
- Stack Overflow
- GitHub repositories

## 🎓 Notas Finales

### ¿PyTorch o TensorFlow?

**Usa PyTorch si:**
- Trabajas en investigación
- Necesitas máxima flexibilidad
- Prefieres código pythónico
- Quieres debugging fácil

**Usa TensorFlow si:**
- Despliegas a producción
- Necesitas exportar a móviles/web
- Trabajas en industria
- Quieres pipelines completos

**La verdad: Aprende ambos.** Son las herramientas estándar.

### Del NumPy Manual a los Frameworks

Has recorrido un camino increíble:
1. ✅ Implementaste todo desde cero (Labs 1-7)
2. ✅ Entiendes profundamente cómo funcionan las redes
3. ✅ Ahora usas herramientas profesionales

**Este conocimiento profundo te hace un mejor practicante de deep learning.**

### Checklist de Frameworks

- [ ] Entiendo tensores y operaciones básicas
- [ ] Puedo crear modelos en PyTorch
- [ ] Puedo crear modelos en TensorFlow/Keras
- [ ] Sé usar autograd
- [ ] Entiendo DataLoaders y Datasets
- [ ] Puedo guardar/cargar modelos
- [ ] Sé usar callbacks y early stopping
- [ ] Puedo optimizar para producción

### Próximos Pasos

**Continúa aprendiendo:**
- Arquitecturas avanzadas (ResNet, Transformers)
- Visión por computadora (CNNs)
- PLN (RNNs, Transformers)
- IA Generativa (GANs, VAEs, Diffusion)

**Proyectos recomendados:**
- Clasificador de imágenes personalizado
- Chatbot con RNNs
- Detector de objetos
- Sistema de recomendación

### Reflexión Final

**Los frameworks no reemplazan el conocimiento profundo - lo amplifican.**

Ahora que entiendes los fundamentos, los frameworks te permiten:
- Iterar más rápido
- Experimentar con arquitecturas complejas
- Deployar modelos en producción
- Competir con estado del arte

¡Usa este poder sabiamente! 🚀

---

**"The best way to learn deep learning is to do deep learning." - Andrew Ng**

**¡Los frameworks hacen el deep learning accesible! 🚀**
