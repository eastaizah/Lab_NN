# Teoría: Frameworks de Deep Learning

## Introducción

Hasta ahora hemos implementado redes neuronales **desde cero** para entender los fundamentos. Sin embargo, en la práctica usamos **frameworks** que automatizan y optimizan muchos aspectos del deep learning.

## ¿Por qué usar Frameworks?

### Ventajas

1. **Autograd (Diferenciación Automática)**
   - No necesitas implementar backpropagation manualmente
   - Calcula gradientes automáticamente
   
2. **Optimización de Performance**
   - Operaciones optimizadas en CPU/GPU
   - Paralelización automática
   - 10-100x más rápido que implementaciones naive

3. **Abstracciones útiles**
   - Capas predefinidas (Conv2D, LSTM, etc.)
   - Optimizadores (Adam, SGD, etc.)
   - Utilidades (data loaders, checkpoints, etc.)

4. **Soporte de GPU**
   - Entrenamiento en GPU con cambios mínimos
   - Multi-GPU automático

5. **Ecosistema**
   - Modelos pre-entrenados
   - Comunidad grande
   - Debugging tools

### Desventajas

- Curva de aprendizaje inicial
- Abstracciones pueden ocultar detalles
- Dependencias y compatibilidad

## Principales Frameworks

### 1. PyTorch

**Desarrollado por**: Facebook/Meta AI

**Filosofía**: Pythónico, dinámico, imperativo

**Características**:
- Grafos computacionales dinámicos
- Debugging fácil (código Python estándar)
- Muy popular en investigación
- Excelente documentación

**Ideal para**:
- Investigación
- Prototipos rápidos
- Modelos con flujo dinámico

**Ejemplo básico**:
```python
import torch
import torch.nn as nn

# Definir modelo
class MiRed(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instanciar
model = MiRed()

# Forward pass
x = torch.randn(32, 784)
output = model(x)

# Backward (automático!)
loss = criterion(output, target)
loss.backward()
```

### 2. TensorFlow/Keras

**Desarrollado por**: Google

**Filosofía**: Producción, escalabilidad, despliegue

**Características**:
- Grafos estáticos (TF 1.x) y dinámicos (TF 2.x)
- Keras como API de alto nivel
- TensorBoard para visualización
- Excelente para producción
- TF Lite para móviles

**Ideal para**:
- Producción
- Despliegue en móviles/web
- Pipelines completos

**Ejemplo básico**:
```python
import tensorflow as tf
from tensorflow import keras

# Definir modelo (Sequential API)
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```

### 3. JAX

**Desarrollado por**: Google Research

**Filosofía**: Composabilidad funcional, transformaciones

**Características**:
- Programación funcional
- JIT compilation
- Vectorización automática
- Muy rápido

**Ideal para**: Investigación avanzada, computación científica

## PyTorch vs TensorFlow

| Aspecto | PyTorch | TensorFlow |
|---------|---------|------------|
| **Paradigma** | Dinámico (eager) | Dinámico (TF 2.x) |
| **Sintaxis** | Pythónico | API múltiples (Keras) |
| **Debugging** | Más fácil | Más complejo |
| **Producción** | Mejorando | Excelente |
| **Comunidad** | Investigación | Industria |
| **Curva aprendizaje** | Más suave | Más empinada |
| **GPU** | Excelente | Excelente |

**Recomendación**: 
- **PyTorch** si estás aprendiendo o investigando
- **TensorFlow/Keras** si vas a producción

## Conceptos Comunes en Frameworks

### 1. Tensores

Generalización de arrays de NumPy con soporte GPU.

**PyTorch**:
```python
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
x = x.to('cuda')  # Mover a GPU
```

**TensorFlow**:
```python
x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
# TF maneja GPU automáticamente
```

### 2. Módulos/Capas

Building blocks de redes neuronales.

**PyTorch**:
```python
class MiCapa(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return torch.relu(self.linear(x))
```

**TensorFlow/Keras**:
```python
class MiCapa(keras.layers.Layer):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.out_features))
        self.b = self.add_weight(shape=(self.out_features,))
    
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
```

### 3. Autograd (Diferenciación Automática)

**PyTorch**:
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()  # Calcula dy/dx
print(x.grad)  # Gradiente: 2*2 + 3 = 7
```

**TensorFlow**:
```python
x = tf.Variable([2.0])
with tf.GradientTape() as tape:
    y = x ** 2 + 3 * x + 1

dy_dx = tape.gradient(y, x)  # Gradiente
```

### 4. Optimizadores

**PyTorch**:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    optimizer.zero_grad()  # Limpiar gradientes
    output = model(x)
    loss = criterion(output, y)
    loss.backward()  # Calcular gradientes
    optimizer.step()  # Actualizar parámetros
```

**TensorFlow/Keras**:
```python
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# En Keras es automático con model.fit()
model.compile(optimizer=optimizer, loss='mse')
model.fit(x, y, epochs=10)
```

### 5. Data Loaders

**PyTorch**:
```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    # Entrenar en batch
    pass
```

**TensorFlow**:
```python
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(1000).batch(32)

for batch_x, batch_y in dataset:
    # Entrenar en batch
    pass
```

## Flujo de Trabajo Típico

### En PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Definir modelo
class MiModelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# 2. Instanciar modelo, pérdida, optimizador
model = MiModelo()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Loop de entrenamiento
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        # Forward
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validación
    model.eval()
    with torch.no_grad():
        val_loss = ...
    model.train()
```

### En TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras

# 1. Definir modelo
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# 2. Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Entrenar (¡todo automático!)
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val)
)
```

## Transferencia de Conocimiento

Lo que aprendiste "desde cero" aplica directamente:

| Concepto | Desde Cero | En Framework |
|----------|-----------|--------------|
| **Forward pass** | Manual | `model(x)` |
| **Backprop** | Manual | `loss.backward()` |
| **Update** | `W -= lr * dW` | `optimizer.step()` |
| **Activaciones** | Funciones propias | `nn.ReLU()`, `tf.nn.relu` |
| **Pérdidas** | Funciones propias | `nn.CrossEntropyLoss()` |

## Mejores Prácticas

### 1. Organización de Código

```python
# Separar en módulos
models/
    __init__.py
    my_model.py
utils/
    data_loader.py
    metrics.py
train.py
evaluate.py
config.py
```

### 2. Configuración

```python
# Usar archivos de configuración
config = {
    'model': {
        'hidden_size': 128,
        'num_layers': 3
    },
    'training': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    }
}
```

### 3. Checkpoints

**PyTorch**:
```python
# Guardar
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Cargar
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

**TensorFlow**:
```python
# Guardar
model.save('my_model.h5')

# Cargar
model = keras.models.load_model('my_model.h5')
```

### 4. Logging y Visualización

**TensorBoard** (ambos frameworks):
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment1')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Accuracy/train', acc, epoch)
```

## Recursos de Aprendizaje

### PyTorch
- Documentación oficial: pytorch.org
- Tutoriales: pytorch.org/tutorials
- Curso: "PyTorch for Deep Learning" (fast.ai)

### TensorFlow
- Documentación oficial: tensorflow.org
- Guías: tensorflow.org/guide
- Curso: "Intro to TensorFlow for Deep Learning" (Udacity)

## Resumen

**Frameworks** simplifican enormemente el desarrollo:
- Autograd automático
- Optimizaciones de rendimiento
- Abstracciones útiles
- Soporte de GPU

**Pero** es crucial entender los fundamentos (lo que hemos aprendido) para:
- Debugging efectivo
- Arquitecturas personalizadas
- Optimizaciones específicas
- Comprender papers de investigación

## Próximo Paso

Implementaremos el mismo modelo en **PyTorch** y **TensorFlow** para ver las diferencias y similitudes.
