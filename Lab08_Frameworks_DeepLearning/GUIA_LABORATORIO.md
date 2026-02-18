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

### 1.2 Autograd: El Corazón de PyTorch

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

### 1.3 Primera Red Neuronal en PyTorch

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

**Actividad 1.1:** Crea una red 20 → 50 → 30 → 10 con ReLU y entrénala en un problema de regresión.

### 1.4 Clasificación con PyTorch

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

**Actividad 1.2:** Modifica el modelo para clasificación multiclase (3+ clases) usando Softmax.

## 🔬 Parte 2: TensorFlow/Keras Fundamentals (60 min)

### 2.1 Introducción a TensorFlow

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

### 2.2 Primera Red con Keras Sequential API

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

### 2.3 Keras Functional API (Más Flexible)

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

### 2.4 Subclassing (Máximo Control)

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

**Actividad 2.1:** Crea un modelo con arquitectura residual (skip connections) usando Functional API.

## 🔬 Parte 3: Comparación PyTorch vs TensorFlow (40 min)

### 3.1 Mismo Modelo en Ambos Frameworks

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

### 3.2 DataLoaders y Pipelines

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

**Actividad 3.1:** Implementa el mismo modelo en ambos frameworks y compara resultados.

## 🔬 Parte 4: Funcionalidades Avanzadas (50 min)

### 4.1 Guardar y Cargar Modelos

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

### 4.2 Callbacks y Early Stopping

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

### 4.3 Visualización con TensorBoard

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

### 4.4 Transfer Learning Básico

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

**Actividad 4.1:** Implementa un sistema de checkpointing en PyTorch similar a los callbacks de Keras.

## 📊 Análisis Final de Rendimiento

### Benchmark Completo

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

## 🎯 EJERCICIOS PROPUESTOS

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
