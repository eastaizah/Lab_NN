# Teoría: Entrenamiento de Redes Neuronales

## Introducción

El **entrenamiento** es el proceso completo de ajustar los parámetros de una red neuronal para que aprenda patrones de los datos. Combina todo lo que hemos aprendido: forward pass, funciones de pérdida, backpropagation y optimización.

## El Loop de Entrenamiento

### Estructura Básica

```python
for epoch in range(num_epochs):
    for batch in data:
        # 1. Forward pass
        predictions = model(batch)
        
        # 2. Calcular pérdida
        loss = loss_function(predictions, targets)
        
        # 3. Backward pass
        gradients = backpropagation(loss)
        
        # 4. Actualizar parámetros
        optimizer.step(gradients)
```

## Conceptos Fundamentales

### 1. Épocas (Epochs)

**Definición**: Una época es un pase completo a través de todo el conjunto de datos de entrenamiento.

```
1 epoch = procesar todos los datos una vez
```

**Típicamente**: Se entrenan redes por 10-1000+ épocas

**Por qué múltiples épocas**:
- Una época raramente es suficiente
- El modelo necesita "ver" los datos múltiples veces
- Aprendizaje gradual e iterativo

### 2. Batches (Lotes)

**Definición**: Dividir los datos en grupos pequeños para procesamiento.

```
Total de datos: 1000 muestras
Batch size: 32
Número de batches: 1000 / 32 = 31.25 ≈ 32 batches
```

**Ventajas de batches**:
- **Eficiencia**: Operaciones matriciales más rápidas
- **Memoria**: No carga todos los datos a la vez
- **Regularización**: Añade ruido que ayuda a generalizar
- **Paralelización**: Uso eficiente de GPUs

**Tamaños comunes**: 16, 32, 64, 128, 256

### 3. Iteraciones

**Definición**: Una iteración es un paso de actualización de parámetros (procesar un batch).

```
Iteraciones por época = número_de_muestras / batch_size
```

**Ejemplo**:
```
Datos: 10,000 muestras
Batch size: 100
Épocas: 50

Iteraciones por época = 10,000 / 100 = 100
Total de iteraciones = 100 * 50 = 5,000
```

### 4. Learning Rate (Tasa de Aprendizaje)

**Definición**: Controla qué tan grande es cada paso de actualización.

```
θ_new = θ_old - learning_rate * gradient
```

**Efectos**:

```
LR muy pequeño (0.0001):
  + Aprendizaje estable
  - Muy lento
  - Puede quedar atrapado

LR óptimo (0.001-0.01):
  + Balance ideal
  + Convergencia razonable
  
LR muy grande (1.0):
  + Rápido inicialmente
  - Inestable
  - Puede divergir
```

**Estrategias**:

1. **Constante**: Mismo LR todo el tiempo
2. **Decay**: Reducir LR gradualmente
   ```python
   lr = lr_initial * 0.95^epoch
   ```
3. **Step decay**: Reducir en intervalos
   ```python
   if epoch % 10 == 0:
       lr = lr * 0.5
   ```
4. **Learning rate scheduling**: Basado en pérdida

### 5. Conjuntos de Datos

**Training Set (Entrenamiento)**:
- Datos para entrenar el modelo
- El modelo los "ve" y aprende
- Típicamente: 60-80% de los datos

**Validation Set (Validación)**:
- Datos para ajustar hiperparámetros
- Evaluar durante el entrenamiento
- Detectar overfitting
- Típicamente: 10-20% de los datos

**Test Set (Prueba)**:
- Datos que el modelo NUNCA ha visto
- Evaluación final del modelo
- Típicamente: 10-20% de los datos

**División típica**: 70% train, 15% validation, 15% test

```
Total: 10,000 muestras
Train: 7,000
Validation: 1,500
Test: 1,500
```

## Proceso de Entrenamiento Completo

### Paso 1: Preparar Datos

```python
# 1. Cargar datos
X, y = load_data()

# 2. Normalizar
X = (X - X.mean()) / X.std()

# 3. Dividir
X_train, X_val, X_test = split_data(X, y)

# 4. Crear batches
train_batches = create_batches(X_train, y_train, batch_size=32)
```

### Paso 2: Inicializar Modelo

```python
model = NeuralNetwork(
    input_size=784,
    hidden_sizes=[128, 64],
    output_size=10
)
```

### Paso 3: Loop de Entrenamiento

```python
for epoch in range(num_epochs):
    # Modo entrenamiento
    model.train()
    
    # Iterar sobre batches
    for X_batch, y_batch in train_batches:
        # Forward
        predictions = model(X_batch)
        loss = loss_function(predictions, y_batch)
        
        # Backward
        gradients = backprop(loss)
        
        # Update
        optimizer.step(gradients)
    
    # Validación
    model.eval()
    val_loss = evaluate(model, val_data)
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
```

## Monitoreo del Entrenamiento

### Métricas Importantes

1. **Pérdida de entrenamiento**: Debe disminuir
2. **Pérdida de validación**: Debe disminuir (si sube → overfitting)
3. **Accuracy**: Para clasificación
4. **Tiempo por época**: Para estimar duración total

### Curvas de Aprendizaje

**Caso ideal**:
```
Loss  |  train \_____
      |  val   \______
      +----------------> Epochs
```

**Overfitting**:
```
Loss  |  train \_____
      |  val   \__/‾‾‾
      +----------------> Epochs
```

**Underfitting**:
```
Loss  |  train \_______ alto
      |  val   \_______ alto
      +----------------> Epochs
```

## Técnicas de Regularización

### 1. Early Stopping

**Concepto**: Detener el entrenamiento cuando la validación deja de mejorar.

```python
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(max_epochs):
    train()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()
        counter = 0
    else:
        counter += 1
    
    if counter >= patience:
        print("Early stopping!")
        break
```

### 2. Dropout

**Concepto**: Aleatoriamente "apagar" neuronas durante entrenamiento.

```python
def dropout(x, rate=0.5, training=True):
    if not training:
        return x
    
    mask = np.random.binomial(1, 1-rate, size=x.shape) / (1-rate)
    return x * mask
```

**Por qué funciona**: Previene co-adaptación de neuronas

### 3. Weight Decay (L2 Regularization)

**Concepto**: Penalizar pesos grandes.

```
Loss_total = Loss_data + λ * Σ(w²)
```

### 4. Data Augmentation

**Concepto**: Aumentar datos artificialmente.

Para imágenes:
- Rotaciones
- Flips
- Zoom
- Ruido

## Inicialización de Pesos

**Importante**: La inicialización afecta significativamente el entrenamiento.

### Malas Inicializaciones

**Todos ceros**:
```python
W = np.zeros((n_in, n_out))  # ✗ MAL
```
Problema: Todas las neuronas aprenden lo mismo

**Muy grandes**:
```python
W = np.random.randn(n_in, n_out) * 10  # ✗ MAL
```
Problema: Activaciones saturan, gradientes desaparecen

### Buenas Inicializaciones

**Xavier/Glorot** (para Sigmoid/Tanh):
```python
W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)
```

**He** (para ReLU):
```python
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

## Optimizadores Avanzados

### 1. Momentum

**Idea**: Mantener velocidad de iteraciones anteriores.

```python
v = 0.9 * v - learning_rate * gradient
θ = θ + v
```

**Ventaja**: Acelera en direcciones consistentes

### 2. Adam (Adaptive Moment Estimation)

**Idea**: Combina momentum y adaptive learning rates.

```python
m = β1 * m + (1 - β1) * gradient  # Primer momento
v = β2 * v + (1 - β2) * gradient²  # Segundo momento
θ = θ - lr * m / (sqrt(v) + ε)
```

**Hiperparámetros típicos**:
- β1 = 0.9
- β2 = 0.999
- lr = 0.001

**Ventajas**: Generalmente funciona bien sin ajustar mucho

## Debugging del Entrenamiento

### Síntomas y Soluciones

**Pérdida no baja**:
- Verificar implementación (gradient checking)
- Aumentar learning rate
- Verificar que datos estén normalizados
- Revisar arquitectura

**Pérdida es NaN**:
- Learning rate muy grande → reducir
- Overflow numérico → verificar softmax/sigmoid estables
- División por cero → añadir epsilon

**Overfitting rápido**:
- Añadir regularización (dropout, L2)
- Más datos de entrenamiento
- Modelo más simple
- Early stopping

**Convergencia muy lenta**:
- Aumentar learning rate
- Mejor inicialización
- Normalizar datos
- Usar optimizador mejor (Adam)

## Checklist de Entrenamiento

Antes de entrenar:
- [ ] Datos normalizados
- [ ] División train/val/test correcta
- [ ] Arquitectura apropiada para el problema
- [ ] Inicialización correcta de pesos
- [ ] Funciones de activación y pérdida adecuadas

Durante entrenamiento:
- [ ] Monitorear train y val loss
- [ ] Verificar que train loss disminuye
- [ ] Detectar overfitting temprano
- [ ] Guardar mejores modelos

Después de entrenar:
- [ ] Evaluar en test set
- [ ] Analizar errores
- [ ] Visualizar predicciones
- [ ] Documentar hiperparámetros

## Hiperparámetros Típicos

Para empezar, prueba:
```
Learning rate: 0.001
Batch size: 32
Epochs: 100
Optimizer: Adam
Hidden layers: 2
Hidden size: 64-128
Activation: ReLU
Output activation: Softmax (clasificación) o None (regresión)
```

## Próximos Pasos

En el siguiente laboratorio exploraremos **frameworks de deep learning** (PyTorch, TensorFlow) que automatizan mucho de esto.
