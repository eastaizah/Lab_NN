# Guía de Laboratorio: Funciones de Pérdida y Optimización

## 📋 Información del Laboratorio

**Título:** Funciones de Pérdida y Optimización  
**Código:** Lab 04  
**Duración:** 2-3 horas  
**Nivel:** Básico-Intermedio  

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender qué son funciones de pérdida y su propósito
2. Implementar MSE, MAE, Cross-Entropy desde cero
3. Elegir función de pérdida apropiada para cada problema
4. Implementar gradient descent básico
5. Comprender efecto del learning rate
6. Reconocer y detectar overfitting
7. Calcular derivadas de funciones de pérdida
8. Combinar pérdida con activación eficientemente
9. Aplicar regularización básica

## 📚 Prerrequisitos

### Conocimientos

- Completar Lab 01-03
- Python intermedio (clases, funciones, NumPy)
- Álgebra lineal básica
- Comprensión de conceptos de labs anteriores

### Software

- Python 3.8+
- NumPy 1.19+
- Matplotlib 3.0+
- Jupyter Notebook (recomendado)

### Material de Lectura

Antes de comenzar, lee:
- `teoria.md` - Marco teórico completo
- `README.md` - Visión general del laboratorio

## 📖 Introducción

Las **funciones de pérdida** cuantifican qué tan bien las predicciones del modelo se ajustan a los datos reales. Son el corazón del aprendizaje en redes neuronales.

### Contexto del Problema

Hasta ahora construimos redes y generamos predicciones, pero ¿cómo sabemos si son buenas? Necesitamos una métrica que:
- Cuantifique el error
- Guíe la optimización
- Permita comparar modelos

### Funciones de Pérdida

Miden discrepancia entre predicciones (ŷ) y valores reales (y):

```
L = f(y, ŷ)
Objetivo: L → 0 (minimizar pérdida)
```

### Conceptos Fundamentales

**1. Tipos principales:**
- **MSE:** Regresión (penaliza errores grandes)
- **MAE:** Regresión (robusta a outliers)
- **Binary Cross-Entropy:** Clasificación binaria
- **Categorical Cross-Entropy:** Clasificación multiclase

**2. Optimización:** Gradient descent ajusta parámetros para minimizar pérdida

**3. Learning Rate:** Controla tamaño del paso de actualización

### Aplicaciones

- Regresión: Predecir precios, temperaturas
- Clasificación: Spam detection, reconocimiento de imágenes
- Optimización: Entrenar cualquier modelo de ML

## 🔬 Parte 1: Funciones para Regresión (45 min)

### 1.1 Mean Squared Error (MSE)

```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_derivada(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

# Ejemplo
y_true = np.array([2, 4, 3])
y_pred = np.array([3, 5, 2])
print(f"MSE: {mse(y_true, y_pred)}")  # 1.0
```

### 1.2 Mean Absolute Error (MAE)

```python
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_derivada(y_true, y_pred):
    return np.sign(y_pred - y_true) / len(y_true)

# Ejemplo
print(f"MAE: {mae(y_true, y_pred)}")  # 1.0
```

### 1.3 Comparación con Outliers

```python
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.1, 3.1, 4.1, 100])  # último es outlier

print(f"MSE: {mse(y_true, y_pred):.2f}")  # Alto (penaliza mucho outlier)
print(f"MAE: {mae(y_true, y_pred):.2f}")  # Menor (más robusto)
```

### Actividades

1. Implementar Huber Loss
2. Comparar MSE vs MAE con diferentes datos
3. Visualizar curvas de pérdida

## 🔬 Parte 2: Funciones para Clasificación (45 min)

### 2.1 Binary Cross-Entropy

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # Evitar log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivada(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

# Ejemplo
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])
print(f"BCE: {binary_cross_entropy(y_true, y_pred):.4f}")
```

### 2.2 Categorical Cross-Entropy

```python
def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # y_true: one-hot encoded
    # y_pred: probabilidades de softmax
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Ejemplo
y_true = np.array([[0, 1, 0], [1, 0, 0]])  # One-hot
y_pred = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])  # Softmax
print(f"CCE: {categorical_cross_entropy(y_true, y_pred):.4f}")
```

### 2.3 Sparse Categorical Cross-Entropy

```python
def sparse_categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # y_true: índices de clase (no one-hot)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n_samples = y_pred.shape[0]
    log_probs = np.log(y_pred[range(n_samples), y_true])
    return -np.mean(log_probs)

# Ejemplo
y_true = np.array([1, 0])  # Índices
y_pred = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
print(f"Sparse CCE: {sparse_categorical_cross_entropy(y_true, y_pred):.4f}")
```

### Actividades

1. Implementar todas las pérdidas
2. Comparar BCE vs MSE para clasificación
3. Verificar derivadas numéricamente

## 🔬 Parte 3: Gradient Descent (45 min)

### 3.1 Implementación Básica

```python
def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    # Inicializar parámetros
    w = np.zeros(X.shape[1])
    b = 0
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = X @ w + b
        
        # Calcular pérdida
        loss = mse(y, y_pred)
        losses.append(loss)
        
        # Calcular gradientes
        dw = 2 * X.T @ (y_pred - y) / len(y)
        db = 2 * np.mean(y_pred - y)
        
        # Actualizar parámetros
        w -= learning_rate * dw
        b -= learning_rate * db
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return w, b, losses
```

### 3.2 Efecto del Learning Rate

```python
def comparar_learning_rates(X, y):
    lrs = [0.001, 0.01, 0.1, 1.0]
    
    plt.figure(figsize=(12, 4))
    
    for lr in lrs:
        _, _, losses = gradient_descent(X, y, learning_rate=lr, epochs=100)
        plt.plot(losses, label=f'LR={lr}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Efecto del Learning Rate')
    plt.grid(True)
    plt.savefig('learning_rates.png')
```

### 3.3 Variantes de Gradient Descent

```python
# Batch Gradient Descent (ya implementado arriba)

# Stochastic Gradient Descent
def sgd(X, y, learning_rate=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    
    for epoch in range(epochs):
        for i in range(len(X)):
            xi = X[i:i+1]
            yi = y[i:i+1]
            
            y_pred = xi @ w + b
            dw = 2 * xi.T @ (y_pred - yi)
            db = 2 * (y_pred - yi)
            
            w -= learning_rate * dw.flatten()
            b -= learning_rate * db[0]
    
    return w, b

# Mini-batch Gradient Descent
def mini_batch_gd(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, len(X), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            y_pred = X_batch @ w + b
            dw = 2 * X_batch.T @ (y_pred - y_batch) / len(X_batch)
            db = 2 * np.mean(y_pred - y_batch)
            
            w -= learning_rate * dw
            b -= learning_rate * db
    
    return w, b
```

### Actividades

1. Implementar gradient descent para clasificación
2. Comparar batch, SGD, mini-batch
3. Encontrar learning rate óptimo

## 🔬 Parte 4: Overfitting y Regularización (40 min)

### 4.1 Demostración de Overfitting

```python
def simular_overfitting():
    # Generar datos
    np.random.seed(42)
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = 2 * X + 1 + np.random.randn(20, 1) * 2
    
    # Split train/val
    X_train, X_val = X[:15], X[15:]
    y_train, y_val = y[:15], y[15:]
    
    # Entrenar con diferentes complejidades
    for degree in [1, 3, 10]:
        # Crear features polinomiales
        X_poly_train = np.column_stack([X_train**i for i in range(1, degree+1)])
        X_poly_val = np.column_stack([X_val**i for i in range(1, degree+1)])
        
        w, b, _ = gradient_descent(X_poly_train, y_train, epochs=1000)
        
        train_loss = mse(y_train, X_poly_train @ w + b)
        val_loss = mse(y_val, X_poly_val @ w + b)
        
        print(f"Degree {degree}: Train={train_loss:.2f}, Val={val_loss:.2f}")
```

### 4.2 L2 Regularization

```python
def gradient_descent_l2(X, y, lambda_reg=0.01, learning_rate=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    
    for epoch in range(epochs):
        y_pred = X @ w + b
        
        # Pérdida con regularización
        loss = mse(y, y_pred) + lambda_reg * np.sum(w**2)
        
        # Gradientes con regularización
        dw = 2 * X.T @ (y_pred - y) / len(y) + 2 * lambda_reg * w
        db = 2 * np.mean(y_pred - y)
        
        w -= learning_rate * dw
        b -= learning_rate * db
    
    return w, b
```

### 4.3 Early Stopping

```python
def train_with_early_stopping(X_train, y_train, X_val, y_val, patience=10):
    w = np.zeros(X_train.shape[1])
    b = 0
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_w, best_b = w.copy(), b
    
    for epoch in range(1000):
        # Entrenar
        y_pred = X_train @ w + b
        dw = 2 * X_train.T @ (y_pred - y_train) / len(y_train)
        db = 2 * np.mean(y_pred - y_train)
        w -= 0.01 * dw
        b -= 0.01 * db
        
        # Validar
        y_val_pred = X_val @ w + b
        val_loss = mse(y_val, y_val_pred)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_w, best_b = w.copy(), b
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping en epoch {epoch}")
            break
    
    return best_w, best_b
```

### Actividades

1. Demostrar overfitting con datos polinomiales
2. Aplicar L2 regularization
3. Implementar early stopping

## 📊 Análisis Final de Rendimiento

### Comparación de Implementaciones

En esta sección compararás diferentes enfoques de implementación para entender las ventajas de cada uno.

**Criterios de comparación:**
- Velocidad de ejecución
- Uso de memoria
- Claridad del código
- Mantenibilidad

### Métricas de Desempeño

Mide y compara:
- Tiempo de forward pass
- Escalabilidad con tamaño de datos
- Eficiencia computacional

## 🎯 EJERCICIOS PROPUESTOS

### Ejercicio 1: Implementar Huber Loss (Básico)

```python
Huber(y, ŷ) = 0.5(y-ŷ)² si |y-ŷ| ≤ δ
            = δ|y-ŷ| - 0.5δ² en otro caso
```

### Ejercicio 2: Learning Rate Scheduler (Intermedio)

Implementa:
```python
lr_new = lr_initial * decay_rate^epoch
```

Compara con LR fijo.

### Ejercicio 3: Early Stopping Mejorado (Intermedio)

- Monitorea pérdida de validación
- Guarda mejor modelo
- Restaura al finalizar

### Ejercicio 4: Comparación de Optimizadores (Avanzado)

Compara:
- Batch GD
- SGD
- Mini-batch GD
- Momentum (bonus)

### Ejercicio 5: Detección Automática de Overfitting (Proyecto)

Sistema que:
- Detecta divergencia train/val
- Recomienda λ de regularización
- Aplica early stopping automáticamente

## 📝 Entregables

### 1. Código Implementado (60%)

**Requisitos mínimos:**
- Implementaciones completas y funcionales
- Código limpio y bien documentado
- Pruebas y validación
- Manejo apropiado de errores

### 2. Notebook de Experimentación (25%)

**Debe incluir:**
- Experimentos con diferentes configuraciones
- Visualizaciones claras
- Análisis de resultados
- Comparaciones y conclusiones

### 3. Reporte Técnico (15%)

**Secciones:**
1. Introducción y objetivos
2. Metodología
3. Resultados experimentales
4. Análisis y discusión
5. Conclusiones

**Extensión:** 3-5 páginas

### Formato de Entrega

```
Lab04_Entrega/
├── codigo/
│   └── [archivos .py]
├── notebooks/
│   └── experimentos.ipynb
├── reporte/
│   └── reporte_lab04.pdf
└── README.md
```

## 🎯 Criterios de Evaluación (CDIO)

### Concebir (25%)

**Comprender el problema:**
- Identificar requisitos y restricciones
- Analizar alternativas de solución
- Reconocer implicaciones de decisiones de diseño

### Diseñar (25%)

**Planificar soluciones:**
- Diseñar arquitecturas apropiadas
- Estructurar código eficientemente
- Considerar escalabilidad y mantenibilidad

### Implementar (30%)

**Construcción:**
- Código funcional y correcto
- Implementación eficiente
- Documentación adecuada
- Pruebas comprehensivas

### Operar (20%)

**Validación y análisis:**
- Experimentación sistemática
- Análisis crítico de resultados
- Visualizaciones informativas
- Conclusiones fundamentadas

### Rúbrica Detallada

| Criterio | Excelente (90-100%) | Bueno (75-89%) | Satisfactorio (60-74%) | Insuficiente (<60%) |
|----------|-------------------|---------------|---------------------|-------------------|
| **Implementación** | Impecable, eficiente, documentado | Funcional con docs | Básico funcional | Con errores |
| **Experimentación** | Análisis profundo | Completo | Básico | Incompleto |
| **Documentación** | Excelente | Buena | Básica | Pobre |
| **Comprensión** | Dominio total | Buen entendimiento | Comprensión básica | Comprensión limitada |

## 📚 Referencias Adicionales

### Libros

1. **"Deep Learning" - Goodfellow, Bengio, Courville**
   - Capítulos relevantes para este lab
   - www.deeplearningbook.org

2. **"Neural Networks and Deep Learning" - Michael Nielsen**
   - neuralnetworksanddeeplearning.com

### Recursos Online

1. **CS231n: Stanford**
   - http://cs231n.stanford.edu/

2. **3Blue1Brown: Neural Networks**
   - Videos educativos excelentes

3. **TensorFlow Playground**
   - https://playground.tensorflow.org/

### Documentación

- NumPy: https://numpy.org/doc/
- Matplotlib: https://matplotlib.org/
- Python: https://docs.python.org/3/

## 🎓 Notas Finales

### Conceptos Clave para Recordar

1. **Pérdida cuantifica error:** Entre predicciones y realidad
2. **MSE para regresión:** Penaliza errores grandes
3. **Cross-Entropy para clasificación:** Interpretación probabilística
4. **Gradient descent minimiza:** Ajusta parámetros iterativamente
5. **Learning rate crucial:** Balance velocidad/estabilidad
6. **Mini-batch es estándar:** Mejor que batch o SGD puro
7. **Overfitting es común:** Entrenar mucho en pocos datos
8. **Regularización ayuda:** L2, L1, dropout, early stopping

### Preparación para el Siguiente Lab

**Lab 05: Backpropagation**

Aprenderás:
- Chain rule para redes
- Grafos computacionales
- Cálculo eficiente de gradientes
- Implementación completa de backprop

Prepárate repasando cálculo y chain rule.

### Consejos de Estudio

1. **Implementa desde cero** - No uses frameworks todavía
2. **Visualiza** - Dibuja y grafica para entender
3. **Experimenta** - Prueba diferentes configuraciones
4. **Debug sistemáticamente** - Verifica paso a paso
5. **Documenta** - Anota hallazgos y experimentos

### Solución de Problemas Comunes

**Errores de dimensiones:**
- Verifica shape de todas las matrices
- Usa print(variable.shape) liberalmente

**Resultados inesperados:**
- Verifica inicialización
- Asegura reproducibilidad con seed
- Revisa cada paso del cálculo

**Código lento:**
- Usa vectorización de NumPy
- Evita loops innecesarios
- Procesa en batches

### Certificación de Completitud

Has completado exitosamente Lab 04 cuando puedas:

- [ ] Comprender qué son funciones de pérdida y su propósito
- [ ] Implementar MSE, MAE, Cross-Entropy desde cero
- [ ] Elegir función de pérdida apropiada para cada problema
- [ ] Implementar gradient descent básico
- [ ] Comprender efecto del learning rate
- [ ] Reconocer y detectar overfitting
- [ ] Calcular derivadas de funciones de pérdida
- [ ] Combinar pérdida con activación eficientemente
- [ ] Aplicar regularización básica

**¡Felicitaciones!** Continúa con el siguiente laboratorio.

---

**¿Preguntas?** Revisa teoría, experimenta, y consulta referencias.

**¡Éxito en tu aprendizaje! 🚀**
