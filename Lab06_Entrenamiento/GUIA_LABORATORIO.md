# Guía de Laboratorio: Entrenamiento de Redes Neuronales

## 📋 Información del Laboratorio

**Título:** Fundamentos de Deep Learning - Entrenamiento de Redes Neuronales  
**Código:** Lab 06  
**Duración:** 2-3 horas  
**Nivel:** Intermedio-Avanzado  

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Implementar un loop de entrenamiento completo end-to-end
2. Dividir datos correctamente en conjuntos train/validation/test
3. Comprender y aplicar conceptos de época, batch e iteración
4. Implementar early stopping para prevenir overfitting
5. Monitorear métricas de entrenamiento y validación en tiempo real
6. Detectar y diagnosticar overfitting y underfitting
7. Aplicar técnicas de regularización (L1, L2, Dropout)
8. Implementar learning rate scheduling y decay
9. Optimizar hiperparámetros mediante validación
10. Guardar y cargar modelos (checkpointing)

## 📚 Prerrequisitos

### Conocimientos

- Python intermedio-avanzado (POO, manejo de datos)
- NumPy avanzado (operaciones matriciales, broadcasting)
- Backpropagation y cálculo de gradientes (Lab 05)
- Funciones de pérdida y activación (Labs 03-04)
- Conceptos básicos de overfitting

### Software

- Python 3.8+
- NumPy 1.19+
- Matplotlib (visualizaciones)
- Scikit-learn (división de datos, métricas)
- Jupyter Notebook (recomendado)

### Material de Lectura

Antes de comenzar, lee:
- `teoria.md` - Marco teórico completo sobre entrenamiento
- `README.md` - Estructura del laboratorio y recursos
- Labs anteriores (especialmente Lab 05 sobre Backpropagation)

## 📖 Introducción

### Del Gradiente a la Inteligencia

Has aprendido a calcular gradientes con backpropagation. Ahora viene la parte emocionante: **entrenar** una red neuronal para que realmente aprenda a resolver problemas.

El entrenamiento es el proceso iterativo mediante el cual:
1. La red hace predicciones
2. Medimos qué tan incorrectas son (pérdida)
3. Calculamos cómo mejorar (gradientes)
4. Ajustamos los parámetros (optimización)
5. ¡Repetimos miles de veces!

**Analogía del aprendizaje:**

Imagina aprender a tocar guitarra:
- **Época**: Practicar la canción completa una vez
- **Batch**: Practicar un fragmento específico
- **Iteración**: Un intento de tocar ese fragmento
- **Learning rate**: Qué tan drástico ajustas tu técnica
- **Validation**: Tocar para un amigo que te da feedback
- **Early stopping**: Dejar de practicar cuando ya lo tocas bien

### El Loop de Entrenamiento

El corazón de todo entrenamiento es este loop simple pero poderoso:

```
PARA cada época:
    PARA cada batch de datos:
        1. Forward pass: hacer predicciones
        2. Calcular pérdida
        3. Backward pass: calcular gradientes
        4. Actualizar parámetros
    
    Evaluar en validation set
    
    SI validation no mejora:
        Aplicar early stopping
```

### Conceptos Clave

**Época (Epoch):**
Un pase completo a través de todos los datos de entrenamiento.
```
1 época = procesar 100% de los datos de entrenamiento
```

**Batch:**
Subconjunto de datos procesados simultáneamente.
```
Dataset de 1000 muestras, batch size 32
→ 32 batches por época (1000 / 32 ≈ 31.25)
```

**Iteración:**
Un paso de actualización de parámetros (procesar un batch).
```
Iteraciones por época = total_muestras / batch_size
```

**Learning Rate:**
Controla el tamaño del paso de optimización.
```
W_nuevo = W_viejo - learning_rate × gradiente
```

### División de Datos

**Train (Entrenamiento)**: 70%
- Datos que el modelo ve durante entrenamiento
- Se usan para ajustar parámetros (W, b)

**Validation (Validación)**: 15%
- Datos para evaluar durante entrenamiento
- Se usan para ajustar hiperparámetros
- Detectan overfitting

**Test (Prueba)**: 15%
- Datos que el modelo NUNCA ve durante entrenamiento
- Evaluación final del rendimiento real
- Simulan datos del mundo real

**Regla de oro:** ¡NUNCA uses datos de test para tomar decisiones de entrenamiento!

### Problemas Comunes

**Underfitting (Subajuste):**
```
Pérdida de entrenamiento: ALTA
Pérdida de validación: ALTA
→ Modelo demasiado simple
```

**Overfitting (Sobreajuste):**
```
Pérdida de entrenamiento: BAJA
Pérdida de validación: ALTA
→ Modelo memorizó datos de entrenamiento
```

**Buen ajuste:**
```
Pérdida de entrenamiento: BAJA
Pérdida de validación: BAJA y cercana a train
→ Modelo generaliza bien
```

### Aplicaciones en el Mundo Real

El entrenamiento efectivo es crucial para:
- **Medicina**: Modelos que diagnostican enfermedades con precisión
- **Vehículos autónomos**: Redes que deben generalizar a cualquier carretera
- **Finanzas**: Prevenir overfitting en datos históricos
- **PLN**: Modelos de lenguaje entrenados en billones de palabras
- **Visión**: ImageNet (14M imágenes, semanas de entrenamiento)

## 🤔 Preguntas de Reflexión Iniciales

1. ¿Por qué necesitamos dividir datos en train/val/test?
2. ¿Qué pasaría si usamos todo el dataset para entrenar?
3. ¿Cómo sabemos cuándo detener el entrenamiento?
4. ¿Por qué procesar datos en batches en lugar de todos a la vez?
5. ¿Qué indica que un modelo está en overfitting?

## 🔬 Parte 1: Fundamentos del Entrenamiento (45 min)

### 1.1 Loop de Entrenamiento Básico

Empecemos con la estructura más simple:

#### Fundamento Teórico: División de Datos y Normalización

Antes de ejecutar cualquier entrenamiento, es imprescindible preparar los datos correctamente. La **división en conjuntos train/validación/test** obedece a un principio estadístico fundamental: medir la capacidad de generalización del modelo en datos que nunca ha visto. El conjunto de entrenamiento ajusta los parámetros internos (pesos y sesgos); el conjunto de validación nos guía para tomar decisiones de diseño (hiperparámetros, arquitectura, cuándo parar) sin contaminar la estimación final; y el conjunto de test proporciona una medida honesta e imparcial del rendimiento real del modelo sobre datos del mundo real. Usar datos de test durante el desarrollo equivale a "hacer trampa en el examen" y produce estimaciones de rendimiento optimistas que no se sostienen en producción.

La distribución estándar **70% train / 15% val / 15% test** es un buen punto de partida para datasets de tamaño medio (miles de muestras). Para datasets muy grandes (millones de ejemplos) puede usarse una partición 98/1/1 porque incluso el 1% de test representa decenas de miles de muestras suficientes para estimaciones estadísticamente robustas. En datasets muy pequeños (cientos de muestras), se recomienda la **validación cruzada K-fold** en lugar de una sola división, porque maximiza el uso de los datos disponibles para entrenamiento y proporciona estimaciones más confiables del rendimiento.

```
División de datos:
─────────────────────────────────────────────────────────────────
Dataset completo (N muestras)
         │
         ├──► Train set (70%)  → Ajustar W, b por backpropagation
         │
         ├──► Validation set (15%) → Monitorear, early stopping,
         │                           selección de hiperparámetros
         │
         └──► Test set (15%)   → Evaluación FINAL (solo una vez)
─────────────────────────────────────────────────────────────────
```

La **normalización de características** (restar la media y dividir por la desviación estándar) es igualmente crítica. Cuando las características tienen escalas muy distintas —por ejemplo, una columna con valores en el rango [0, 1] y otra en [0, 10000]— los gradientes de los pesos asociados a la característica grande dominan la actualización, haciendo que el entrenamiento sea extremadamente lento o inestable. Con los datos normalizados, todas las características contribuyen de forma equilibrada a la función de pérdida, la superficie de error se vuelve más esférica y el descenso por gradiente converge con menos oscilaciones.

```
Sin normalización:           Con normalización (Z-score):
  Pérdida                       Pérdida
    │  zig-zag                    │  descenso suave
    │ /\/\/\/\                    │ ╲
    │/        \___                │  ╲___
    └──────────► épocas           └──────────► épocas
```

**Importante:** la media y desviación estándar deben calcularse **sólo** sobre el conjunto de entrenamiento y luego aplicarse a validación y test; de lo contrario, estaríamos filtrando información futura al modelo (data leakage). La fórmula de normalización es:

```text
X_normalizado = (X - μ_train) / (σ_train + ε)

donde:
  μ_train = media calculada en X_train
  σ_train = desviación estándar calculada en X_train
  ε = 1e-8  (evita división por cero)
```

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generar datos sintéticos
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,
                          n_informative=15, n_redundant=5, random_state=42)

# Dividir datos
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape[0]} muestras")
print(f"Validation: {X_val.shape[0]} muestras")
print(f"Test: {X_test.shape[0]} muestras")

# Normalizar datos (importante!)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / (std + 1e-8)
X_val = (X_val - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)
```

**Simple Training Loop:**

#### Fundamento Teórico: La Clase SimpleTrainer

La clase `SimpleTrainer` encapsula el **loop de entrenamiento completo** siguiendo el ciclo de cuatro pasos que define el aprendizaje supervisado: *forward pass*, cálculo de pérdida, *backward pass* y actualización de parámetros. Comprender cada paso es fundamental antes de trabajar con frameworks de alto nivel como PyTorch o TensorFlow, que los abstraen automáticamente.

```
Loop de entrenamiento (una época):
════════════════════════════════════════════════════════
  X_train ──► [Forward Pass] ──► ŷ (predicciones)
                                   │
                               [Pérdida L]
                               L = -mean(y·log(ŷ) + (1-y)·log(1-ŷ))
                                   │
                           [Backward Pass]
                           ∂L/∂W₂, ∂L/∂b₂, ∂L/∂W₁, ∂L/∂b₁
                                   │
                         [Actualización GD]
                         W ← W - η · ∂L/∂W
                         b ← b - η · ∂L/∂b
════════════════════════════════════════════════════════
```

En el **forward pass**, los datos de entrada se propagan capa por capa hasta producir una predicción; la función de pérdida (en este caso *Binary Cross-Entropy*) cuantifica el error asignando un escalar positivo que crece cuanto más se equivoca el modelo. El *backward pass* aplica la regla de la cadena para propagar el gradiente de la pérdida hacia atrás a través de cada capa, obteniendo `∂L/∂W` y `∂L/∂b` para cada conjunto de parámetros. La **regla de actualización** `W ← W − η·∂L/∂W` mueve cada peso en la dirección que reduce la pérdida, siendo `η` (learning rate) el hiperparámetro que controla el tamaño del paso.

El hecho de que `SimpleTrainer` ejecute el paso completo con todos los datos a la vez por época se denomina **Batch Gradient Descent** puro. Es conceptualmente correcto pero ineficiente con datasets grandes —lo cual motiva la siguiente sección sobre mini-batches. Para este dataset de 1000 muestras, el comportamiento esperado es:

- **Épocas 1-20:** Descenso rápido de la pérdida (fase de aprendizaje principal)
- **Épocas 20-60:** Descenso más lento, convergencia gradual
- **Épocas 60+:** Plateau, pequeñas oscilaciones alrededor del mínimo

Si la pérdida no desciende en las primeras 10 épocas, el learning rate probablemente es demasiado pequeño (< 0.001) o demasiado grande (> 1.0) y está causando divergencia.

```python
class SimpleTrainer:
    """Trainer básico para redes neuronales"""
    
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.lr = learning_rate
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self, X, y):
        """Entrena una época completa"""
        # Forward
        predictions = self.model.forward(X)
        
        # Loss
        loss = self.compute_loss(predictions, y)
        
        # Backward
        grad = predictions - y.reshape(-1, 1)
        self.model.backward(grad)
        
        # Update
        self.model.update(self.lr)
        
        return loss
    
    def compute_loss(self, predictions, targets):
        """Binary Cross-Entropy"""
        targets = targets.reshape(-1, 1)
        epsilon = 1e-8
        loss = -np.mean(
            targets * np.log(predictions + epsilon) +
            (1 - targets) * np.log(1 - predictions + epsilon)
        )
        return loss
    
    def evaluate(self, X, y):
        """Evalúa el modelo en un dataset"""
        predictions = self.model.forward(X)
        loss = self.compute_loss(predictions, y)
        
        # Accuracy
        pred_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_classes.flatten() == y)
        
        return loss, accuracy
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Loop de entrenamiento completo"""
        print("Iniciando entrenamiento...")
        print("=" * 60)
        
        for epoch in range(epochs):
            # Entrenar
            train_loss = self.train_epoch(X_train, y_train)
            
            # Evaluar
            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            # Guardar historia
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Mostrar progreso
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f}")
        
        print("=" * 60)
        print("Entrenamiento completado!")

# Ejemplo de uso (asumiendo que tienes un modelo)
# trainer = SimpleTrainer(model, learning_rate=0.01)
# trainer.train(X_train, y_train, X_val, y_val, epochs=100)
```

### 1.2 Procesamiento en Batches

#### Fundamento Teórico: Tres Variantes de Descenso por Gradiente

El procesamiento por batches no es un mero truco de eficiencia: tiene profundas implicaciones teóricas sobre la calidad del entrenamiento. Existen tres variantes principales del descenso por gradiente que se diferencian en cuántos ejemplos se usan para calcular el gradiente en cada actualización:

**1. Batch Gradient Descent (GD puro, batch_size = N):**
Usa el dataset completo en cada paso de actualización. El gradiente calculado es exacto (sin ruido estadístico), produciendo actualizaciones suaves. Sin embargo, es computacionalmente prohibitivo en datasets grandes, no cabe en memoria GPU con millones de ejemplos, y puede quedar atrapado en mínimos locales al no tener ruido que le ayude a escapar.

```
GD puro:
Iteración 1: gradiente con 1000 muestras → W actualizado
Iteración 2: gradiente con 1000 muestras → W actualizado
...
1 época = 1 actualización de parámetros
```

**2. Stochastic Gradient Descent (SGD, batch_size = 1):**
Actualiza los parámetros tras procesar **un único ejemplo**. El gradiente es muy ruidoso (alta varianza), lo que paradójicamente actúa como **regularización implícita**: el ruido estocástico permite al optimizador escapar de mínimos locales poco profundos. El inconveniente es que la convergencia es errática y no aprovecha el paralelismo hardware.

```
SGD (batch=1):
Iteración 1: gradiente con muestra[0] → W actualizado
Iteración 2: gradiente con muestra[1] → W actualizado
...
1 época = 1000 actualizaciones de parámetros
```

**3. Mini-batch SGD (batch_size típico: 16–256):**
Combina lo mejor de ambos mundos. Al calcular el gradiente sobre un subconjunto pequeño pero representativo, se reduce suficientemente el ruido para tener actualizaciones direccionalmente correctas, mientras se mantiene el beneficio regularizador del ruido estocástico. Los mini-batches aprovechan al máximo las operaciones matriciales vectorizadas de las GPU/CPU modernas.

```
Mini-batch SGD (batch=32):
Iteración 1: gradiente con muestras[0:32]   → W actualizado
Iteración 2: gradiente con muestras[32:64]  → W actualizado
...
Iteración 31: gradiente con muestras[992:1000] → W actualizado
1 época = 32 actualizaciones de parámetros
```

**Comparación de las tres variantes:**

| Propiedad | Batch GD | SGD (b=1) | Mini-batch SGD |
|-----------|----------|-----------|----------------|
| Varianza del gradiente | Nula (exacto) | Muy alta | Baja-moderada |
| Velocidad por época | Lenta (1 update) | Rápida (N updates) | Balanceada |
| Uso de memoria GPU | Muy alto | Mínimo | Configurable |
| Regularización implícita | No | Sí (mucho ruido) | Sí (ruido moderado) |
| Estándar en industria | Raro | Raro | **Sí** |

**¿Por qué batch_size=32 es tan común?** La elección de 32 tiene raíces empíricas y prácticas: es suficientemente grande para aprovechar la paralelización hardware (múltiplo de potencias de 2), lo bastante pequeño para que el gradiente tenga varianza estocástica beneficiosa, y produce actualizaciones frecuentes que aceleran la convergencia. Investigaciones como las de Keskar et al. (2017) muestran que los batch sizes muy grandes tienden a converger a **mínimos planos** (con mejor generalización) mientras los muy pequeños pueden caer en **mínimos agudos** (menos robustos). Como regla práctica, empieza con 32 y ajusta según los recursos computacionales disponibles.

El **shuffle aleatorio** antes de cada época es fundamental: asegura que cada mini-batch sea una muestra representativa del dataset completo, evitando que el modelo sobreajuste al orden de los datos.

El procesamiento por batches es esencial para eficiencia:

```python
class BatchTrainer:
    """Trainer con mini-batch SGD"""
    
    def __init__(self, model, learning_rate=0.01, batch_size=32):
        self.model = model
        self.lr = learning_rate
        self.batch_size = batch_size
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def create_batches(self, X, y):
        """Divide datos en batches"""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)  # Importante: mezclar datos
        
        batches = []
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches
    
    def train_epoch(self, X_train, y_train):
        """Entrena una época con mini-batches"""
        batches = self.create_batches(X_train, y_train)
        epoch_loss = 0
        
        for batch_X, batch_y in batches:
            # Forward
            predictions = self.model.forward(batch_X)
            
            # Loss
            loss = self.compute_loss(predictions, batch_y)
            epoch_loss += loss
            
            # Backward
            grad = predictions - batch_y.reshape(-1, 1)
            self.model.backward(grad)
            
            # Update
            self.model.update(self.lr)
        
        # Pérdida promedio de la época
        return epoch_loss / len(batches)
    
    def compute_loss(self, predictions, targets):
        """Binary Cross-Entropy"""
        targets = targets.reshape(-1, 1)
        epsilon = 1e-8
        loss = -np.mean(
            targets * np.log(predictions + epsilon) +
            (1 - targets) * np.log(1 - predictions + epsilon)
        )
        return loss
    
    def evaluate(self, X, y):
        """Evalúa modelo"""
        predictions = self.model.forward(X)
        loss = self.compute_loss(predictions, y)
        
        pred_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_classes.flatten() == y)
        
        return loss, accuracy
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, verbose=True):
        """Loop de entrenamiento con batches"""
        
        for epoch in range(epochs):
            # Entrenar con batches
            train_loss = self.train_epoch(X_train, y_train)
            train_loss_full, train_acc = self.evaluate(X_train, y_train)
            
            # Validar
            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            # Guardar historia
            self.history['train_loss'].append(train_loss_full)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Mostrar progreso
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                      f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f}")
        
        return self.history
```

**Actividad 1.1:** Implementa el trainer y prueba con diferentes batch sizes (1, 16, 32, 128). ¿Qué observas?

> **¿Qué debes observar y documentar?** Al variar el batch size notarás diferencias claras en la *suavidad* de las curvas de pérdida: con batch_size=1 la pérdida oscilará fuertemente época a época; con batch_size grande las curvas serán más suaves pero la convergencia inicial puede ser más lenta. Documenta el tiempo de entrenamiento por época para cada configuración y observa si los modelos con batch size pequeño alcanzan menor pérdida final (efecto regularizador del ruido). Reflexiona sobre el compromiso velocidad-estabilidad-calidad del modelo final.

### 1.3 Visualización del Entrenamiento

#### Fundamento Teórico: Interpretación de Curvas de Aprendizaje

Las **curvas de aprendizaje** son la herramienta de diagnóstico más poderosa durante el entrenamiento de redes neuronales. Representan cómo evoluciona la pérdida (y la exactitud) en los conjuntos de entrenamiento y validación a lo largo de las épocas, y su forma nos da información directa sobre el estado de salud del modelo.

**Patrones de diagnóstico en las curvas de pérdida:**

```
BUEN AJUSTE:              OVERFITTING:              UNDERFITTING:
 Pérdida                   Pérdida                   Pérdida
   │ train───────────╮       │ train──────────╮         │ train──────────
   │ val──────────╮  │       │              ╰╯ val       │ val────────────
   │              ╰──╯       │ val↗ (diverge)           │ (ambas altas)
   └─────────── épocas       └─────────── épocas        └─────────── épocas

  Gap pequeño y estable    Gap creciente con épocas    Ambas curvas altas
```

**Overfitting (Sobreajuste):** Se diagnostica cuando la pérdida de entrenamiento continúa bajando mientras la pérdida de validación deja de mejorar o comienza a subir. El **gap** `val_loss − train_loss` es el indicador cuantitativo clave: un gap creciente con cada época es la firma digital del overfitting. Visualmente, las dos curvas se separan en forma de tijera. El modelo ha aprendido los patrones específicos del conjunto de entrenamiento (incluido el ruido) en lugar de las relaciones generalizables.

| Gap | Diagnóstico | Acción recomendada |
|-----|-------------|-------------------|
| < 0.05 | Buen ajuste | Continuar o aumentar capacidad |
| 0.05 – 0.15 | Ligero overfitting | Monitorear, considerar regularización |
| > 0.15 | Overfitting severo | Aplicar L2/Dropout, early stopping |
| Negativo | Underfitting | Aumentar capacidad o épocas |

**Underfitting (Subajuste):** Tanto la pérdida de entrenamiento como la de validación permanecen altas. Las curvas están cerca entre sí (gap pequeño) pero en un nivel de pérdida elevado. Esto indica que el modelo carece de capacidad suficiente para capturar la complejidad del problema.

**Buen ajuste:** Ambas curvas descienden juntas y se estabilizan en un nivel bajo, con un gap pequeño y estable. La curva de validación puede ser ligeramente superior a la de entrenamiento (es normal) pero no debería separarse de ella significativamente.

**¿Por qué monitorear tanto pérdida como exactitud?** La pérdida guía directamente la optimización y detecta problemas sutiles que la exactitud puede ocultar: un modelo puede tener exactitud alta pero pérdida creciente si está sobreconfiado en sus predicciones incorrectas. La exactitud es más intuitiva para comunicar el rendimiento a no especialistas. Usar ambas métricas juntas proporciona una imagen completa del comportamiento del modelo. Si ambas métricas cuentan historias diferentes (alta exactitud pero pérdida creciente), la pérdida es el indicador más confiable del estado real del modelo.

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    """Visualiza curvas de aprendizaje"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Pérdida
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Curva de Pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Curva de Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Diagnóstico
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    gap = final_val_loss - final_train_loss
    
    print("\n=== DIAGNÓSTICO ===")
    print(f"Pérdida final - Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f}")
    print(f"Gap (Val - Train): {gap:.4f}")
    
    if gap > 0.1:
        print("⚠️  OVERFITTING detectado!")
        print("Soluciones: Regularización, Dropout, Más datos, Early stopping")
    elif final_train_loss > 0.5:
        print("⚠️  UNDERFITTING detectado!")
        print("Soluciones: Modelo más complejo, Más épocas, Ajustar learning rate")
    else:
        print("✓ Modelo bien ajustado")

# Usar
# plot_training_history(trainer.history)
```

## 🔬 Parte 2: Early Stopping (30 min)

### 2.1 Implementación de Early Stopping

Early stopping previene overfitting deteniendo el entrenamiento cuando validation deja de mejorar:

#### Fundamento Teórico: Early Stopping, Patience y Checkpointing

El **early stopping** es quizás la técnica de regularización más elegante porque no modifica la arquitectura del modelo ni la función de pérdida: simplemente detiene el entrenamiento en el momento óptimo antes de que el modelo comience a memorizar el ruido de los datos de entrenamiento. Desde una perspectiva teórica, el entrenamiento sigue una trayectoria en el espacio de parámetros: en las primeras épocas el modelo aprende patrones genuinos (mejora en validación), pero a partir de cierto punto comienza a sobreajustar los ejemplos de entrenamiento individuales (validación empeora). El early stopping identifica ese punto de inflexión y "congela" el modelo en su mejor estado.

```
Comportamiento típico del entrenamiento con early stopping:

  Val Loss
    │
    │\
    │ \
    │  \____
    │       \___
    │           \___
    │               ╲___╱╲          ← punto de inflexión
    │                    ╲___╱╲___  ← overfitting inicia aquí
    │                ↑
    │         MEJOR CHECKPOINT
    └────────────────────────────── épocas
    
    [←─── patience ──→]
         Sin mejora     → STOP y restaurar checkpoint
```

El parámetro **patience** define cuántas épocas consecutivas sin mejora en la validación se toleran antes de detener el entrenamiento. Un patience bajo (ej: 5) detiene el entrenamiento agresivamente y puede interrumpirlo en una meseta temporal antes de que el modelo retome su mejora; un patience alto (ej: 30) es más tolerante con la fluctuaciones pero puede resultar en más épocas de cómputo innecesarias. La elección depende de la suavidad esperada de las curvas: datasets ruidosos requieren patience mayor.

| Patience | Ventaja | Desventaja | Cuándo usarlo |
|----------|---------|------------|---------------|
| 5-7 | Ahorra tiempo de cómputo | Puede detenerse en mesetas | Curvas muy suaves |
| 10-15 | Balance equilibrado | Estándar recomendado | **Caso general** |
| 20-30 | Explora más épocas | Mayor cómputo | Curvas con mesetas largas |

El concepto de **min_delta** (mejora mínima para considerarse progreso) complementa al patience: en lugar de considerar "mejora" cualquier reducción por mínima que sea de la pérdida de validación, se exige que la reducción supere un umbral `δ`. Esto evita que pequeñas fluctuaciones numéricas retrasen el early stopping indefinidamente. Por ejemplo, si `min_delta=0.001`, una reducción de pérdida de 0.0001 no se contabiliza como mejora genuina.

```
Lógica de early stopping con min_delta:

  nueva_val_loss < mejor_val_loss - min_delta?
        │
        ├── SÍ → Mejora genuina detectada
        │         • Actualizar mejor_val_loss
        │         • patience_counter = 0
        │         • Guardar checkpoint
        │
        └── NO → Sin mejora suficiente
                  • patience_counter += 1
                  • Si patience_counter >= patience: STOP y restaurar
```

El **checkpointing** (guardado del mejor modelo) es inseparable del early stopping: como el entrenamiento se detiene sólo después de `patience` épocas sin mejora, el último estado del modelo NO es el mejor. El checkpoint restaura los pesos correspondientes a la época con menor pérdida de validación, garantizando que se usa el modelo en su punto óptimo de generalización y no el modelo "degradado" por las últimas épocas de sobreajuste. En sistemas de producción, los checkpoints también protegen contra interrupciones inesperadas del entrenamiento (fallas de hardware, cortes de luz).

```python
class TrainerWithEarlyStopping:
    """Trainer con early stopping"""
    
    def __init__(self, model, learning_rate=0.01, batch_size=32):
        self.model = model
        self.lr = learning_rate
        self.batch_size = batch_size
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Para early stopping
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.patience_counter = 0
    
    def create_batches(self, X, y):
        """Divide datos en batches"""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        batches = []
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches
    
    def train_epoch(self, X_train, y_train):
        """Entrena una época"""
        batches = self.create_batches(X_train, y_train)
        epoch_loss = 0
        
        for batch_X, batch_y in batches:
            predictions = self.model.forward(batch_X)
            loss = self.compute_loss(predictions, batch_y)
            epoch_loss += loss
            
            grad = predictions - batch_y.reshape(-1, 1)
            self.model.backward(grad)
            self.model.update(self.lr)
        
        return epoch_loss / len(batches)
    
    def compute_loss(self, predictions, targets):
        """Binary Cross-Entropy"""
        targets = targets.reshape(-1, 1)
        epsilon = 1e-8
        return -np.mean(
            targets * np.log(predictions + epsilon) +
            (1 - targets) * np.log(1 - predictions + epsilon)
        )
    
    def evaluate(self, X, y):
        """Evalúa modelo"""
        predictions = self.model.forward(X)
        loss = self.compute_loss(predictions, y)
        pred_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_classes.flatten() == y)
        return loss, accuracy
    
    def save_checkpoint(self):
        """Guarda mejor modelo"""
        # En una implementación real, guardarías W y b de cada capa
        self.best_weights = {
            'layer1_W': self.model.layer1.W.copy(),
            'layer1_b': self.model.layer1.b.copy(),
            'layer2_W': self.model.layer2.W.copy(),
            'layer2_b': self.model.layer2.b.copy(),
        }
    
    def load_checkpoint(self):
        """Restaura mejor modelo"""
        if self.best_weights is not None:
            self.model.layer1.W = self.best_weights['layer1_W'].copy()
            self.model.layer1.b = self.best_weights['layer1_b'].copy()
            self.model.layer2.W = self.best_weights['layer2_W'].copy()
            self.model.layer2.b = self.best_weights['layer2_b'].copy()
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, patience=10, verbose=True):
        """
        Entrenar con early stopping
        
        patience: número de épocas sin mejora antes de detener
        """
        print(f"Entrenando con early stopping (patience={patience})...")
        print("=" * 70)
        
        for epoch in range(epochs):
            # Entrenar
            train_loss = self.train_epoch(X_train, y_train)
            train_loss_full, train_acc = self.evaluate(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            # Guardar historia
            self.history['train_loss'].append(train_loss_full)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint()
                improvement = "✓ Mejora!"
            else:
                self.patience_counter += 1
                improvement = f"No mejora ({self.patience_counter}/{patience})"
            
            # Mostrar progreso
            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train: L={train_loss_full:.4f} A={train_acc:.4f} | "
                      f"Val: L={val_loss:.4f} A={val_acc:.4f} | {improvement}")
            
            # Detener si no hay mejora
            if self.patience_counter >= patience:
                print(f"\n⚠️  Early stopping en época {epoch}")
                print(f"No mejora en {patience} épocas consecutivas")
                print(f"Mejor val_loss: {self.best_val_loss:.4f}")
                
                # Restaurar mejor modelo
                self.load_checkpoint()
                print("Modelo restaurado al mejor checkpoint")
                break
        
        else:
            print("\nEntrenamiento completado sin early stopping")
        
        print("=" * 70)
        return self.history

# Ejemplo de uso
# trainer = TrainerWithEarlyStopping(model, learning_rate=0.01, batch_size=32)
# history = trainer.train(X_train, y_train, X_val, y_val, 
#                         epochs=200, patience=15)
```

**Actividad 2.1:** Experimenta con diferentes valores de patience (5, 10, 20). ¿Cómo afecta al entrenamiento?

> **¿Qué debes observar y documentar?** Registra en qué época se detiene el entrenamiento para cada valor de patience y cuál es la pérdida de validación del checkpoint restaurado. Con patience=5 probablemente el entrenamiento se detenga prematuramente durante una meseta temporal; con patience=20 puede completar más épocas pero también gastar más tiempo de cómputo. Compara las pérdidas finales en el conjunto de **test** (no validación) de los tres modelos para evaluar cuál generaliza mejor. Esto ilustra el tradeoff entre detención temprana y exploración suficiente del espacio de soluciones.

## 🔬 Parte 3: Regularización y Técnicas Avanzadas (50 min)

### 3.1 Regularización L2 (Weight Decay)

#### Fundamento Teórico: Regularización como Penalización de Complejidad

La **regularización** es el conjunto de técnicas que previene el overfitting imponiendo restricciones sobre la complejidad del modelo. Matemáticamente, modifica la función de pérdida añadiendo un **término de penalización** que crece cuando los pesos del modelo toman valores muy grandes:

```
L_total = L_datos + λ · Ω(W)

donde:
  L_datos = pérdida original (ej: cross-entropy, MSE)
  Ω(W)    = penalización sobre los pesos del modelo
  λ       = hiperparámetro que balancea ambos términos
```

La **regularización L2** (*Ridge* o *weight decay*) usa `Ω(W) = ½ · Σ(Wᵢ²)`, la suma de los cuadrados de todos los pesos. Su gradiente `∂Ω/∂W = W` modifica la regla de actualización a:

```
W ← W - η · ∂L_datos/∂W - η · λ · W
W ← W · (1 - η·λ) - η · ∂L_datos/∂W
         ↑
     "weight decay": factor < 1 que reduce W en cada paso
```

Este factor `(1 − η·λ) < 1` es exactamente el "decaimiento" del peso en cada paso, de ahí el nombre *weight decay*. El efecto es que los pesos tienen una presión constante hacia cero, produciendo soluciones más **suaves y distribuidas** donde ningún peso individual domina las predicciones.

La **regularización L1** (*Lasso*) usa `Ω(W) = Σ|Wᵢ|`. Su gradiente es `λ·sign(W)`, que empuja los pesos exactamente a cero para los menos relevantes. Esto produce soluciones **dispersas (sparse)**: muchos pesos quedan en exactamente cero, equivalente a selección automática de características.

**Comparación L1 vs L2:**

| Propiedad | L1 (Lasso) | L2 (Ridge / Weight Decay) |
|-----------|-----------|--------------------------|
| Fórmula | λ·Σ&#124;W&#124; | λ/2·Σ(W²) |
| Tipo de solución | Dispersa (muchos ceros) | Densa (pesos pequeños) |
| Selección de features | **Sí** (implícita) | No |
| Diferenciable en W=0 | No (problema numérico) | Sí |
| Uso típico | Feature selection | **Regularización general** |

**Cómo elegir lambda:** Un `λ` muy pequeño no penaliza suficientemente y el overfitting persiste; un `λ` muy grande fuerza todos los pesos a cero y el modelo pierde capacidad expresiva (underfitting). La práctica estándar es búsqueda en escala logarítmica:

```text
Valores típicos a evaluar: λ ∈ {0.1, 0.01, 0.001, 0.0001}

λ = 0.1    → Regularización fuerte, riesgo de underfitting
λ = 0.01   → Regularización moderada (buen punto de inicio)
λ = 0.001  → Regularización suave
λ = 0.0001 → Regularización muy suave
```

El valor óptimo se selecciona usando validación cruzada: el que maximiza el rendimiento en validación sin degradar el de entrenamiento de forma significativa.

```python
class TrainerWithL2:
    """Trainer con regularización L2"""
    
    def __init__(self, model, learning_rate=0.01, batch_size=32, l2_lambda=0.01):
        self.model = model
        self.lr = learning_rate
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda  # Parámetro de regularización
        self.history = {'train_loss': [], 'val_loss': []}
    
    def compute_loss_with_l2(self, predictions, targets):
        """Pérdida con regularización L2"""
        # Pérdida base (cross-entropy)
        targets = targets.reshape(-1, 1)
        epsilon = 1e-8
        data_loss = -np.mean(
            targets * np.log(predictions + epsilon) +
            (1 - targets) * np.log(1 - predictions + epsilon)
        )
        
        # Término de regularización L2: λ * Σ(W²)
        l2_loss = 0
        l2_loss += np.sum(self.model.layer1.W ** 2)
        l2_loss += np.sum(self.model.layer2.W ** 2)
        l2_loss *= self.l2_lambda / 2
        
        total_loss = data_loss + l2_loss
        
        return total_loss, data_loss, l2_loss
    
    def train_epoch(self, X_train, y_train):
        """Entrena una época con L2"""
        batches = self.create_batches(X_train, y_train)
        epoch_loss = 0
        
        for batch_X, batch_y in batches:
            # Forward
            predictions = self.model.forward(batch_X)
            
            # Loss con L2
            total_loss, data_loss, l2_loss = self.compute_loss_with_l2(
                predictions, batch_y
            )
            epoch_loss += total_loss
            
            # Backward (gradiente de data loss)
            grad = predictions - batch_y.reshape(-1, 1)
            self.model.backward(grad)
            
            # Agregar gradiente L2 a los pesos
            # ∂(λ||W||²)/∂W = λ*W
            self.model.layer1.dW += self.l2_lambda * self.model.layer1.W
            self.model.layer2.dW += self.l2_lambda * self.model.layer2.W
            
            # Update
            self.model.update(self.lr)
        
        return epoch_loss / len(batches)
    
    def create_batches(self, X, y):
        """Crea batches"""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        batches = []
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches

# Comparar con y sin regularización
# trainer_sin_reg = TrainerWithL2(model1, l2_lambda=0.0)
# trainer_con_reg = TrainerWithL2(model2, l2_lambda=0.01)
```

### 3.2 Learning Rate Scheduling

#### Fundamento Teórico: Adaptación Dinámica del Paso de Aprendizaje

Un **learning rate fijo** es subóptimo durante todo el proceso de entrenamiento por razones geométricas claras: en las primeras épocas, el modelo está lejos del óptimo y un learning rate grande acelera la convergencia; pero en las épocas finales, cuando el modelo se acerca al óptimo, ese mismo learning rate grande hace que los parámetros "salten" alrededor del mínimo sin poder asentarse en él. Es el equivalente a intentar enroscar un tornillo con el destornillador a máxima potencia: rápido al principio pero impreciso al final. El **learning rate scheduling** resuelve esto reduciendo gradualmente la tasa de aprendizaje a medida que avanza el entrenamiento.

```
Problema del LR fijo:               Solución con LR scheduling:

  Pérdida                             Pérdida
    │  \                                │  \
    │   \                               │   \
    │    \     LR grande                │    \___
    │     ╲╱╲╱╲╱╲╱── oscilación         │        ╲___ LR reducido
    │                                   │             ╲___
    └─────────── épocas                 └─────────── épocas
```

Las tres estrategias de scheduling más usadas tienen comportamientos distintos:

**Step Decay:**
```
lr(t) = lr₀ × factor^(época // épocas_por_paso)

Ejemplo: lr₀=0.1, factor=0.5, épocas_por_paso=10
  Época 0-9:   lr = 0.1
  Época 10-19: lr = 0.05
  Época 20-29: lr = 0.025
```
Produce una curva de pérdida en escalones descendentes. Ideal cuando se sabe cuántas épocas necesita el modelo.

**Exponential Decay:** `lr(t) = lr₀ · rᵗ` donde `r < 1` (ej: r=0.95). La reducción es continua y suave. El LR decrece siempre, incluso si el modelo sigue mejorando, lo que puede ser una limitación.

**Reduce on Plateau:** Sólo reduce el LR cuando la pérdida de validación deja de mejorar durante `patience` épocas. Es el más adaptativo y es el **estándar recomendado** para la mayoría de problemas.

| Estrategia | Tipo | Ventaja principal | Limitación |
|------------|------|-------------------|------------|
| Step Decay | Manual | Predecible, fácil de depurar | Requiere configurar cuándo bajar |
| Exponential | Automático | Transición suave continua | LR baja siempre, incluso si mejora |
| **Plateau** | Adaptativo | Se adapta al problema | **Recomendado en práctica** |

El **learning rate warmup** (calentamiento) es una técnica complementaria usada en modelos grandes (Transformers, BERT): el LR empieza muy pequeño, aumenta linealmente durante las primeras épocas hasta el valor objetivo, y luego disminuye. El warmup estabiliza el entrenamiento en las primeras iteraciones cuando los pesos están aún muy alejados del óptimo y los gradientes son grandes e inestables.

```
LR con warmup + cosine annealing (estándar en Transformers):

  LR
  │        ╱╲
  │       ╱  ╲___
  │      ╱       ╲___
  │    ╱              ╲___
  │  ╱ (warmup)            ╲___ (decay)
  └────────────────────────── épocas
```

Los **Cyclical Learning Rates** (CLR, Smith 2017) proponen una idea contraintuitiva: en lugar de sólo decrecer, el LR oscila entre un mínimo y un máximo en ciclos. La intuición es que los aumentos periódicos del LR ayudan al modelo a "saltar" de mínimos locales hacia mejores regiones del espacio de pérdida, logrando mejores soluciones finales que el scheduling monotónicamente decreciente.

```python
class TrainerWithLRSchedule:
    """Trainer con learning rate scheduling"""
    
    def __init__(self, model, initial_lr=0.1, batch_size=32):
        self.model = model
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.batch_size = batch_size
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def step_decay(self, epoch, drop_factor=0.5, epochs_drop=10):
        """
        Step decay: reduce LR cada X épocas
        lr = initial_lr * drop_factor^(epoch // epochs_drop)
        """
        new_lr = self.initial_lr * (drop_factor ** (epoch // epochs_drop))
        return new_lr
    
    def exponential_decay(self, epoch, decay_rate=0.95):
        """
        Exponential decay: reducción exponencial
        lr = initial_lr * decay_rate^epoch
        """
        new_lr = self.initial_lr * (decay_rate ** epoch)
        return new_lr
    
    def reduce_on_plateau(self, epoch, val_losses, patience=5, factor=0.5):
        """
        Reduce LR si validación no mejora
        """
        if len(val_losses) < patience + 1:
            return self.current_lr
        
        # Verificar si hubo mejora en las últimas 'patience' épocas
        recent_best = min(val_losses[-(patience+1):-1])
        current = val_losses[-1]
        
        if current >= recent_best:
            new_lr = self.current_lr * factor
            print(f"   → Reducing LR: {self.current_lr:.6f} → {new_lr:.6f}")
            return new_lr
        
        return self.current_lr
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, 
              schedule_type='step', verbose=True):
        """
        Entrenar con LR scheduling
        
        schedule_type: 'step', 'exponential', 'plateau'
        """
        print(f"Entrenando con {schedule_type} LR scheduling...")
        print("=" * 70)
        
        for epoch in range(epochs):
            # Actualizar learning rate
            if schedule_type == 'step':
                self.current_lr = self.step_decay(epoch)
            elif schedule_type == 'exponential':
                self.current_lr = self.exponential_decay(epoch)
            elif schedule_type == 'plateau':
                self.current_lr = self.reduce_on_plateau(
                    epoch, self.history['val_loss']
                )
            
            self.history['learning_rate'].append(self.current_lr)
            
            # Entrenar época
            train_loss = self.train_epoch(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | "
                      f"LR={self.current_lr:.6f} | "
                      f"Train Loss={train_loss:.4f} | "
                      f"Val Loss={val_loss:.4f}")
        
        return self.history
    
    def train_epoch(self, X_train, y_train):
        """Entrena una época"""
        batches = self.create_batches(X_train, y_train)
        epoch_loss = 0
        
        for batch_X, batch_y in batches:
            predictions = self.model.forward(batch_X)
            loss = self.compute_loss(predictions, batch_y)
            epoch_loss += loss
            
            grad = predictions - batch_y.reshape(-1, 1)
            self.model.backward(grad)
            self.model.update(self.current_lr)  # Usar LR actual
        
        return epoch_loss / len(batches)
    
    def compute_loss(self, predictions, targets):
        """Binary Cross-Entropy"""
        targets = targets.reshape(-1, 1)
        epsilon = 1e-8
        return -np.mean(
            targets * np.log(predictions + epsilon) +
            (1 - targets) * np.log(1 - predictions + epsilon)
        )
    
    def evaluate(self, X, y):
        """Evalúa modelo"""
        predictions = self.model.forward(X)
        loss = self.compute_loss(predictions, y)
        pred_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_classes.flatten() == y)
        return loss, accuracy
    
    def create_batches(self, X, y):
        """Crea batches"""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        batches = []
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches

def plot_lr_schedule(history):
    """Visualiza evolución del learning rate"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['learning_rate'])
    plt.xlabel('Época')
    plt.ylabel('Learning Rate')
    plt.title('Evolución del Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Curvas de Aprendizaje')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

**Actividad 3.1:** Compara los tres tipos de scheduling. ¿Cuál converge más rápido?

> **¿Qué debes observar y documentar?** Ejecuta el mismo modelo con los tres tipos de scheduling (step, exponential, plateau) usando el mismo learning rate inicial y el mismo número máximo de épocas. Grafica la evolución del LR junto a las curvas de pérdida para visualizar la correlación entre los cambios de LR y las mejoras en la pérdida. Analiza: ¿qué estrategia alcanza la pérdida mínima primero? ¿Cuál produce la menor pérdida de validación final? ¿Cuál es más robusta a la elección inicial del LR? Documenta tus conclusiones con evidencia cuantitativa de los experimentos.

## 🔬 Parte 4: Monitoreo y Debugging (35 min)

### 4.1 Dashboard de Monitoreo

#### Fundamento Teórico: Métricas Clave y Diagnóstico en Tiempo Real

El **monitoreo activo** durante el entrenamiento es lo que diferencia un experimento de ML bien conducido de un simple script que se ejecuta a ciegas. Un dashboard de métricas permite detectar problemas a tiempo y tomar decisiones informadas: ajustar el LR, aumentar la regularización, ampliar la capacidad del modelo o detener el experimento por completo.

**¿Qué métricas son más importantes?**

| Métrica | Panel | Qué indica |
|---------|-------|-----------|
| `val_loss` | Curva de pérdida | Señal de optimización más sensible |
| `val_acc` | Curva de accuracy | Rendimiento interpretable |
| `val_loss - train_loss` | Gap de generalización | Indicador directo de overfitting |
| `learning_rate` | LR schedule | Verificar que el scheduler funciona |
| `epoch_time` | Tiempo por época | Detectar cuellos de botella |

**Interpretación del gap de generalización a lo largo del tiempo:**

```
Gap = val_loss - train_loss

  Gap
  │    /
  │   /  ← creciente: overfitting progresivo
  │  /
  │ /
  │──── estable: equilibrio saludable
  │
  └────────────── épocas

Señales de alarma:
• Gap > 0.15 y creciente → overfitting severo
• Gap oscilante fuertemente → batch size muy pequeño
• Gap < 0 → el modelo puede necesitar más capacidad
```

**Señales de desvanecimiento de gradiente (Vanishing Gradient):** Si la pérdida de entrenamiento deja de disminuir desde las primeras épocas (se "congela" en un valor alto), puede indicar que los gradientes se vuelven cero o infinitesimalmente pequeños en las capas profundas. La solución es revisar las funciones de activación (ReLU en lugar de sigmoid/tanh en capas ocultas), la inicialización de pesos (Xavier/He), o añadir *batch normalization*.

```
Síntomas de problemas comunes durante el entrenamiento:

Problema               │ Síntoma en dashboard             │ Acción
───────────────────────┼──────────────────────────────────┼─────────────────
LR muy alto            │ Pérdida explota o oscila mucho   │ Reducir LR ÷10
LR muy bajo            │ Pérdida no baja en 20+ épocas    │ Aumentar LR ×10
Vanishing gradient     │ Pérdida se congela (no baja)     │ Cambiar activación
Overfitting            │ Gap > 0.15 y creciente           │ L2, Dropout, Early stop
Underfitting           │ Ambas pérdidas altas             │ Más épocas o modelo mayor
Data leakage           │ Val < Train (val mejor que train)│ Revisar preprocesamiento
```

**¿Cuándo intervenir?** Interrumpe el entrenamiento si: (1) la pérdida de entrenamiento no disminuye en las primeras 20 épocas (posible problema de LR o inicialización); (2) el gap de generalización supera 0.2 y sigue creciendo (overfitting severo); (3) la pérdida explota a NaN o infinito (LR demasiado grande o problema numérico). En todos estos casos, intervenir temprano ahorra tiempo de cómputo y permite corregir la configuración.

```python
class TrainingMonitor:
    """Monitor completo de entrenamiento"""
    
    def __init__(self):
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'batch_time': [],
            'epoch_time': []
        }
    
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
        """Actualiza métricas"""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['learning_rate'].append(lr)
        self.metrics['epoch_time'].append(epoch_time)
    
    def print_summary(self):
        """Imprime resumen del entrenamiento"""
        print("\n" + "=" * 70)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("=" * 70)
        
        best_val_idx = np.argmin(self.metrics['val_loss'])
        best_epoch = self.metrics['epoch'][best_val_idx]
        
        print(f"\nMejor época: {best_epoch}")
        print(f"  Train Loss: {self.metrics['train_loss'][best_val_idx]:.4f}")
        print(f"  Val Loss: {self.metrics['val_loss'][best_val_idx]:.4f}")
        print(f"  Train Acc: {self.metrics['train_acc'][best_val_idx]:.4f}")
        print(f"  Val Acc: {self.metrics['val_acc'][best_val_idx]:.4f}")
        
        print(f"\nÚltima época: {self.metrics['epoch'][-1]}")
        print(f"  Train Loss: {self.metrics['train_loss'][-1]:.4f}")
        print(f"  Val Loss: {self.metrics['val_loss'][-1]:.4f}")
        print(f"  Train Acc: {self.metrics['train_acc'][-1]:.4f}")
        print(f"  Val Acc: {self.metrics['val_acc'][-1]:.4f}")
        
        total_time = sum(self.metrics['epoch_time'])
        avg_time = np.mean(self.metrics['epoch_time'])
        print(f"\nTiempo total: {total_time:.2f}s")
        print(f"Tiempo promedio por época: {avg_time:.2f}s")
        
        # Diagnóstico
        print("\n" + "-" * 70)
        self.diagnose()
        print("=" * 70)
    
    def diagnose(self):
        """Diagnostica problemas comunes"""
        train_loss = self.metrics['train_loss'][-1]
        val_loss = self.metrics['val_loss'][-1]
        gap = val_loss - train_loss
        
        print("DIAGNÓSTICO:")
        
        if gap > 0.15:
            print("  ⚠️  OVERFITTING detectado")
            print("      - Val loss >> Train loss")
            print("      - Recomendaciones:")
            print("        * Aumentar regularización (L2, Dropout)")
            print("        * Early stopping con patience menor")
            print("        * Más datos de entrenamiento")
            print("        * Reducir complejidad del modelo")
        
        elif train_loss > 0.5:
            print("  ⚠️  UNDERFITTING detectado")
            print("      - Train loss alto")
            print("      - Recomendaciones:")
            print("        * Aumentar complejidad del modelo")
            print("        * Entrenar más épocas")
            print("        * Ajustar learning rate")
            print("        * Verificar preprocesamiento de datos")
        
        elif abs(gap) < 0.05:
            print("  ✓ BUEN AJUSTE")
            print("      - Train y Val loss similares")
            print("      - Modelo generaliza bien")
        
        # Verificar convergencia
        if len(train_loss) > 10:
            recent_improvement = train_loss[-10] - train_loss[-1]
            if recent_improvement < 0.01:
                print("\n  ℹ️  CONVERGENCIA alcanzada")
                print("      - Pérdida estable en últimas 10 épocas")
    
    def plot_dashboard(self):
        """Visualiza dashboard completo"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Pérdidas
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(self.metrics['epoch'], self.metrics['train_loss'], 
                label='Train', linewidth=2)
        ax1.plot(self.metrics['epoch'], self.metrics['val_loss'], 
                label='Val', linewidth=2)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.set_title('Curvas de Pérdida')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(self.metrics['epoch'], self.metrics['train_acc'], 
                label='Train', linewidth=2)
        ax2.plot(self.metrics['epoch'], self.metrics['val_acc'], 
                label='Val', linewidth=2)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Curvas de Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning Rate
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(self.metrics['epoch'], self.metrics['learning_rate'], 
                linewidth=2, color='green')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Gap (Overfitting indicator)
        ax4 = plt.subplot(2, 3, 4)
        gap = np.array(self.metrics['val_loss']) - np.array(self.metrics['train_loss'])
        ax4.plot(self.metrics['epoch'], gap, linewidth=2, color='red')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.fill_between(self.metrics['epoch'], 0, gap, 
                        where=(gap>0), alpha=0.3, color='red', label='Overfitting')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Val Loss - Train Loss')
        ax4.set_title('Gap de Generalización')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Tiempo por época
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(self.metrics['epoch'], self.metrics['epoch_time'], 
                linewidth=2, color='purple')
        ax5.set_xlabel('Época')
        ax5.set_ylabel('Tiempo (s)')
        ax5.set_title('Tiempo por Época')
        ax5.grid(True, alpha=0.3)
        
        # 6. Resumen numérico
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        best_idx = np.argmin(self.metrics['val_loss'])
        summary_text = f"""
        RESUMEN
        
        Mejor Época: {self.metrics['epoch'][best_idx]}
        Mejor Val Loss: {self.metrics['val_loss'][best_idx]:.4f}
        Mejor Val Acc: {self.metrics['val_acc'][best_idx]:.4f}
        
        Final:
        Train Loss: {self.metrics['train_loss'][-1]:.4f}
        Val Loss: {self.metrics['val_loss'][-1]:.4f}
        Gap: {gap[-1]:.4f}
        
        Total Épocas: {len(self.metrics['epoch'])}
        Tiempo Total: {sum(self.metrics['epoch_time']):.1f}s
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=11, 
                family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
```

**Actividad 4.1:** Usa el monitor para entrenar un modelo y analiza el dashboard completo.

> **¿Qué debes observar y documentar?** Analiza los seis paneles del dashboard sistemáticamente: (1) en las curvas de pérdida, identifica en qué época el modelo alcanza su mejor rendimiento de validación; (2) en las curvas de exactitud, verifica que la exactitud de validación no empieza a degradarse mientras la de entrenamiento sigue subiendo; (3) en el panel de LR, confirma que el scheduler opera como se diseñó; (4) en el gap de generalización, observa si es creciente (overfitting), decreciente (el modelo aún puede aprender) o estable (equilibrio); (5) en el tiempo por época, comprueba que no hay variaciones inesperadas. Escribe un párrafo de diagnóstico usando el vocabulario técnico aprendido: overfitting, underfitting, convergencia, generalización.

## 📊 Análisis Final de Rendimiento

### Experimento Completo: Comparación de Técnicas

#### Fundamento Teórico: Experimentación Controlada en Machine Learning

Un **experimento controlado** en ML sigue los mismos principios del método científico: se varía **una sola variable independiente** a la vez (la técnica de entrenamiento) manteniendo todo lo demás constante (arquitectura del modelo, dataset, semilla aleatoria, número de épocas). La función `run_experiment` implementa exactamente este diseño: crea un modelo fresco con la misma arquitectura e inicialización en cada experimento, garantizando que las diferencias en resultados se deben exclusivamente a la técnica evaluada.

```
Diseño de experimento controlado:

Variable controlada: técnica de entrenamiento
Variables fijas: arquitectura, datos, semilla aleatoria

  Experimento 1: Baseline (SGD simple)        ─┐
  Experimento 2: + Mini-batches (batch=32)     ├── Misma arquitectura
  Experimento 3: + Early stopping              ├── Mismo dataset
  Experimento 4: + Regularización L2           ├── Misma semilla aleatoria
  Experimento 5: + LR scheduling               ─┘

  Comparar en: val_acc, val_loss, gap, tiempo
```

**¿Por qué comparar múltiples configuraciones?** Ninguna técnica es universalmente superior: la efectividad del early stopping, la regularización L2 y el LR scheduling depende del dataset específico, la arquitectura, y el nivel de ruido de los datos. Comparar sistemáticamente permite: (a) cuantificar el beneficio marginal de cada técnica en el problema concreto, (b) identificar si técnicas adicionales generan mejora o complejidad innecesaria, y (c) desarrollar intuición sobre qué técnicas funcionan mejor en qué contextos.

**Cómo extraer conclusiones válidas:**

| Principio | Descripción |
|-----------|-------------|
| Una variable a la vez | Solo cambiar la técnica, no la arquitectura |
| Múltiples semillas | Repetir 3-5 veces para estimar varianza |
| Evaluar en test set | Nunca en validación para comparar |
| Reportar media ± std | No solo el mejor resultado obtenido |
| Contexto importa | Una técnica puede ganar en un dataset y perder en otro |

Para que las comparaciones sean estadísticamente significativas, es buena práctica repetir cada experimento con múltiples semillas aleatorias y reportar la media ± desviación estándar del rendimiento. Un único experimento puede dar resultados favorables o desfavorables por puro azar. Además, la comparación debe hacerse siempre en el conjunto de **test** (nunca en validación), y todas las decisiones de diseño deben haberse tomado sin consultar el test set.

```python
import time

def run_experiment(model_fn, X_train, y_train, X_val, y_val, 
                   config_name, **kwargs):
    """Ejecuta un experimento completo de entrenamiento"""
    print(f"\n{'='*70}")
    print(f"Experimento: {config_name}")
    print(f"{'='*70}")
    
    # Crear modelo fresco
    model = model_fn()
    
    # Crear trainer
    trainer = kwargs.get('trainer_class', SimpleTrainer)(model, **kwargs.get('trainer_params', {}))
    
    # Entrenar
    start_time = time.time()
    history = trainer.train(X_train, y_train, X_val, y_val, 
                           epochs=kwargs.get('epochs', 100),
                           verbose=False)
    end_time = time.time()
    
    # Resultados
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    print(f"\nResultados:")
    print(f"  Train Loss: {final_train_loss:.4f} | Accuracy: {final_train_acc:.4f}")
    print(f"  Val Loss: {final_val_loss:.4f} | Accuracy: {final_val_acc:.4f}")
    print(f"  Gap: {final_val_loss - final_train_loss:.4f}")
    print(f"  Tiempo: {end_time - start_time:.2f}s")
    
    return {
        'config': config_name,
        'history': history,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'train_acc': final_train_acc,
        'val_acc': final_val_acc,
        'time': end_time - start_time
    }

# Ejecutar múltiples experimentos
results = []

# Experimento 1: Baseline
results.append(run_experiment(
    create_model, X_train, y_train, X_val, y_val,
    "Baseline",
    trainer_class=SimpleTrainer,
    trainer_params={'learning_rate': 0.01},
    epochs=100
))

# Experimento 2: Con mini-batches
results.append(run_experiment(
    create_model, X_train, y_train, X_val, y_val,
    "Mini-batch SGD (batch=32)",
    trainer_class=BatchTrainer,
    trainer_params={'learning_rate': 0.01, 'batch_size': 32},
    epochs=100
))

# Experimento 3: Con early stopping
results.append(run_experiment(
    create_model, X_train, y_train, X_val, y_val,
    "Early Stopping (patience=10)",
    trainer_class=TrainerWithEarlyStopping,
    trainer_params={'learning_rate': 0.01, 'batch_size': 32},
    epochs=200  # Más épocas pero con early stopping
))

# Comparar resultados
print("\n" + "="*70)
print("COMPARACIÓN DE EXPERIMENTOS")
print("="*70)

for result in results:
    print(f"\n{result['config']}:")
    print(f"  Val Accuracy: {result['val_acc']:.4f}")
    print(f"  Val Loss: {result['val_loss']:.4f}")
    print(f"  Gap: {result['val_loss'] - result['train_loss']:.4f}")
    print(f"  Tiempo: {result['time']:.2f}s")
```

## 🎯 EJERCICIOS PROPUESTOS

### Nivel Básico

**Ejercicio 1:** Loop de Entrenamiento Básico
```
Implementa un loop de entrenamiento desde cero para:
- Clasificación binaria en dataset sintético
- Mostrar progreso cada 10 épocas
- Graficar curvas de aprendizaje
```

**Ejercicio 2:** División de Datos
```
Dado un dataset, implementa:
- División train/val/test (70/15/15)
- Normalización apropiada
- Verificación de distribución de clases
```

**Ejercicio 3:** Batch Processing
```
Compara el entrenamiento con diferentes batch sizes:
- Batch completo (batch size = tamaño del dataset)
- Mini-batch (32, 64, 128)
- Stochastic (batch size = 1)
Analiza tiempo y convergencia.
```

### Nivel Intermedio

**Ejercicio 4:** Early Stopping
```
Implementa early stopping con:
- Patience configurable
- Guardado del mejor modelo
- Restauración automática
- Visualización de cuándo se detuvo
```

**Ejercicio 5:** Learning Rate Finder
```
Implementa el método "learning rate range test":
- Incrementa LR exponencialmente
- Grafica pérdida vs LR
- Encuentra el LR óptimo automáticamente
```

**Ejercicio 6:** Regularización
```
Compara modelos con:
- Sin regularización
- L1 regularización
- L2 regularización
- L1 + L2 (Elastic Net)
Analiza impacto en overfitting.
```

### Nivel Avanzado

**Ejercicio 7:** Sistema Completo de Entrenamiento
```
Implementa un sistema con:
- Mini-batch SGD
- Early stopping
- LR scheduling (reduce on plateau)
- Checkpointing
- Logging completo
- Dashboard de visualización
```

**Ejercicio 8:** Optimizadores Avanzados
```
Implementa desde cero:
- SGD con Momentum
- RMSprop
- Adam
Compara convergencia en diferentes problemas.
```

**Ejercicio 9:** K-Fold Cross-Validation
```
Implementa K-fold CV para:
- Evaluar robustez del modelo
- Estimar error de generalización
- Seleccionar hiperparámetros
Promedia resultados de K modelos.
```

## 📝 Entregables

### 1. Código Fuente
- `trainer.py`: Clase principal de entrenamiento
- `early_stopping.py`: Implementación de early stopping
- `lr_scheduler.py`: Schedulers de learning rate
- `regularization.py`: Técnicas de regularización
- `monitor.py`: Sistema de monitoreo
- `experiments.ipynb`: Notebook con experimentos

### 2. Experimentos
- Comparación de batch sizes
- Análisis de early stopping
- Evaluación de regularización
- Comparación de LR schedules
- Resultados en datasets reales

### 3. Visualizaciones
- Curvas de aprendizaje
- Dashboards de entrenamiento
- Comparaciones de configuraciones
- Análisis de convergencia

### 4. Reporte (3-4 páginas)
- Metodología de experimentación
- Resultados y análisis
- Conclusiones sobre mejores prácticas
- Recomendaciones para diferentes escenarios

## 🎯 Criterios de Evaluación (CDIO)

### Conceive (Concebir) - 25%
- [ ] Comprensión del proceso de entrenamiento completo
- [ ] Identificación de hiperparámetros clave
- [ ] Diseño de experimentos apropiados
- [ ] Planificación de estrategias de validación

### Design (Diseñar) - 25%
- [ ] Implementación correcta del loop de entrenamiento
- [ ] Código modular y extensible
- [ ] Sistema de monitoreo efectivo
- [ ] Manejo apropiado de datos

### Implement (Implementar) - 30%
- [ ] Early stopping funciona correctamente
- [ ] Regularización reduce overfitting
- [ ] LR scheduling mejora convergencia
- [ ] Resultados reproducibles

### Operate (Operar) - 20%
- [ ] Experimentos bien diseñados
- [ ] Análisis crítico de resultados
- [ ] Comparaciones significativas
- [ ] Documentación clara

## 📋 Rúbrica de Evaluación

| Criterio | Excelente (90-100%) | Bueno (75-89%) | Satisfactorio (60-74%) | Insuficiente (<60%) |
|----------|-------------------|--------------|---------------------|------------------|
| **Loop Entrenamiento** | Completo, robusto, eficiente | Funcional y correcto | Básico pero funciona | Errores o incompleto |
| **Early Stopping** | Implementado perfectamente | Funciona bien | Implementación básica | No funciona |
| **Regularización** | Múltiples técnicas, bien aplicadas | Al menos una técnica | Intentado pero limitado | No implementado |
| **Experimentos** | Extensivos, bien diseñados | Buenos experimentos | Experimentos básicos | Experimentos insuficientes |
| **Análisis** | Profundo, insights valiosos | Buen análisis | Análisis superficial | Análisis pobre |

## 📚 Referencias Adicionales

### Papers Fundamentales
1. Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"
2. Hinton, G. et al. (2012). "Improving neural networks by preventing co-adaptation of feature detectors" (Dropout)
3. Ioffe, S. & Szegedy, C. (2015). "Batch Normalization"
4. Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks"

### Recursos Online
- **Deep Learning Book** (Goodfellow): Capítulo 8 - Optimization
- **CS231n Stanford**: Notas sobre entrenamiento de redes neuronales
- **Fast.ai**: Practical Deep Learning for Coders
- **Distill.pub**: Visualizaciones interactivas sobre optimización

### Herramientas
- Scikit-learn: train_test_split, cross_validation
- TensorBoard: Visualización de entrenamiento
- Weights & Biases: Tracking de experimentos

## 🎓 Notas Finales

### Mejores Prácticas

1. **Siempre normaliza tus datos**: Hace que el entrenamiento sea más estable y rápido.

2. **Usa early stopping**: Previene overfitting y ahorra tiempo de cómputo.

3. **Monitorea train y validation**: La relación entre ambas te dice mucho sobre tu modelo.

4. **Empieza simple**: Baseline simple primero, luego añade complejidad.

5. **Guarda checkpoints**: Nunca sabes cuándo necesitarás volver a un modelo anterior.

### Errores Comunes

❌ **No mezclar datos antes de crear batches**: Lleva a mal entrenamiento
❌ **Usar datos de test para early stopping**: ¡Test debe ser intocable!
❌ **No normalizar datos**: Entrenamiento inestable
❌ **Learning rate muy alto**: Divergencia
❌ **No monitorear validation**: No detectas overfitting

### Checklist de Entrenamiento

Antes de entrenar:
- [ ] Datos normalizados
- [ ] División train/val/test correcta
- [ ] Batch size razonable (16-128)
- [ ] Learning rate inicial apropiado (0.001-0.01)
- [ ] Early stopping configurado
- [ ] Monitoreo activado

Durante entrenamiento:
- [ ] Verificar que pérdida baja
- [ ] Monitorear gap train-val
- [ ] Observar convergencia
- [ ] Verificar tiempo por época

Después de entrenar:
- [ ] Evaluar en test set
- [ ] Analizar errores
- [ ] Guardar modelo
- [ ] Documentar configuración

### Reflexión Final

**El entrenamiento es donde la teoría se encuentra con la práctica**. Puedes tener el mejor algoritmo del mundo, pero sin un buen proceso de entrenamiento, no funcionará.

Las técnicas que aprendiste aquí:
- Son usadas en TODOS los modelos de producción
- Son la diferencia entre 80% y 95% de accuracy
- Te permiten diagnosticar y solucionar problemas
- Son transferibles a cualquier framework (PyTorch, TensorFlow)

### Próximos Pasos

En el siguiente laboratorio (Lab 07), aprenderás:
- Métricas de evaluación detalladas
- Matriz de confusión
- Precision, Recall, F1-Score
- ROC curves y AUC
- Análisis de errores sistemático

¡El entrenamiento es donde todo cobra vida! 🚀

---

**"In theory, there is no difference between theory and practice. In practice, there is."** - Yogi Berra

**¡El entrenamiento es donde todo cobra vida! 🚀**
