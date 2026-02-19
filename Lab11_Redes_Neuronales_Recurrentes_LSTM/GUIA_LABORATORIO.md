# Guía de Laboratorio: Redes Neuronales Recurrentes y LSTM

## 📋 Información del Laboratorio

**Título:** Redes Neuronales Recurrentes, LSTM y Secuencias  
**Código:** Lab 11  
**Duración:** 3-4 horas  
**Nivel:** Avanzado  

---

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender la arquitectura fundamental de las Redes Neuronales Recurrentes (RNN)
2. Implementar una celda RNN desde cero usando NumPy
3. Explicar y aplicar Backpropagation Through Time (BPTT)
4. Identificar el problema del gradiente desvaneciente/explosivo en RNNs
5. Implementar la arquitectura LSTM con sus tres puertas (forget, input, output)
6. Comparar RNN, LSTM y GRU en términos de capacidad y eficiencia
7. Construir modelos para clasificación de texto con PyTorch
8. Aplicar RNNs/LSTMs a predicción de series de tiempo
9. Implementar LSTMs bidireccionales y apiladas (Stacked LSTM)
10. Entender la arquitectura Encoder-Decoder para secuencias de longitud variable
11. Aplicar técnicas de regularización específicas para modelos recurrentes (Dropout, Gradient Clipping)
12. Evaluar y comparar el rendimiento de distintas arquitecturas recurrentes
13. Conectar conceptos de RNNs con la motivación de los mecanismos de atención y Transformers (Lab 12)

---

## 📚 Prerrequisitos

### Conocimientos

- Python avanzado y NumPy (Labs 01-07)
- Backpropagation y descenso de gradiente (Lab 05)
- Frameworks: PyTorch (Lab 08)
- Redes Convolucionales y técnicas de regularización (Lab 10)
- Cálculo matricial básico (derivadas parciales, regla de la cadena)

### Software

- Python 3.8+
- PyTorch 1.9+ (`pip install torch`)
- NumPy, Matplotlib (`pip install numpy matplotlib`)
- scikit-learn (`pip install scikit-learn`)
- pandas (`pip install pandas`)

### Material de Lectura

Antes de comenzar, revisa:
- `teoria.md` - Fundamentos de RNNs, LSTM, GRU y BPTT
- `README.md` - Recursos y estructura del laboratorio
- Hochreiter & Schmidhuber (1997): *Long Short-Term Memory* (paper original)

---

## 📖 Introducción

### El Problema de las Secuencias

En los laboratorios anteriores construiste redes neuronales que procesan entradas de tamaño fijo: una imagen, un vector de características, una muestra puntual. Sin embargo, el mundo real está lleno de **datos secuenciales** donde el **orden importa**:

- **Texto**: "El banco estaba lleno" — ¿banco de un río o banco financiero? El contexto previo importa.
- **Audio**: El fonema que escuchas depende del fonema anterior.
- **Series de tiempo**: El precio de una acción mañana depende de los precios históricos.
- **Video**: El fotograma actual tiene sentido en función de los fotogramas previos.

Una red neuronal densa (Fully Connected) no puede manejar esto directamente:

```
Problema 1: Tamaño fijo de entrada
  - Oración de 5 palabras ≠ Oración de 20 palabras
  - ¿Cómo unificar la dimensión de entrada?

Problema 2: Sin memoria
  - Cada forward pass es independiente
  - La red "olvida" todo lo que procesó antes

Problema 3: No comparte parámetros en el tiempo
  - Aprende que "rojo" es un color en posición 1
  - No generaliza que "rojo" es un color en posición 10
```

### La Solución: Redes Recurrentes

Las **RNNs (Redes Neuronales Recurrentes)** resuelven esto manteniendo un **estado oculto** `h_t` que actúa como memoria:

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

La red procesa un elemento de la secuencia a la vez, actualizando su estado oculto con cada nueva entrada. Así, cuando procesa `x_t`, ya "sabe" lo que vio en `x_0, x_1, ..., x_{t-1}`.

### Tipos de Problemas de Secuencias

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **One-to-One** | Vector → Vector | Clasificación estándar |
| **One-to-Many** | Vector → Secuencia | Image Captioning |
| **Many-to-One** | Secuencia → Vector | Análisis de sentimiento |
| **Many-to-Many (sync)** | Secuencia → Secuencia (igual longitud) | POS Tagging |
| **Many-to-Many (async)** | Secuencia → Secuencia (diferente longitud) | Traducción automática |

### Motivación Histórica y Estado del Arte

Las RNNs tienen raíces en la década de 1980 (Rumelhart et al., 1986). Los LSTM fueron introducidos en 1997 por Hochreiter y Schmidhuber como solución al problema del gradiente desvaneciente. En 2014, Cho et al. propusieron las **GRU (Gated Recurrent Units)**, una simplificación del LSTM con solo dos puertas que reduce el número de parámetros manteniendo una capacidad similar; las GRUs se convirtieron rápidamente en la alternativa más popular al LSTM. Durante los años 2010, LSTM y GRU dominaron tareas de NLP, reconocimiento de voz y predicción de series de tiempo.

Con la llegada de los **Transformers** (Vaswani et al., 2017, que estudiarás en el Lab 12), las RNNs cedieron terreno en NLP de alta capacidad. Sin embargo, siguen siendo fundamentales para:

- Sistemas embebidos con restricciones de memoria
- Series de tiempo con dependencias causales estrictas
- Modelos que procesan datos en tiempo real (streaming)
- Comprensión profunda de los mecanismos de atención

> **Nota:** Entender RNNs y LSTMs es la base conceptual para entender por qué los Transformers con atención son más poderosos.

---

## 🤔 Preguntas de Reflexión Iniciales

Antes de comenzar el código, reflexiona sobre estas preguntas:

1. Si quieres predecir la temperatura de mañana, ¿cuántos días de historial crees que necesitas? ¿Por qué?
2. ¿Cómo representarías una palabra como entrada numérica para una red neuronal?
3. ¿Por qué multiplicar muchos números entre 0 y 1 puede ser problemático en backpropagation?
4. Si tienes una frase de 100 palabras y necesitas hacer backpropagation a través de todas ellas, ¿qué desafíos computacionales anticipas?
5. ¿En qué se parece la "memoria" de un LSTM a la memoria humana a corto/largo plazo?
6. Si pudieras diseñar una arquitectura para procesar secuencias, ¿qué mecanismo añadirías para que la red "decida" qué recordar y qué olvidar?
7. ¿Por qué crees que los LSTMs Bidireccionales pueden ser útiles? ¿En qué aplicaciones NO podrías usarlos?

---

## 🔬 Parte 1: RNN desde Cero (45 min)

### 1.1 Celda RNN Básica con NumPy

Comenzamos implementando una celda RNN desde cero para entender exactamente qué ocurre en cada paso temporal.

```python
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("IMPLEMENTACIÓN DE RNN DESDE CERO CON NUMPY")
print("=" * 60)

class CeldaRNN:
    """
    Celda RNN básica implementada desde cero.
    
    Ecuaciones:
        h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        y_t = W_hy @ h_t + b_y
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicialización de pesos (Xavier/Glorot)
        scale_xh = np.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale_hy = np.sqrt(2.0 / (hidden_size + output_size))
        
        self.W_xh = np.random.randn(hidden_size, input_size) * scale_xh
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.b_h  = np.zeros((hidden_size, 1))
        
        self.W_hy = np.random.randn(output_size, hidden_size) * scale_hy
        self.b_y  = np.zeros((output_size, 1))
        
        # Cache para backpropagation
        self.cache = {}
    
    def forward_step(self, x_t, h_prev):
        """
        Un paso forward de la celda RNN.
        
        Args:
            x_t:    (input_size, 1) - entrada en tiempo t
            h_prev: (hidden_size, 1) - estado oculto anterior
        
        Returns:
            h_t: (hidden_size, 1) - nuevo estado oculto
            y_t: (output_size, 1) - salida
        """
        # Pre-activación
        a_t = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h
        
        # Activación no lineal
        h_t = np.tanh(a_t)
        
        # Salida
        y_t = self.W_hy @ h_t + self.b_y
        
        return h_t, y_t, a_t
    
    def forward_sequence(self, X):
        """
        Forward pass sobre una secuencia completa.
        
        Args:
            X: (seq_len, input_size, 1) - secuencia de entradas
        
        Returns:
            H: lista de estados ocultos
            Y: lista de salidas
        """
        T = len(X)
        h = np.zeros((self.hidden_size, 1))
        
        H = [h]
        Y = []
        A = []
        
        for t in range(T):
            x_t = X[t]
            h, y_t, a_t = self.forward_step(x_t, h)
            H.append(h)
            Y.append(y_t)
            A.append(a_t)
        
        # Guardar en cache para backprop
        self.cache = {'X': X, 'H': H, 'A': A, 'Y': Y}
        
        return H[1:], Y


# Ejemplo de uso
print("\n--- Ejemplo: RNN con secuencia de 5 pasos ---")
np.random.seed(42)

INPUT_SIZE  = 3
HIDDEN_SIZE = 4
OUTPUT_SIZE = 2
SEQ_LEN     = 5

rnn = CeldaRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# Secuencia de entrada aleatoria
X = [np.random.randn(INPUT_SIZE, 1) for _ in range(SEQ_LEN)]

H, Y = rnn.forward_sequence(X)

print(f"Secuencia de longitud: {SEQ_LEN}")
print(f"Tamaño de entrada (input_size): {INPUT_SIZE}")
print(f"Tamaño del estado oculto (hidden_size): {HIDDEN_SIZE}")
print(f"Tamaño de salida (output_size): {OUTPUT_SIZE}")
print()

for t, (h_t, y_t) in enumerate(zip(H, Y)):
    print(f"t={t}: h_t shape={h_t.shape}, y_t shape={y_t.shape}")
    print(f"       h_t = {h_t.flatten().round(4)}")
    print(f"       y_t = {y_t.flatten().round(4)}")
```

### 1.2 Forward Pass en Secuencias — Visualización

Para comprender intuitivamente cómo una RNN procesa una secuencia, es fundamental **visualizar la red "desenrollada"** (*unfolded*) en el tiempo. Al desenrollar la RNN, lo que conceptualmente es un único módulo con un bucle recurrente se transforma en una cadena de copias idénticas —una por cada paso temporal— donde cada copia comparte exactamente los mismos pesos.

**¿Qué muestra el diagrama?**

| Elemento | Color | Descripción |
|---|---|---|
| **Celda RNN** | Verde | Módulo con pesos compartidos `W_xh`, `W_hh`, `b_h` |
| **Entrada** `x_t` | Azul | Vector de entrada en el instante `t` |
| **Salida** `y_t` | Naranja | Predicción producida en el instante `t` |
| **Estado oculto** `h_t` | Gris | Memoria que fluye de izquierda a derecha entre celdas |

La flecha horizontal `h_t` que conecta cada celda con la siguiente es el corazón de la arquitectura recurrente: **es el mecanismo mediante el cual la información del pasado se transmite al futuro**, permitiendo que la celda del paso `t+1` "recuerde" lo que ocurrió antes.

**Dos tipos fundamentales de tarea secuencial:**

- **Many-to-One** *(muchos a uno)*: la red consume toda la secuencia pero produce **una única salida** al final (por ejemplo, análisis de sentimiento: leer una reseña completa y clasificarla como positiva o negativa). Sólo se utiliza `H[-1]`, el estado oculto del último paso.
- **Many-to-Many** *(muchos a muchos)*: la red produce **una salida en cada paso temporal** (por ejemplo, etiquetado morfosintáctico *Part-of-Speech*: asignar una etiqueta a cada palabra de una frase). Se utilizan todas las salidas `Y[0], Y[1], …, Y[T-1]`.

Al ejecutar el código observa cómo:
- Las **celdas RNN son idénticas** (mismos pesos), lo que hace que el número de parámetros sea independiente de la longitud de la secuencia.
- El **estado oculto crece** de izquierda a derecha, acumulando contexto progresivamente.
- En *Many-to-One* solo la última salida lleva información de toda la secuencia; en *Many-to-Many* cada salida lleva información hasta ese instante.

**Resultados esperados:**

- Se generará y guardará la figura `rnn_desenvuelta.png` con el diagrama de la RNN de 4 pasos temporales.
- En la consola aparecerá la confirmación `"Figura guardada como 'rnn_desenvuelta.png'"`.
- La salida de *Many-to-One* tendrá `shape = (output_size, 1)` (un único vector); la salida de *Many-to-Many* será una lista de `seq_len` vectores del mismo shape, mostrando que la red produjo una predicción por cada elemento de la secuencia.

```python
print("\n--- Visualización del flujo de información en RNN ---")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualizar_rnn_desenvuelta(seq_len=4):
    """Diagrama del RNN desenrollado en el tiempo."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-0.5, seq_len + 0.5)
    ax.set_ylim(-1, 4)
    ax.axis('off')
    ax.set_title("RNN Desenrollada en el Tiempo (Unfolded)", fontsize=14, fontweight='bold')
    
    colors = {'rnn': '#4CAF50', 'input': '#2196F3', 'output': '#FF9800', 'arrow': '#555555'}
    
    for t in range(seq_len):
        # Caja RNN
        rect = mpatches.FancyBboxPatch((t + 0.1, 1.5), 0.8, 0.8,
                                        boxstyle="round,pad=0.05",
                                        facecolor=colors['rnn'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(t + 0.5, 1.9, f"RNN\nt={t}", ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Entrada x_t
        ax.annotate('', xy=(t + 0.5, 1.5), xytext=(t + 0.5, 0.5),
                    arrowprops=dict(arrowstyle='->', color=colors['input'], lw=2))
        ax.text(t + 0.5, 0.2, f"$x_{{{t}}}$", ha='center', va='center', fontsize=11,
                color=colors['input'], fontweight='bold')
        
        # Salida y_t (en modo many-to-many)
        ax.annotate('', xy=(t + 0.5, 3.2), xytext=(t + 0.5, 2.3),
                    arrowprops=dict(arrowstyle='->', color=colors['output'], lw=2))
        ax.text(t + 0.5, 3.5, f"$y_{{{t}}}$", ha='center', va='center', fontsize=11,
                color=colors['output'], fontweight='bold')
        
        # Flecha de estado oculto h_t
        if t < seq_len - 1:
            ax.annotate('', xy=(t + 1.1, 1.9), xytext=(t + 0.9, 1.9),
                        arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2.5))
            ax.text(t + 1.0, 2.1, f"$h_{{{t}}}$", ha='center', va='center', fontsize=9, color=colors['arrow'])
    
    # Leyenda
    patch1 = mpatches.Patch(color=colors['rnn'], label='Celda RNN (pesos compartidos)')
    patch2 = mpatches.Patch(color=colors['input'], label='Entradas $x_t$')
    patch3 = mpatches.Patch(color=colors['output'], label='Salidas $y_t$')
    patch4 = mpatches.Patch(color=colors['arrow'], label='Estado oculto $h_t$')
    ax.legend(handles=[patch1, patch2, patch3, patch4], loc='lower center',
              ncol=4, fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('rnn_desenvuelta.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Figura guardada como 'rnn_desenvuelta.png'")

visualizar_rnn_desenvuelta(seq_len=4)


# -------------------------------------------------------
# Demostración: muchos-a-uno (Many-to-One) vs muchos-a-muchos
# -------------------------------------------------------
print("\n--- Demo: Many-to-One vs Many-to-Many ---")

class RNN_ManyToOne(CeldaRNN):
    """Solo devuelve la salida del último paso temporal."""
    def predict(self, X):
        H, Y = self.forward_sequence(X)
        return Y[-1]   # Última salida


class RNN_ManyToMany(CeldaRNN):
    """Devuelve salida en cada paso temporal."""
    def predict(self, X):
        H, Y = self.forward_sequence(X)
        return Y       # Todas las salidas


rnn_m2o = RNN_ManyToOne(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
rnn_m2m = RNN_ManyToMany(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

y_m2o = rnn_m2o.predict(X)
y_m2m = rnn_m2m.predict(X)

print(f"Many-to-One (análisis de sentimiento): salida shape = {y_m2o.shape}")
print(f"Many-to-Many (etiquetado POS): {len(y_m2m)} salidas, cada una de shape {y_m2m[0].shape}")
```

### 1.3 Backpropagation Through Time (BPTT)

El algoritmo de retropropagación estándar (*backpropagation*) calcula gradientes en redes con una arquitectura estática de capas. Las RNN, sin embargo, procesan secuencias con un grafo computacional que **se extiende en el tiempo**: cada paso temporal `t` depende del estado anterior `h_{t-1}`. Para entrenar una RNN es necesario adaptar la retropropagación a esta estructura temporal, lo que da lugar al algoritmo **Backpropagation Through Time (BPTT)**.

**¿Cómo fluyen los gradientes hacia atrás en el tiempo?**

Dado que el estado oculto en cada paso se calcula como `h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)`, la pérdida total `L` depende de `h_t` tanto de manera directa (a través de la salida `y_t`) como indirecta (a través de todos los pasos futuros). El gradiente del peso `W` acumula contribuciones de **todos los pasos temporales**:

$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial W}$$

En la práctica, el algoritmo recorre la secuencia **al revés** (de `t = T-1` hasta `t = 0`), propagando el gradiente a través de la derivada de la función `tanh`:

```
da_t = dh_t ⊙ (1 - h_t²)       ← derivada de tanh
dh_{t-1} = W_hh.T · da_t        ← propagar hacia atrás en el tiempo
```

**¿Por qué se incluye *gradient clipping*?**

En secuencias largas, la multiplicación repetida por `W_hh` en cada paso puede hacer que la norma del gradiente crezca exponencialmente (*gradient explosion*). La técnica de *gradient clipping* resuelve esto: si la norma total del gradiente supera un umbral `max_norm`, **reescala todos los gradientes proporcionalmente** para que su norma total sea exactamente `max_norm`. Esto estabiliza el entrenamiento sin alterar la dirección del gradiente.

**¿Qué observarás durante el entrenamiento?**

- La pérdida (*cross-entropy*) comenzará aproximadamente en **~0.69** (equivalente a predicción aleatoria con 2 clases: `log(2) ≈ 0.693`).
- A medida que el optimizador SGD ajusta los pesos con BPTT, la pérdida descenderá de forma continua.
- Después de **200 épocas**, la pérdida debería acercarse a **0.0**, indicando que la red ha memorizado la muestra de entrenamiento y asigna prácticamente toda la probabilidad a la clase correcta.

> ⚠️ **Sin *gradient clipping***, en secuencias largas o con pesos mal inicializados, los gradientes pueden volverse `NaN` o `Inf` en pocas iteraciones, impidiendo completamente el aprendizaje. El *clipping* es especialmente importante en RNN entrenadas con BPTT.

**Resultados esperados:**

- La pérdida inicial será aproximadamente **~0.69**.
- La curva de convergencia descenderá suavemente hasta valores cercanos a **0.0** al final de las 200 épocas.
- Se guardará la figura `bptt_convergencia.png` con la curva de entrenamiento.
- Si se activa el *clipping*, verás líneas de la forma `[Clipping] norma=X.XXX → escalado por 0.XXXX` indicando que se normalizaron los gradientes.

```python
print("\n--- Backpropagation Through Time (BPTT) ---")

class RNN_BPTT:
    """
    RNN con implementación completa de BPTT.
    Problema: Many-to-One (clasificación de secuencias).
    """
    
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr          = lr
        
        # Inicializar pesos
        self.W_xh = np.random.randn(hidden_size, input_size)  * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h  = np.zeros((hidden_size, 1))
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y  = np.zeros((output_size, 1))
    
    def softmax(self, z):
        e = np.exp(z - np.max(z))
        return e / e.sum()
    
    def forward(self, X):
        """Forward pass completo: devuelve estados, salidas y activaciones."""
        T = len(X)
        H = [np.zeros((self.hidden_size, 1))]   # h_0 = 0
        A = []
        
        for t in range(T):
            a_t = self.W_xh @ X[t] + self.W_hh @ H[t] + self.b_h
            h_t = np.tanh(a_t)
            H.append(h_t)
            A.append(a_t)
        
        # Many-to-One: usar solo el último estado
        logits = self.W_hy @ H[-1] + self.b_y
        y_pred = self.softmax(logits)
        
        return H, A, y_pred
    
    def loss(self, y_pred, y_true):
        """Cross-entropy."""
        return -np.log(y_pred[y_true, 0] + 1e-12)
    
    def backward(self, X, H, A, y_pred, y_true):
        """BPTT: retropropagación a través del tiempo."""
        T = len(X)
        
        # Gradientes iniciales
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h  = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y  = np.zeros_like(self.b_y)
        
        # Gradiente de la pérdida respecto a logits (softmax + cross-entropy)
        dlogits        = y_pred.copy()
        dlogits[y_true] -= 1.0
        
        # Gradientes de la capa de salida
        dW_hy = dlogits @ H[-1].T
        db_y  = dlogits
        
        # Gradiente hacia el último hidden state
        dh = self.W_hy.T @ dlogits
        
        # Retropropagación a través del tiempo
        for t in reversed(range(T)):
            # Gradiente a través de tanh
            da = dh * (1 - H[t+1] ** 2)   # dtanh = 1 - tanh²
            
            # Acumular gradientes de pesos
            dW_xh += da @ X[t].T
            dW_hh += da @ H[t].T
            db_h  += da
            
            # Propagar gradiente hacia el paso anterior
            dh = self.W_hh.T @ da
        
        grads = {'W_xh': dW_xh, 'W_hh': dW_hh, 'b_h': db_h,
                 'W_hy': dW_hy, 'b_y': db_y}
        return grads
    
    def clip_gradients(self, grads, max_norm=5.0):
        """Gradient clipping para evitar explosión."""
        total_norm = 0.0
        for g in grads.values():
            total_norm += np.sum(g ** 2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            factor = max_norm / total_norm
            for key in grads:
                grads[key] *= factor
            print(f"  [Clipping] norma={total_norm:.3f} → escalado por {factor:.4f}")
        
        return grads
    
    def step(self, X, y_true):
        """Un paso de entrenamiento completo."""
        H, A, y_pred = self.forward(X)
        loss = self.loss(y_pred, y_true)
        grads = self.backward(X, H, A, y_pred, y_true)
        grads = self.clip_gradients(grads)
        
        # Actualizar pesos (SGD)
        self.W_xh -= self.lr * grads['W_xh']
        self.W_hh -= self.lr * grads['W_hh']
        self.b_h  -= self.lr * grads['b_h']
        self.W_hy -= self.lr * grads['W_hy']
        self.b_y  -= self.lr * grads['b_y']
        
        return loss, y_pred


# Demostración de un paso de entrenamiento
print("\nDemostración de BPTT (un paso de entrenamiento):")
np.random.seed(0)

rnn_bptt = RNN_BPTT(input_size=3, hidden_size=4, output_size=2, lr=0.01)
X_demo   = [np.random.randn(3, 1) for _ in range(5)]
y_clase  = 1   # Clase objetivo

loss, y_pred = rnn_bptt.step(X_demo, y_clase)
print(f"Pérdida inicial: {loss:.4f}")
print(f"Predicciones (softmax): {y_pred.flatten().round(4)}")

# Entrenar por varias épocas
losses = []
for epoch in range(200):
    loss, _ = rnn_bptt.step(X_demo, y_clase)
    losses.append(float(loss))

print(f"\nPérdida final (200 épocas): {losses[-1]:.6f}")

plt.figure(figsize=(8, 4))
plt.plot(losses, color='steelblue')
plt.xlabel('Época'), plt.ylabel('Cross-Entropy Loss')
plt.title('Convergencia de RNN con BPTT')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('bptt_convergencia.png', dpi=100)
plt.show()
print("Figura guardada como 'bptt_convergencia.png'")
```

---

## 🔬 Parte 2: Problemas de Gradiente y LSTM (45 min)

### 2.1 Demostración del Gradiente que Desaparece

El **problema del gradiente desvaneciente** (*vanishing gradient*) es la principal limitación de las RNN estándar y la razón directa por la que se inventaron las redes LSTM. Para entenderlo, analicemos qué ocurre cuando retropropagamos la señal de error a través de muchos pasos temporales.

**Análisis matemático:**

Al aplicar BPTT, el gradiente respecto al estado oculto `k` pasos atrás involucra el producto de `k` matrices Jacobianas:

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=1}^{k} \frac{\partial h_{t-i+1}}{\partial h_{t-i}} \approx W_{hh}^k$$

El comportamiento de esta expresión depende completamente de la **norma espectral** de `W_hh` (su mayor valor singular `σ_1`):

| Régimen | Condición | Consecuencia |
|---|---|---|
| **Desvaneciente** | `σ_1 < 1` *(‖W‖ < 1)* | `‖W^k‖ → 0` exponencialmente; la señal del pasado remoto es nula |
| **Estable** | `σ_1 ≈ 1` *(‖W‖ ≈ 1)* | La norma del gradiente se mantiene constante; el aprendizaje es posible |
| **Explosivo** | `σ_1 > 1` *(‖W‖ > 1)* | `‖W^k‖ → ∞` exponencialmente; los gradientes se vuelven `NaN` |

En la práctica, los pesos se inicializan aleatoriamente y casi siempre caen en el régimen desvaneciente. Esto significa que **la RNN es incapaz de aprender dependencias que abarquen más de ~10 pasos temporales**: el gradiente que llega desde el paso `t=0` hasta el paso `t=50` es tan pequeño que el optimizador no puede ajustar los pesos con información de largo plazo.

**¿Qué simulará el código?**

La función `simular_flujo_gradiente` inicializa una matriz `W_hh` con tres normas espectrales diferentes (`0.5`, `1.0`, `1.5`) y calcula iterativamente `grad = grad @ W`, midiendo la norma del gradiente en cada paso. Esto reproduce fielmente el comportamiento de `W^k` sin necesidad de hacer un forward pass completo.

La gráfica usará **escala logarítmica en el eje Y** para que las tres curvas sean visibles simultáneamente, ya que sus rangos de variación difieren en muchos órdenes de magnitud.

**Resultados esperados:**

- **Curva roja** *(escala=0.5, desvaneciente)*: la norma caerá de `1.0` a valores próximos a **`1e-15`** o menores después de 50 pasos —prácticamente cero.
- **Curva verde** *(escala=1.0, estable)*: la norma se mantendrá relativamente constante alrededor de **`1.0`** a lo largo de todos los pasos.
- **Curva naranja** *(escala=1.5, explosivo)*: la norma crecerá hasta valores del orden de **`1e6`** o superiores, representando una explosión de gradiente.
- El análisis cuantitativo imprimirá el *ratio* `norma_final / norma_inicial` para cada caso, confirmando los órdenes de magnitud esperados.
- Se guardará la figura `gradiente_desvaneciente.png`.

> 💡 Este experimento explica por qué las LSTM fueron diseñadas con una **celda de memoria** y **puertas de olvido**: en lugar de multiplicar el gradiente por la misma matriz `W_hh` en cada paso, las LSTM permiten que el gradiente fluya hacia atrás casi sin atenuación a través de la *constant error carousel* (CEC).

```python
print("=" * 60)
print("GRADIENTE DESVANECIENTE EN RNN ESTÁNDAR")
print("=" * 60)

def simular_flujo_gradiente(T=50, hidden_size=10, seed=42):
    """
    Simula cómo el gradiente se propaga (o desaparece)
    hacia atrás en el tiempo para diferentes normas espectrales de W_hh.
    """
    np.random.seed(seed)
    resultados = {}
    
    for etiqueta, escala in [("Gradiente desvaneciente (escala=0.5)", 0.5),
                              ("Estable (escala=1.0)", 1.0),
                              ("Gradiente explosivo (escala=1.5)", 1.5)]:
        # Crear W_hh con norma espectral controlada
        W = np.random.randn(hidden_size, hidden_size)
        W = W / np.linalg.norm(W, 2) * escala   # norma espectral = escala
        
        normas = []
        grad = np.eye(hidden_size)   # Gradiente inicial (identidad)
        
        for t in range(T):
            # ∂h_t/∂h_{t-k} ≈ W_hh^k (simplificado, ignorando tanh)
            normas.append(np.linalg.norm(grad))
            grad = grad @ W
        
        resultados[etiqueta] = normas
    
    return resultados

resultados = simular_flujo_gradiente(T=50)

plt.figure(figsize=(10, 5))
colores = ['#e74c3c', '#27ae60', '#e67e22']
for (etiqueta, normas), color in zip(resultados.items(), colores):
    plt.plot(normas, label=etiqueta, color=color, linewidth=2)

plt.xlabel('Pasos hacia atrás en el tiempo', fontsize=12)
plt.ylabel('Norma del gradiente', fontsize=12)
plt.title('Problema del Gradiente Desvaneciente/Explosivo en RNN', fontsize=13)
plt.legend(fontsize=10)
plt.yscale('log')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('gradiente_desvaneciente.png', dpi=100)
plt.show()
print("Figura guardada como 'gradiente_desvaneciente.png'")

# Análisis cuantitativo
print("\n--- Análisis del gradiente después de 50 pasos ---")
for etiqueta, normas in resultados.items():
    ratio = normas[-1] / normas[0] if normas[0] > 0 else 0
    print(f"{etiqueta}")
    print(f"  Norma inicial: {normas[0]:.4f} | Norma final: {normas[-1]:.2e} | Ratio: {ratio:.2e}\n")


# Demostración práctica: RNN no puede aprender dependencias largas
print("--- RNN vs dependencias de largo plazo ---")

def crear_tarea_long_range(n=1000, seq_len=20, gap=15):
    """
    Tarea de clasificación donde la clase depende del primer elemento,
    pero la predicción se hace en el último paso (gap pasos de distancia).
    """
    X, y = [], []
    for _ in range(n):
        seq   = np.random.randn(seq_len, 1)
        clase = int(seq[0, 0] > 0)   # Clase basada en el PRIMER elemento
        X.append(seq)
        y.append(clase)
    return X, y

X_lr, y_lr = crear_tarea_long_range(n=500, seq_len=20, gap=15)
print(f"Tarea de dependencia larga: {len(X_lr)} secuencias, longitud {len(X_lr[0])}")
print("La clase depende del primer elemento pero se predice en el último paso.")
print("→ Las RNN estándar fallan en esta tarea con gap grande.")
print("→ Los LSTM están diseñados para resolverla.\n")
```

### 2.2 Arquitectura LSTM: Las Tres Puertas

La **Long Short-Term Memory (LSTM)** fue propuesta por Hochreiter y Schmidhuber en 1997 precisamente para solucionar el problema del gradiente desvaneciente. Su innovación central es el **estado de celda** $C_t$, que actúa como una "cinta transportadora" de información que atraviesa toda la secuencia con modificaciones mínimas y controladas. A diferencia de la RNN estándar, donde el gradiente se multiplica por $W_{hh}$ en cada paso y se atenúa exponencialmente, el gradiente en la LSTM puede fluir hacia atrás a través de $C_t$ casi sin atenuación gracias al mecanismo de puertas.

#### Las tres puertas y el flujo de información

Cada puerta aplica una función sigmoide $\sigma(\cdot)$ que produce valores en $[0, 1]$, actuando como un **interruptor suave**: un valor cercano a 0 "cierra" el paso de información y un valor cercano a 1 lo "abre" completamente. Esto permite al modelo aprender, de forma diferenciable, qué información conservar y cuál descartar.

**1. Puerta de olvido** $f_t$ — *¿Qué información del pasado ya no es relevante?*

$$f_t = \sigma(W_f \cdot [h_{t-1},\, x_t] + b_f)$$

Produce un vector en $(0,1)^H$. Cada componente indica cuánto del estado de celda anterior $C_{t-1}$ se conserva: $f_t \approx 0$ descarta completamente esa dimensión; $f_t \approx 1$ la preserva intacta.

**2. Puerta de entrada** $i_t$ y candidato $\tilde{C}_t$ — *¿Qué información nueva merece almacenarse?*

$$i_t = \sigma(W_i \cdot [h_{t-1},\, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1},\, x_t] + b_C)$$

$\tilde{C}_t$ propone nuevos valores candidatos (en $[-1,1]^H$) y $i_t$ selecciona cuáles de ellos se escriben realmente en el estado de celda. La actualización del estado de celda combina ambas puertas:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Esta ecuación es la clave del éxito de la LSTM: el gradiente $\partial C_t / \partial C_{t-1} = f_t$ es multiplicativo pero **aprendido**, y cuando $f_t \approx 1$ el gradiente fluye sin atenuación.

**3. Puerta de salida** $o_t$ — *¿Qué parte del estado de celda debe emitirse como salida?*

$$o_t = \sigma(W_o \cdot [h_{t-1},\, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

El estado oculto $h_t$ es la representación que se pasa al siguiente paso y a las capas superiores. El $\tanh$ comprime $C_t$ a $[-1,1]$ antes de filtrarlo con $o_t$, de modo que $h_t$ contiene sólo la información relevante para la salida en el instante $t$.

**Resultados esperados:**

Al ejecutar el código se mostrarán en consola los valores numéricos de $f_t$, $i_t$, $\tilde{C}_t$, $C_t$, $o_t$ y $h_t$ para un estado de ejemplo con `hidden_size=4`. La gráfica generada (`lstm_gates.png`) mostrará **cuatro paneles de barras** de colores:
- 🔴 **Forget Gate** ($f_t$): valores entre 0 y 1 — indican qué dimensiones del estado anterior se conservan.
- 🟢 **Input Gate** ($i_t$): valores entre 0 y 1 — indican qué dimensiones nuevas se escriben.
- 🔵 **Cell State** ($C_t$): puede tomar valores positivos y negativos — es la "memoria" acumulada.
- 🟡 **Output Gate** ($o_t$): valores entre 0 y 1 — filtran lo que se publica como $h_t$.

```python
print("=" * 60)
print("ARQUITECTURA LSTM: FORGET, INPUT Y OUTPUT GATES")
print("=" * 60)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def visualizar_gates_lstm():
    """Visualiza el rol de cada puerta LSTM con un ejemplo numérico."""
    np.random.seed(7)
    
    # Estado anterior y entrada
    h_prev = np.array([0.8, -0.3, 0.5, 0.2])
    C_prev = np.array([1.0,  0.5, -0.7, 0.3])
    x_t    = np.array([0.1, -0.2, 0.9, -0.4])
    
    # Concatenar [h_{t-1}, x_t]
    concat = np.concatenate([h_prev, x_t])
    d = len(concat)
    hidden = 4
    
    # Pesos aleatorios para las puertas
    W_f = np.random.randn(hidden, d) * 0.5
    b_f = np.zeros(hidden)
    W_i = np.random.randn(hidden, d) * 0.5
    b_i = np.zeros(hidden)
    W_C = np.random.randn(hidden, d) * 0.5
    b_C = np.zeros(hidden)
    W_o = np.random.randn(hidden, d) * 0.5
    b_o = np.zeros(hidden)
    
    # ---- Forget Gate ----
    f_t = sigmoid(W_f @ concat + b_f)
    print("🔴 Forget Gate (f_t):")
    print(f"   f_t = σ(W_f · [h_{{t-1}}, x_t] + b_f)")
    print(f"   f_t = {f_t.round(4)}")
    print(f"   → Valores cercanos a 0 = OLVIDAR, cercanos a 1 = RECORDAR")
    print(f"   → Ejemplo: f[2]={f_t[2]:.3f} significa que la celda 2 {'se olvida' if f_t[2] < 0.5 else 'se recuerda'}\n")
    
    # ---- Input Gate ----
    i_t   = sigmoid(W_i @ concat + b_i)
    C_til = np.tanh(W_C @ concat + b_C)
    print("🟢 Input Gate (i_t) y Candidato (C̃_t):")
    print(f"   i_t = σ(W_i · [h_{{t-1}}, x_t] + b_i)")
    print(f"   C̃_t = tanh(W_C · [h_{{t-1}}, x_t] + b_C)")
    print(f"   i_t = {i_t.round(4)}")
    print(f"   C̃_t = {C_til.round(4)}")
    print(f"   → La entrada nueva filtrada: {(i_t * C_til).round(4)}\n")
    
    # ---- Cell State Update ----
    C_t = f_t * C_prev + i_t * C_til
    print("🔵 Actualización del Cell State:")
    print(f"   C_t = f_t ⊙ C_{{t-1}} + i_t ⊙ C̃_t")
    print(f"   C_prev = {C_prev.round(4)}")
    print(f"   C_t    = {C_t.round(4)}")
    print(f"   → Parte recordada:   {(f_t * C_prev).round(4)}")
    print(f"   → Parte nueva añadida: {(i_t * C_til).round(4)}\n")
    
    # ---- Output Gate ----
    o_t = sigmoid(W_o @ concat + b_o)
    h_t = o_t * np.tanh(C_t)
    print("🟡 Output Gate (o_t) y nuevo Hidden State (h_t):")
    print(f"   o_t = σ(W_o · [h_{{t-1}}, x_t] + b_o)")
    print(f"   h_t = o_t ⊙ tanh(C_t)")
    print(f"   o_t = {o_t.round(4)}")
    print(f"   h_t = {h_t.round(4)}\n")
    
    return f_t, i_t, C_t, o_t, h_t

f_t, i_t, C_t, o_t, h_t = visualizar_gates_lstm()

# Visualización de las puertas
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
nombres = ['Forget Gate\n$f_t$', 'Input Gate\n$i_t$', 'Cell State\n$C_t$', 'Output Gate\n$o_t$']
datos   = [f_t, i_t, C_t, o_t]
colores = ['#e74c3c', '#27ae60', '#3498db', '#f39c12']

for ax, nombre, dato, color in zip(axes, nombres, datos, colores):
    bars = ax.bar(range(len(dato)), dato, color=color, alpha=0.8, edgecolor='black')
    ax.set_title(nombre, fontsize=11)
    ax.set_xlabel('Dimensión')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(alpha=0.3)

plt.suptitle('Valores de las Puertas LSTM en un Paso Temporal', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('lstm_gates.png', dpi=100)
plt.show()
print("Figura guardada como 'lstm_gates.png'")
```

### 2.3 Celda LSTM Completa desde Cero

En esta sección construimos una celda LSTM **completamente funcional desde cero** utilizando únicamente NumPy, sin ninguna biblioteca de deep learning. El objetivo es consolidar la comprensión de las ecuaciones vistas en la sección anterior y verificar que las dimensiones de los tensores son correctas antes de usar implementaciones optimizadas como las de PyTorch.

#### Decisiones de diseño de la implementación

La clase `CeldaLSTM` encapsula todos los parámetros y la lógica forward de la celda. Se destacan tres decisiones relevantes:

1. **Concatenación $[h_{t-1},\, x_t]$**: en lugar de mantener matrices de pesos separadas para la entrada ($W_x \in \mathbb{R}^{H \times D_{in}}$) y el estado oculto ($W_h \in \mathbb{R}^{H \times H}$), se concatenan ambos vectores y se utiliza una sola matriz $W \in \mathbb{R}^{H \times (H + D_{in})}$ por puerta. Esto es algebraicamente equivalente pero más eficiente en implementación y hardware.

2. **Inicialización con escala $\sqrt{1/D}$**: limita la magnitud de las pre-activaciones al inicio del entrenamiento, evitando saturación inmediata de las sigmoides y el $\tanh$.

3. **Sesgo de olvido inicializado en $+1$** (Jozefowicz et al., 2015): inicializar $b_f = 1$ hace que $f_t \approx \sigma(1) \approx 0.73$ al principio, es decir, la celda **recuerda por defecto**. Esto es crucial en las primeras iteraciones del entrenamiento, cuando los gradientes son ruidosos y es preferible no perder información del estado de celda.

#### Las dos funciones principales

- **`forward_step(x_t, h_prev, C_prev)`**: ejecuta un único paso temporal aplicando las cuatro ecuaciones de la LSTM y devuelve $(h_t, C_t, \text{cache})$. El `cache` almacena todos los valores intermedios necesarios para el backpropagation.

- **`forward_sequence(X)`**: itera sobre una secuencia de longitud $T$, llamando a `forward_step` en cada instante y acumulando los históricos de $h_t$ y $C_t$. Devuelve listas con todos los estados para su posterior análisis.

#### Conteo de parámetros

Para una LSTM con `input_size=D` y `hidden_size=H`, cada una de las **4 puertas** tiene una matriz $W \in \mathbb{R}^{H \times (D+H)}$ y un sesgo $b \in \mathbb{R}^H$:

$$\text{Parámetros}_{\text{LSTM}} = 4 \times \bigl((D + H) \times H + H\bigr)$$

Para los valores del código (`input_size=5`, `hidden_size=8`): $4 \times ((5+8) \times 8 + 8) = 4 \times (104 + 8) = \mathbf{448}$ parámetros — exactamente **4 veces más** que una RNN estándar equivalente.

**Resultados esperados:**

- La línea `Parámetros totales: 448` confirma el conteo teórico para `input_size=5`, `hidden_size=8`.
- Para los primeros 4 pasos temporales se imprimirán los valores de $f_t$, $i_t$, $o_t$, $C_t$ y $h_t$. Se puede observar que:
  - Los valores de $f_t$ son mayoritariamente > 0.5 (efecto del sesgo de olvido $b_f = 1$).
  - $C_t$ crece en magnitud conforme la celda acumula información a lo largo de los pasos.
  - $h_t$ permanece en $(-1, 1)$ gracias al $\tanh$ aplicado sobre $C_t$.
- La comparación final mostrará que la LSTM tiene aproximadamente **4× más parámetros** que una RNN equivalente con los mismos `input_size`, `hidden_size` y `output_size`.

```python
print("\n--- Implementación completa de LSTM desde cero ---")

class CeldaLSTM:
    """
    Celda LSTM completa implementada con NumPy.
    
    Ecuaciones:
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
        C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
        h_t = o_t ⊙ tanh(C_t)
    """
    
    def __init__(self, input_size, hidden_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        D = input_size + hidden_size
        H = hidden_size
        
        # Inicialización ortogonal para estabilidad
        def init_gate(D, H):
            W = np.random.randn(H, D) * np.sqrt(1.0 / D)
            b = np.zeros(H)
            return W, b
        
        self.W_f, self.b_f = init_gate(D, H)
        self.W_i, self.b_i = init_gate(D, H)
        self.W_C, self.b_C = init_gate(D, H)
        self.W_o, self.b_o = init_gate(D, H)
        
        # El forget gate se inicializa con sesgo positivo (recordar por defecto)
        self.b_f += 1.0
    
    def forward_step(self, x_t, h_prev, C_prev):
        """
        Un paso forward de la celda LSTM.
        
        Args:
            x_t:    (input_size,) - entrada en tiempo t
            h_prev: (hidden_size,) - hidden state anterior
            C_prev: (hidden_size,) - cell state anterior
        
        Returns:
            h_t, C_t, cache - nuevos estados y cache para backprop
        """
        # Concatenar entrada y estado oculto anterior
        concat = np.concatenate([h_prev, x_t])
        
        # Calcular las tres puertas y el candidato
        f_t   = sigmoid(self.W_f @ concat + self.b_f)   # Forget gate
        i_t   = sigmoid(self.W_i @ concat + self.b_i)   # Input gate
        C_til = np.tanh(self.W_C @ concat + self.b_C)   # Candidato
        o_t   = sigmoid(self.W_o @ concat + self.b_o)   # Output gate
        
        # Actualizar cell state y hidden state
        C_t = f_t * C_prev + i_t * C_til
        h_t = o_t * np.tanh(C_t)
        
        # Guardar para backprop
        cache = {
            'x_t': x_t, 'h_prev': h_prev, 'C_prev': C_prev,
            'f_t': f_t, 'i_t': i_t, 'C_til': C_til,
            'o_t': o_t, 'C_t': C_t, 'h_t': h_t, 'concat': concat
        }
        return h_t, C_t, cache
    
    def forward_sequence(self, X):
        """Procesa una secuencia completa."""
        T = len(X)
        H_SIZE = self.hidden_size
        
        h = np.zeros(H_SIZE)
        C = np.zeros(H_SIZE)
        
        states_h = [h]
        states_C = [C]
        caches   = []
        
        for t in range(T):
            h, C, cache = self.forward_step(X[t], h, C)
            states_h.append(h)
            states_C.append(C)
            caches.append(cache)
        
        return states_h, states_C, caches
    
    def get_param_count(self):
        """Número total de parámetros."""
        D = self.input_size + self.hidden_size
        H = self.hidden_size
        return 4 * (D * H + H)   # 4 matrices de pesos + 4 sesgos


# Prueba del LSTM
print("\nPrueba de CeldaLSTM:")
np.random.seed(99)

INPUT_SIZE  = 5
HIDDEN_SIZE = 8
SEQ_LEN     = 10

lstm = CeldaLSTM(INPUT_SIZE, HIDDEN_SIZE)
X_test = [np.random.randn(INPUT_SIZE) for _ in range(SEQ_LEN)]

states_h, states_C, caches = lstm.forward_sequence(X_test)

print(f"Secuencia de longitud: {SEQ_LEN}")
print(f"Parámetros totales: {lstm.get_param_count():,}")
print()

for t in range(min(4, SEQ_LEN)):
    cache = caches[t]
    print(f"t={t}:")
    print(f"  f_t (forget): {cache['f_t'].round(3)}")
    print(f"  i_t (input):  {cache['i_t'].round(3)}")
    print(f"  o_t (output): {cache['o_t'].round(3)}")
    print(f"  C_t (cell):   {cache['C_t'].round(3)}")
    print(f"  h_t (hidden): {cache['h_t'].round(3)}")
    print()

# Comparación de parámetros: RNN vs LSTM
def contar_parametros_rnn(input_size, hidden_size, output_size):
    return (hidden_size * input_size +     # W_xh
            hidden_size * hidden_size +    # W_hh
            hidden_size +                  # b_h
            output_size * hidden_size +    # W_hy
            output_size)                   # b_y

def contar_parametros_lstm(input_size, hidden_size, output_size):
    D = input_size + hidden_size
    H = hidden_size
    lstm_params = 4 * (D * H + H)
    salida = output_size * hidden_size + output_size
    return lstm_params + salida

I, H, O = 50, 128, 10
rnn_params  = contar_parametros_rnn(I, H, O)
lstm_params = contar_parametros_lstm(I, H, O)

print("--- Comparación de parámetros ---")
print(f"RNN  (input={I}, hidden={H}, output={O}): {rnn_params:,} parámetros")
print(f"LSTM (input={I}, hidden={H}, output={O}): {lstm_params:,} parámetros")
print(f"Factor LSTM/RNN: {lstm_params/rnn_params:.2f}x más parámetros")
```

---

## 🔬 Parte 3: Aplicaciones con PyTorch (60 min)

### 3.1 LSTM para Clasificación de Texto (Análisis de Sentimiento)

En esta sección pasamos de NumPy a **PyTorch** y construimos un pipeline completo de clasificación de texto con LSTM. El análisis de sentimiento es una tarea **Many-to-One**: la LSTM procesa una secuencia de tokens y produce una única etiqueta de clase al final de la secuencia.

#### Pipeline de clasificación de texto

El flujo de datos sigue estos pasos:

```
Texto → Tokenización → Índices enteros → Capa Embedding → LSTM → Dropout → Clasificador lineal → Clase
```

1. **Tokenización**: cada palabra (o subpalabra) se convierte en un índice entero dentro de un vocabulario de tamaño `VOCAB_SIZE`.
2. **Capa Embedding** (`nn.Embedding`): transforma cada índice en un vector denso de `EMBED_DIM` dimensiones. A diferencia de la codificación *one-hot* (vectores dispersos de dimensión `VOCAB_SIZE`), los *embeddings* son representaciones **densas y aprendibles** que capturan relaciones semánticas entre palabras (p. ej., palabras con significado similar tendrán vectores cercanos). Los parámetros de la capa Embedding son `VOCAB_SIZE × EMBED_DIM`.
3. **LSTM** (`nn.LSTM`): procesa la secuencia de embeddings paso a paso, produciendo un hidden state $h_t$ en cada instante. Para clasificación usamos **únicamente el último hidden state** $h_T$ (Many-to-One), que resume toda la secuencia.
4. **Dropout**: regularización estocástica que desactiva aleatoriamente neuronas con probabilidad `p` durante el entrenamiento, reduciendo el sobreajuste.
5. **Capa lineal**: proyecta el último hidden state desde `HIDDEN_SIZE` dimensiones a `NUM_CLASSES` logits, sobre los que se aplica `CrossEntropyLoss`.

#### Dataset sintético

La clase `SentimentDataset` genera datos sintéticos donde la posición en el vocabulario codifica el sentimiento: los tokens en la mitad superior del vocabulario (`[VOCAB_SIZE//2, VOCAB_SIZE)`) corresponden a la clase **positiva (1)**, y los de la mitad inferior (`[0, VOCAB_SIZE//2)`) a la clase **negativa (0)**. Esto crea una señal clara y aprendible, ideal para validar que la arquitectura funciona correctamente antes de escalarla a datos reales.

#### Hiperparámetros y su rol

| Hiperparámetro | Valor | Función |
|---|---|---|
| `VOCAB_SIZE` | 1000 | Tamaño del vocabulario; define la dimensión de la tabla de embeddings |
| `EMBED_DIM` | 64 | Dimensión del espacio de representación de cada token |
| `HIDDEN_SIZE` | 128 | Capacidad de memoria de la LSTM; más alto ≈ más expresividad |
| `NUM_CLASSES` | 2 | Positivo / Negativo |
| `MAX_LEN` | 30 | Longitud fija de cada secuencia de entrada |
| `BATCH_SIZE` | 32 | Número de ejemplos procesados en paralelo por iteración |
| `NUM_EPOCHS` | 10 | Número de pasadas completas sobre el conjunto de entrenamiento |

**Resultados esperados:**

- El entrenamiento de 10 épocas sobre 800 muestras de entrenamiento y 200 de validación es rápido (< 30 segundos en CPU).
- Dado que el dataset sintético es altamente separable, la precisión en validación debe alcanzar **entre el 92 % y el 96 %** a partir de la época 5-7, y estabilizarse ahí.
- La pérdida de entrenamiento debe descender de forma monotónica; si se observan oscilaciones grandes, podría indicar una tasa de aprendizaje demasiado alta.
- Se imprimirá la accuracy final en el conjunto de test, que debe estar en el mismo rango que la validación, confirmando que el modelo generaliza y no sobreajusta.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("LSTM PARA ANÁLISIS DE SENTIMIENTO - PYTORCH")
print("=" * 60)

# -----------------------------------------------
# Dataset sintético de análisis de sentimiento
# -----------------------------------------------
# En la práctica usarías IMDB, SST-2 o un dataset real.
# Aquí creamos datos sintéticos para demostración.

VOCAB_SIZE  = 1000
EMBED_DIM   = 64
HIDDEN_SIZE = 128
NUM_CLASSES = 2
MAX_LEN     = 30
BATCH_SIZE  = 32
NUM_EPOCHS  = 10

class SentimentDataset(Dataset):
    """Dataset sintético de análisis de sentimiento."""
    
    def __init__(self, n_samples=1000, max_len=MAX_LEN, vocab_size=VOCAB_SIZE, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generar secuencias sintéticas
        # Secuencias con muchos índices altos → clase 1 (positivo)
        # Secuencias con muchos índices bajos  → clase 0 (negativo)
        self.data   = []
        self.labels = []
        
        for _ in range(n_samples):
            clase = np.random.randint(0, 2)
            if clase == 1:
                # "Positivo": tokens concentrados en la mitad superior del vocabulario
                seq = np.random.randint(vocab_size // 2, vocab_size, max_len)
            else:
                # "Negativo": tokens concentrados en la mitad inferior del vocabulario
                seq = np.random.randint(0, vocab_size // 2, max_len)
            
            self.data.append(torch.tensor(seq, dtype=torch.long))
            self.labels.append(clase)
        
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# -----------------------------------------------
# Modelo LSTM para clasificación
# -----------------------------------------------
class SentimentLSTM(nn.Module):
    """
    LSTM para clasificación de texto.
    
    Arquitectura:
        Embedding → LSTM → Dropout → Linear → Softmax
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_dir       = 2 if bidirectional else 1
        
        # Capa de Embedding: convierte índices de tokens a vectores densos
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM apilado (Stacked LSTM)
        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,    # (batch, seq, features)
            dropout       = dropout if num_layers > 1 else 0,
            bidirectional = bidirectional
        )
        
        # Regularización
        self.dropout = nn.Dropout(dropout)
        
        # Capa de clasificación
        self.classifier = nn.Linear(hidden_size * self.num_dir, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - índices de tokens
        Returns:
            logits: (batch_size, num_classes)
        """
        # Embedding: (batch, seq) → (batch, seq, embed_dim)
        embedded = self.dropout(self.embedding(x))
        
        # LSTM: (batch, seq, embed_dim) → (batch, seq, hidden * num_dir)
        # lstm_out: (batch, seq, hidden * num_dir)
        # hidden:   (num_layers * num_dir, batch, hidden)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Para Many-to-One: usar el último paso temporal
        # Alternativa: average pooling sobre todos los pasos
        last_out = lstm_out[:, -1, :]   # (batch, hidden * num_dir)
        
        # Clasificar
        out = self.dropout(last_out)
        logits = self.classifier(out)
        
        return logits


# -----------------------------------------------
# Entrenamiento
# -----------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}\n")

# Datos
train_ds = SentimentDataset(n_samples=800)
val_ds   = SentimentDataset(n_samples=200, seed=123)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# Modelo
model = SentimentLSTM(
    vocab_size   = VOCAB_SIZE,
    embed_dim    = EMBED_DIM,
    hidden_size  = HIDDEN_SIZE,
    num_classes  = NUM_CLASSES,
    num_layers   = 2,
    dropout      = 0.3,
    bidirectional= False
).to(device)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
print(f"Arquitectura del modelo:")
print(model)
print(f"\nParámetros totales: {total_params:,}\n")

# Optimizador y función de pérdida
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def evaluate(model, dl, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dl:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)
            total_loss += loss.item()
            preds   = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return total_loss / len(dl), correct / total


history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print("Entrenando modelo LSTM para análisis de sentimiento...")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    ep_loss, correct, total = 0, 0, 0
    
    for X, y in train_dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping (importante para RNNs)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        ep_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    
    scheduler.step()
    train_loss = ep_loss / len(train_dl)
    train_acc  = correct / total
    val_loss, val_acc = evaluate(model, val_dl, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    if epoch % 2 == 0 or epoch == 1:
        print(f"Época {epoch:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")

# Graficar resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(1, NUM_EPOCHS + 1)

ax1.plot(epochs, history['train_loss'], label='Train', color='steelblue')
ax1.plot(epochs, history['val_loss'],   label='Val',   color='tomato', linestyle='--')
ax1.set_title('Pérdida durante entrenamiento')
ax1.set_xlabel('Época')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs, history['train_acc'], label='Train', color='steelblue')
ax2.plot(epochs, history['val_acc'],   label='Val',   color='tomato', linestyle='--')
ax2.set_title('Precisión durante entrenamiento')
ax2.set_xlabel('Época')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('LSTM - Análisis de Sentimiento', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('lstm_sentimiento.png', dpi=100)
plt.show()
print("Figura guardada como 'lstm_sentimiento.png'")
```

### 3.2 Predicción de Series de Tiempo con LSTM

```python
print("\n--- LSTM para Predicción de Series de Tiempo ---")

# -----------------------------------------------
# Generar serie de tiempo sintética (patrón sinusoidal + ruido)
# -----------------------------------------------
def generar_serie_sinusoide(n=1000, freq=0.05, ruido=0.1, seed=42):
    np.random.seed(seed)
    t   = np.arange(n)
    serie = (np.sin(2 * np.pi * freq * t) +
             0.5 * np.sin(2 * np.pi * freq * 3 * t) +
             ruido * np.random.randn(n))
    return serie.astype(np.float32)


def crear_dataset_ventana(serie, ventana=20, horizonte=1):
    """
    Crea pares (X, y) deslizando una ventana sobre la serie.
    X: últimas `ventana` observaciones
    y: próximas `horizonte` observaciones
    """
    X, y = [], []
    for i in range(len(serie) - ventana - horizonte + 1):
        X.append(serie[i : i + ventana])
        y.append(serie[i + ventana : i + ventana + horizonte])
    return np.array(X), np.array(y)


serie    = generar_serie_sinusoide(n=2000)
VENTANA  = 30
HORIZONTE = 1

X_serie, y_serie = crear_dataset_ventana(serie, VENTANA, HORIZONTE)
print(f"Serie de tiempo: {len(serie)} puntos")
print(f"Ventana: {VENTANA} | Horizonte: {HORIZONTE}")
print(f"Muestras: X={X_serie.shape}, y={y_serie.shape}")

# Split train/val/test
n      = len(X_serie)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)

X_tr, y_tr = X_serie[:n_train],         y_serie[:n_train]
X_va, y_va = X_serie[n_train:n_train+n_val], y_serie[n_train:n_train+n_val]
X_te, y_te = X_serie[n_train+n_val:],   y_serie[n_train+n_val:]

# Convertir a tensores y reshapear: (batch, seq_len, features)
def to_tensor(X, y):
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
    yt = torch.tensor(y, dtype=torch.float32)                 # (N, horizonte)
    return Xt, yt

Xt_tr, yt_tr = to_tensor(X_tr, y_tr)
Xt_va, yt_va = to_tensor(X_va, y_va)
Xt_te, yt_te = to_tensor(X_te, y_te)

train_series_dl = DataLoader(list(zip(Xt_tr, yt_tr)), batch_size=64, shuffle=True)
val_series_dl   = DataLoader(list(zip(Xt_va, yt_va)), batch_size=64)


# -----------------------------------------------
# Modelo LSTM para regresión de series de tiempo
# -----------------------------------------------
class LSTMSeriesPredictor(nn.Module):
    """LSTM para predicción de series de tiempo (regresión)."""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]   # último paso temporal
        return self.fc(self.dropout(last))


model_ts = LSTMSeriesPredictor(hidden_size=64, num_layers=2, dropout=0.2).to(device)
criterion_ts = nn.MSELoss()
optimizer_ts = optim.Adam(model_ts.parameters(), lr=1e-3)

print("\nEntrenando LSTM para predicción de series de tiempo...")
EPOCHS_TS = 30
for epoch in range(1, EPOCHS_TS + 1):
    model_ts.train()
    ep_loss = 0.0
    for Xb, yb in train_series_dl:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer_ts.zero_grad()
        pred = model_ts(Xb)
        loss = criterion_ts(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model_ts.parameters(), 1.0)
        optimizer_ts.step()
        ep_loss += loss.item()
    
    if epoch % 10 == 0 or epoch == 1:
        model_ts.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_series_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += criterion_ts(model_ts(Xb), yb).item()
        val_loss /= len(val_series_dl)
        print(f"Época {epoch:2d} | Train MSE: {ep_loss/len(train_series_dl):.6f} | Val MSE: {val_loss:.6f}")

# Predicción en test set
model_ts.eval()
with torch.no_grad():
    pred_te = model_ts(Xt_te.to(device)).cpu().numpy().flatten()

# Visualización
plt.figure(figsize=(14, 5))
plt.plot(range(len(y_te)),  y_te.flatten(),   label='Real',    color='steelblue', lw=1.5)
plt.plot(range(len(pred_te)), pred_te,         label='LSTM',    color='tomato',    lw=1.5, alpha=0.8)
plt.xlabel('Paso temporal')
plt.ylabel('Valor')
plt.title('LSTM: Predicción de Serie de Tiempo (Test Set)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_series_tiempo.png', dpi=100)
plt.show()
print("Figura guardada como 'lstm_series_tiempo.png'")
```

### 3.3 GRU como Alternativa al LSTM

```python
print("\n--- GRU (Gated Recurrent Unit) como alternativa al LSTM ---")

print("""
GRU simplifica el LSTM combinando las puertas forget e input
en una sola "update gate", y fusionando cell state y hidden state.

Ecuaciones:
  r_t = σ(W_r · [h_{t-1}, x_t])       ← Reset gate
  z_t = σ(W_z · [h_{t-1}, x_t])       ← Update gate
  h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t]) ← Candidato
  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t ← Nuevo hidden state

Ventajas del GRU vs LSTM:
  ✓ Menos parámetros (3 matrices en lugar de 4)
  ✓ Más rápido de entrenar
  ✓ Comparable en rendimiento en muchas tareas
  ✗ Menor capacidad expresiva en tareas complejas
""")


class GRUSeriesPredictor(nn.Module):
    """GRU para predicción de series de tiempo."""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last = gru_out[:, -1, :]
        return self.fc(self.dropout(last))


model_gru    = GRUSeriesPredictor(hidden_size=64, num_layers=2, dropout=0.2).to(device)
optimizer_gru = optim.Adam(model_gru.parameters(), lr=1e-3)

print("Entrenando GRU para predicción de series de tiempo...")
for epoch in range(1, EPOCHS_TS + 1):
    model_gru.train()
    ep_loss = 0.0
    for Xb, yb in train_series_dl:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer_gru.zero_grad()
        loss = criterion_ts(model_gru(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model_gru.parameters(), 1.0)
        optimizer_gru.step()
        ep_loss += loss.item()
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Época {epoch:2d} | GRU Train MSE: {ep_loss/len(train_series_dl):.6f}")

# Comparar parámetros LSTM vs GRU
lstm_params = sum(p.numel() for p in model_ts.parameters())
gru_params  = sum(p.numel() for p in model_gru.parameters())
print(f"\nParámetros LSTM: {lstm_params:,}")
print(f"Parámetros GRU:  {gru_params:,}")
print(f"Diferencia:      {lstm_params - gru_params:,} ({100*(lstm_params-gru_params)/lstm_params:.1f}% más en LSTM)")
```

---

## 🔬 Parte 4: Arquitecturas Avanzadas (30 min)

### 4.1 LSTM Bidireccional

```python
print("=" * 60)
print("LSTM BIDIRECCIONAL")
print("=" * 60)

print("""
Un LSTM Bidireccional (BiLSTM) procesa la secuencia
en ambas direcciones: hacia adelante y hacia atrás.

Arquitectura:
   x_0 → LSTM_fwd → h_fwd_0
   x_1 → LSTM_fwd → h_fwd_1      → Concatenar → h_bi_t
   ...                            ← LSTM_bwd ← x_t

Ventaja: el hidden state en cada paso contiene
información tanto del pasado como del futuro.

Ejemplo: "El banco [MASK] estaba lleno de gente."
  - LSTM fwd solo sabe: "El banco"
  - BiLSTM sabe: "El banco" Y "estaba lleno de gente"
  → Mayor contexto para la predicción

Limitación: NO se puede usar en tiempo real (streaming),
ya que necesita conocer el futuro.

Casos de uso válidos:
  ✓ NLP offline (clasificación, NER, QA)
  ✓ Reconocimiento de audio completo
  ✗ Generación de texto token por token
  ✗ Predicción de series de tiempo en tiempo real
""")


class BiLSTMClassifier(nn.Module):
    """LSTM Bidireccional para clasificación."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_size,
            num_layers    = 2,
            batch_first   = True,
            dropout       = dropout,
            bidirectional = True     # ← Clave para BiLSTM
        )
        self.dropout    = nn.Dropout(dropout)
        # hidden_size * 2 porque concatenamos fwd y bwd
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        emb      = self.dropout(self.embedding(x))
        out, _   = self.lstm(emb)
        last_out = out[:, -1, :]          # último paso: [fwd_h, bwd_h]
        logits   = self.classifier(self.dropout(last_out))
        return logits


bilstm_model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE // 2, NUM_CLASSES)
bilstm_params = sum(p.numel() for p in bilstm_model.parameters())
unidirec_params = sum(p.numel() for p in model.parameters())

print(f"Parámetros LSTM Unidireccional: {unidirec_params:,}")
print(f"Parámetros BiLSTM:              {bilstm_params:,}")
print(f"Factor BiLSTM/LSTM:             {bilstm_params/unidirec_params:.2f}x")
```

### 4.2 LSTM Apilado (Stacked LSTM)

```python
print("\n--- LSTM Apilado (Stacked LSTM) ---")

print("""
Un Stacked LSTM apila múltiples capas LSTM, donde la salida
de una capa es la entrada de la siguiente.

Layer 1: x_t        → h1_t
Layer 2: h1_t       → h2_t
Layer 3: h2_t       → h3_t
         ...
Salida:  h_last_t   → predicción

Cada capa aprende representaciones más abstractas.

En PyTorch: nn.LSTM(num_layers=N) hace esto automáticamente.
""")

# Comparar distintas profundidades
configs = [
    {'layers': 1, 'hidden': 128, 'label': '1 capa'},
    {'layers': 2, 'hidden': 128, 'label': '2 capas'},
    {'layers': 3, 'hidden': 128, 'label': '3 capas'},
    {'layers': 4, 'hidden': 128, 'label': '4 capas'},
]

print(f"{'Config':<10} {'Parámetros':>12} {'Profundidad':>12}")
print("-" * 36)
for cfg in configs:
    m = nn.LSTM(EMBED_DIM, cfg['hidden'], cfg['layers'], batch_first=True)
    params = sum(p.numel() for p in m.parameters())
    print(f"{cfg['label']:<10} {params:>12,} {cfg['layers']:>12}")
```

### 4.3 Arquitectura Encoder-Decoder

```python
print("\n--- Arquitectura Encoder-Decoder ---")

print("""
La arquitectura Encoder-Decoder (Seq2Seq) permite manejar
secuencias de longitudes diferentes (Many-to-Many asíncrono).

Aplicaciones:
  - Traducción automática  (ES → EN)
  - Resumen automático     (texto largo → texto corto)
  - Generación de código   (descripción → código)
  - Chatbots               (pregunta → respuesta)

Estructura:
  Encoder: Lee la secuencia de entrada y produce un
           "vector de contexto" (último hidden state).

  Decoder: Genera la secuencia de salida token por token,
           usando el vector de contexto como estado inicial.

  ┌──────────────────────────────────────────────────┐
  │  ENCODER                    DECODER              │
  │  x_1 x_2 x_3               <START> y_1 y_2      │
  │   │   │   │                    │    │   │        │
  │  LSTM LSTM LSTM → contexto → LSTM LSTM LSTM      │
  │                       │         │    │   │       │
  │                       └─────→  y_1  y_2  y_3    │
  └──────────────────────────────────────────────────┘

Limitación del Encoder-Decoder clásico:
  El vector de contexto es un cuello de botella:
  toda la información de la secuencia de entrada debe
  comprimirse en un solo vector.
  → Esta limitación motivó el mecanismo de ATENCIÓN (Lab 12).
""")


class Encoder(nn.Module):
    """Encoder LSTM: lee la secuencia de entrada."""
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        # x: (batch, src_len, features)
        output, (hidden, cell) = self.lstm(x)
        # hidden y cell: (num_layers, batch, hidden_size)
        return hidden, cell


class Decoder(nn.Module):
    """Decoder LSTM: genera la secuencia de salida."""
    
    def __init__(self, output_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
    
    def forward_step(self, x_t, hidden, cell):
        # x_t: (batch, 1, output_size)
        output, (hidden, cell) = self.lstm(x_t, (hidden, cell))
        pred = self.fc(output.squeeze(1))  # (batch, output_size)
        return pred, hidden, cell


class Seq2Seq(nn.Module):
    """Modelo Encoder-Decoder para secuencias."""
    
    def __init__(self, encoder, decoder, output_size):
        super().__init__()
        self.encoder     = encoder
        self.decoder     = decoder
        self.output_size = output_size
    
    def forward(self, src, tgt_len):
        """
        Args:
            src: (batch, src_len, input_size) - secuencia de entrada
            tgt_len: int - longitud de la secuencia de salida
        Returns:
            outputs: (batch, tgt_len, output_size)
        """
        batch_size = src.size(0)
        
        # Codificar
        hidden, cell = self.encoder(src)
        
        # Decodificar paso a paso
        outputs = []
        x_t = torch.zeros(batch_size, 1, self.output_size).to(src.device)
        
        for _ in range(tgt_len):
            pred, hidden, cell = self.decoder.forward_step(x_t, hidden, cell)
            outputs.append(pred.unsqueeze(1))
            x_t = pred.unsqueeze(1)   # Usar predicción como siguiente entrada
        
        return torch.cat(outputs, dim=1)


# Crear y probar el modelo Seq2Seq
INPUT_FEAT  = 10
OUTPUT_FEAT = 5
HIDDEN_S2S  = 64
SRC_LEN     = 15
TGT_LEN     = 8
BATCH_S2S   = 4

encoder_s2s = Encoder(INPUT_FEAT, HIDDEN_S2S, num_layers=1)
decoder_s2s = Decoder(OUTPUT_FEAT, HIDDEN_S2S, num_layers=1)
seq2seq     = Seq2Seq(encoder_s2s, decoder_s2s, OUTPUT_FEAT)

# Prueba de forward pass
src_demo = torch.randn(BATCH_S2S, SRC_LEN, INPUT_FEAT)
out_demo = seq2seq(src_demo, TGT_LEN)

print(f"Modelo Encoder-Decoder (Seq2Seq):")
print(f"  Entrada:  (batch={BATCH_S2S}, src_len={SRC_LEN}, features={INPUT_FEAT})")
print(f"  Salida:   (batch={BATCH_S2S}, tgt_len={TGT_LEN}, features={OUTPUT_FEAT})")
print(f"  Shape real de salida: {out_demo.shape}")
print(f"  Parámetros encoder: {sum(p.numel() for p in encoder_s2s.parameters()):,}")
print(f"  Parámetros decoder: {sum(p.numel() for p in decoder_s2s.parameters()):,}")
print(f"\n→ Ver Lab 12 (Transformers) para la solución al cuello de botella con Atención.")
```

---

## 📊 Análisis de Rendimiento

### Benchmark: RNN vs LSTM vs GRU

```python
import time

print("=" * 60)
print("BENCHMARK: RNN vs LSTM vs GRU")
print("=" * 60)

# -----------------------------------------------
# Comparación en tarea de clasificación de secuencias
# -----------------------------------------------
HIDDEN_BENCH = 64
INPUT_BENCH  = 32
SEQ_BENCH    = 50
BATCH_BENCH  = 128
OUTPUT_BENCH = 2

def crear_modelo_recurrente(tipo, input_size, hidden_size, output_size, num_layers=2):
    """Crea un modelo recurrente del tipo especificado."""
    
    class ModeloRecurrente(nn.Module):
        def __init__(self):
            super().__init__()
            RNN_CLASS = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[tipo]
            self.rnn = RNN_CLASS(
                input_size  = input_size,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                batch_first = True,
                dropout     = 0.2 if num_layers > 1 else 0
            )
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])
    
    return ModeloRecurrente()

modelos = {
    'RNN':  crear_modelo_recurrente('RNN',  INPUT_BENCH, HIDDEN_BENCH, OUTPUT_BENCH),
    'LSTM': crear_modelo_recurrente('LSTM', INPUT_BENCH, HIDDEN_BENCH, OUTPUT_BENCH),
    'GRU':  crear_modelo_recurrente('GRU',  INPUT_BENCH, HIDDEN_BENCH, OUTPUT_BENCH),
}

# Datos sintéticos para benchmark
X_bench = torch.randn(BATCH_BENCH, SEQ_BENCH, INPUT_BENCH)
y_bench = torch.randint(0, OUTPUT_BENCH, (BATCH_BENCH,))

print(f"\nConfiguración:")
print(f"  Input size:   {INPUT_BENCH}")
print(f"  Hidden size:  {HIDDEN_BENCH}")
print(f"  Seq length:   {SEQ_BENCH}")
print(f"  Batch size:   {BATCH_BENCH}")
print(f"  Num layers:   2")
print()

resultados_bench = {}

for nombre, modelo in modelos.items():
    modelo = modelo.to(device)
    params = sum(p.numel() for p in modelo.parameters())
    
    optimizer_b = optim.Adam(modelo.parameters(), lr=1e-3)
    criterion_b = nn.CrossEntropyLoss()
    
    # Medir tiempo de entrenamiento (50 pasos)
    modelo.train()
    X_d, y_d = X_bench.to(device), y_bench.to(device)
    
    inicio = time.time()
    for _ in range(50):
        optimizer_b.zero_grad()
        logits = modelo(X_d)
        loss   = criterion_b(logits, y_d)
        loss.backward()
        optimizer_b.step()
    tiempo = time.time() - inicio
    
    resultados_bench[nombre] = {
        'params':  params,
        'tiempo':  tiempo,
        'loss':    loss.item()
    }
    
    print(f"{nombre:<6}: params={params:>8,} | tiempo={tiempo:.3f}s | loss={loss.item():.4f}")

# Gráfica comparativa
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
nombres  = list(resultados_bench.keys())
colores  = ['#e74c3c', '#3498db', '#2ecc71']

for ax, (metrica, titulo, fmt) in zip(
        axes,
        [('params', 'Número de Parámetros', '{:,.0f}'),
         ('tiempo', 'Tiempo de Entrenamiento (50 pasos, s)', '{:.3f}s'),
         ('loss',   'Pérdida Final', '{:.4f}')]):
    valores = [resultados_bench[n][metrica] for n in nombres]
    bars = ax.bar(nombres, valores, color=colores, edgecolor='black', linewidth=1)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                fmt.format(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(titulo, fontsize=11)
    ax.set_ylabel(titulo)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Benchmark: RNN vs LSTM vs GRU', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('benchmark_rnn_lstm_gru.png', dpi=100)
plt.show()
print("Figura guardada como 'benchmark_rnn_lstm_gru.png'")

# Tabla resumen
print("\n--- Tabla Resumen Comparativa ---")
print(f"{'Modelo':<8} {'Parámetros':>12} {'Tiempo (s)':>12} {'Gates':>8} {'Cell State':>12}")
print("-" * 55)
info = {
    'RNN':  {'gates': 0, 'cell_state': 'No'},
    'LSTM': {'gates': 3, 'cell_state': 'Sí'},
    'GRU':  {'gates': 2, 'cell_state': 'No'},
}
for nombre in nombres:
    r = resultados_bench[nombre]
    i = info[nombre]
    print(f"{nombre:<8} {r['params']:>12,} {r['tiempo']:>12.3f} {i['gates']:>8} {i['cell_state']:>12}")
```

---

## 🎯 EJERCICIOS PROPUESTOS

### Nivel Básico

**Ejercicio B1: Implementar GRU desde cero**
Implementa una celda GRU en NumPy siguiendo las ecuaciones:
```python
r_t = σ(W_r · [h_{t-1}, x_t])
z_t = σ(W_z · [h_{t-1}, x_t])
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```
Verifica que los resultados coinciden con `nn.GRU` de PyTorch con los mismos pesos.

**Ejercicio B2: LSTM para clasificación de secuencias numéricas**
Crea un dataset donde:
- Secuencias con suma > 0 son clase 1
- Secuencias con suma ≤ 0 son clase 0

Entrena un LSTM con PyTorch y reporta accuracy en el conjunto de test.

**Ejercicio B3: Predicción multi-paso**
Modifica el modelo de predicción de series de tiempo para predecir los próximos **5 pasos** en lugar de 1. Compara el error con el modelo de 1 paso.

### Nivel Intermedio

**Ejercicio I1: Análisis de sentimiento con embeddings preentrenados**
1. Descarga embeddings de GloVe o usa `torchtext.vocab`
2. Inicializa la capa `nn.Embedding` con embeddings preentrenados
3. Compara el rendimiento con embeddings aleatorios
4. Experimenta con congelar vs. afinar (fine-tune) los embeddings

**Ejercicio I2: LSTM vs CNN para clasificación de texto**
- Implementa un clasificador con 1D-CNN (`nn.Conv1d`) para texto
- Compara con el LSTM en precisión, número de parámetros y tiempo de entrenamiento
- ¿Cuándo preferiría usar CNN vs LSTM?

**Ejercicio I3: Sequence-to-Sequence para sumarización**
Entrena un modelo Encoder-Decoder (Seq2Seq) en un dataset sintético de "sumarización":
- Entrada: oración de 20 tokens
- Salida: primeras 5 palabras más importantes (simplificado)
Usa "teacher forcing" durante el entrenamiento.

### Nivel Avanzado

**Ejercicio A1: BPTT Truncado**
Implementa BPTT truncado en la clase `RNN_BPTT`:
- Procesar secuencias largas (longitud 200) en bloques de k=20 pasos
- Comparar convergencia con BPTT completo
- Medir uso de memoria con `torch.cuda.memory_allocated()`

**Ejercicio A2: Generación de texto con LSTM**
Entrena un LSTM de nivel carácter para generar texto:
1. Carga un corpus (por ejemplo, el libro Don Quixote en texto plano)
2. Entrena el LSTM para predecir el siguiente carácter
3. Implementa muestreo con temperatura para controlar la creatividad
4. Genera 500 caracteres a partir de un prompt dado

**Ejercicio A3: LSTM con mecanismo de atención básico**
Extiende el modelo de análisis de sentimiento con atención:
```python
# En lugar de usar solo el último estado:
# 1. Calcular scores de atención para cada paso temporal
# 2. Crear un contexto como promedio ponderado de los estados ocultos
# 3. Visualizar los pesos de atención

attn_scores = softmax(W_attn @ lstm_out.transpose(1,2))  # (batch, 1, seq_len)
context     = (attn_scores @ lstm_out).squeeze(1)         # (batch, hidden)
```
Compara con el modelo sin atención. ¿Qué palabras reciben más atención para cada clase?

---

## 📝 Entregables

### Código Fuente

- `rnn_desde_cero.py` — Implementación de RNN con BPTT en NumPy
- `lstm_desde_cero.py` — Implementación de celda LSTM en NumPy
- `lstm_sentimiento.py` — Clasificador de sentimiento con PyTorch
- `lstm_series_tiempo.py` — Predictor de series de tiempo
- `gru_comparacion.py` — Comparación GRU vs LSTM
- `encoder_decoder.py` — Modelo Seq2Seq básico
- `benchmark.py` — Análisis de rendimiento comparativo

### Modelos Entrenados

- `modelo_sentimiento.pth` — Pesos del clasificador LSTM
- `modelo_series.pth` — Pesos del predictor de series
- `resultados_benchmark.json` — Métricas de comparación RNN/LSTM/GRU

### Documentación

- Comentarios en todos los bloques de código
- Docstrings en todas las clases y funciones
- README.md actualizado con instrucciones de ejecución

### Reporte Final

Documento en formato Markdown o PDF (2-4 páginas) con:
1. **Resumen** de las arquitecturas implementadas
2. **Comparación RNN vs LSTM vs GRU**: tabla de métricas
3. **Análisis del gradiente desvaneciente**: gráficas y conclusiones
4. **Resultados en las aplicaciones**: análisis de sentimiento y series de tiempo
5. **Reflexión**: ¿Cuándo usarías LSTM vs Transformers?
6. **Referencias** citadas

---

## 🎯 Criterios de Evaluación (CDIO)

### Concebir (25%) — Comprensión del Problema

✅ Identifica correctamente cuándo usar RNN vs LSTM vs GRU vs Transformers  
✅ Explica el problema del gradiente desvaneciente con evidencia matemática  
✅ Relaciona las puertas LSTM con el problema de memoria a largo plazo  
✅ Conecta los conceptos de BPTT con backpropagation estándar (Lab 05)  
✅ Propone arquitecturas adecuadas para los tipos de secuencias (one-to-many, many-to-one, etc.)  

**Evidencia:** Sección de reflexión en el reporte; respuestas a las preguntas iniciales.

### Diseñar (25%) — Arquitectura y Planificación

✅ Selecciona hiperparámetros justificados (hidden size, num_layers, dropout)  
✅ Diseña correctamente el pipeline de datos para secuencias (padding, batching)  
✅ Elige la configuración bidireccional vs unidireccional según la tarea  
✅ Planifica el proceso de entrenamiento con gradient clipping  
✅ Compara diseños alternativos antes de implementar  

**Evidencia:** Diagrama de arquitectura en el reporte; justificación de hiperparámetros.

### Implementar (30%) — Código Funcional

✅ Implementación correcta de RNN y LSTM desde cero con NumPy  
✅ BPTT completo sin errores en el cálculo de gradientes  
✅ Modelos PyTorch con al menos dos aplicaciones funcionando  
✅ Gradient clipping implementado correctamente  
✅ Métricas evaluadas en conjunto de test separado  
✅ Código limpio, comentado y reproducible (random seeds)  

**Evidencia:** Código fuente entregado; tests unitarios básicos.

### Operar (20%) — Análisis y Evaluación

✅ Benchmark comparativo documentado (RNN vs LSTM vs GRU)  
✅ Curvas de aprendizaje analizadas e interpretadas  
✅ Detección y solución de overfitting (dropout, weight decay)  
✅ Conclusiones sobre la idoneidad de cada arquitectura  
✅ Reflexión sobre las limitaciones y el paso hacia Transformers (Lab 12)  

**Evidencia:** Gráficas en el reporte; tabla de resultados comparativos.

---

## 📋 Rúbrica de Evaluación

| Criterio | Excelente (90-100%) | Bueno (75-89%) | Aceptable (60-74%) | Insuficiente (<60%) | Puntos |
|----------|--------------------|-----------------|--------------------|---------------------|--------|
| **Comprensión RNN** | Explica BPTT, gradiente desvaneciente y tipos de secuencias con exactitud y ejemplos propios | Explica correctamente pero sin ejemplos propios | Explicación parcial con errores menores | Definición superficial o incorrecta | 10 |
| **Implementación NumPy** | RNN y LSTM completos, BPTT correcto, resultados verificados contra PyTorch | RNN o LSTM incompleto, BPTT funcional | Solo forward pass, BPTT ausente o incorrecto | No implementado o no funciona | 20 |
| **Puertas LSTM** | Las tres puertas implementadas correctamente; explica intuitivamente el rol de cada una | Implementación correcta pero explicación superficial | Una o dos puertas incorrectas | Sin comprensión de las puertas | 10 |
| **Aplicación Texto** | Clasificador LSTM funcional con >85% de accuracy; análisis de hiperparámetros | >75% accuracy; poca experimentación | >60% accuracy; sin análisis | No funciona o no entregado | 15 |
| **Aplicación Series** | LSTM y GRU comparados; métricas calculadas; análisis de predicciones | Solo LSTM; métricas básicas | Modelo funciona pero sin análisis | No funciona o no entregado | 15 |
| **Arquitecturas avanzadas** | BiLSTM, Stacked LSTM y Seq2Seq implementados y explicados correctamente | Dos de tres implementados | Solo uno implementado | No implementado | 10 |
| **Benchmark y análisis** | Comparación completa RNN/LSTM/GRU con gráficas, tablas y conclusiones sólidas | Comparación con métricas básicas | Comparación superficial | Sin comparación | 10 |
| **Calidad del código** | Código limpio, documentado, reproducible, PEP 8, sin repetición | Código funcional con documentación básica | Código funcional pero difícil de leer | Código desordenado o no funcional | 5 |
| **Reporte final** | Análisis profundo con conexión a Labs anteriores y Lab 12, gráficas claras | Reporte completo con análisis básico | Reporte incompleto | Sin reporte o muy superficial | 5 |
| **TOTAL** | | | | | **100** |

---

## 📚 Referencias Adicionales

### Papers Fundamentales

1. **Hochreiter, S. & Schmidhuber, J. (1997)**  
   *Long Short-Term Memory*  
   Neural Computation, 9(8), 1735-1780.  
   → El paper original de LSTM — lectura obligatoria.

2. **Cho, K. et al. (2014)**  
   *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*  
   arXiv:1406.1078  
   → Introduce el GRU y la arquitectura Encoder-Decoder.

3. **Chung, J. et al. (2014)**  
   *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*  
   arXiv:1412.3555  
   → Comparación empírica de RNN, LSTM y GRU.

4. **Sutskever, I., Vinyals, O. & Le, Q. V. (2014)**  
   *Sequence to Sequence Learning with Neural Networks*  
   NeurIPS 2014.  
   → Arquitectura Seq2Seq clásica para traducción.

5. **Vaswani, A. et al. (2017)**  
   *Attention is All You Need*  
   NeurIPS 2017.  
   → Transición de RNNs a Transformers (Lab 12).

### Tutoriales y Recursos Online

- **PyTorch Official Tutorial — NLP from Scratch:**  
  https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

- **Colah's Blog — Understanding LSTM Networks (2015):**  
  https://colah.github.io/posts/2015-08-Understanding-LSTMs/  
  → La mejor explicación visual de LSTM disponible.

- **The Unreasonable Effectiveness of Recurrent Neural Networks (Karpathy, 2015):**  
  http://karpathy.github.io/2015/05/21/rnn-effectiveness/  
  → Motivación intuitiva para usar RNNs con texto.

- **Dive into Deep Learning — RNN Chapter:**  
  https://d2l.ai/chapter_recurrent-neural-networks/

### Libros de Referencia

- **Goodfellow, I., Bengio, Y. & Courville, A. (2016)**  
  *Deep Learning*, Capítulo 10: "Sequence Modeling: Recurrent and Recursive Nets"  
  MIT Press. Disponible en https://www.deeplearningbook.org/

- **Chollet, F. (2021)**  
  *Deep Learning with Python* (2nd ed.), Capítulo 10  
  Manning Publications.

### Herramientas y Datasets

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| **IMDB Dataset** | 50,000 reseñas de películas (sentimiento) | `torchtext.datasets.IMDB` |
| **Penn Treebank** | Dataset clásico para modelado de lenguaje | Disponible en NLTK |
| **UCR Time Series** | Repositorio de series de tiempo | https://timeseriesclassification.com/ |
| **WMT Translation** | Pares de traducción multilingüe | https://www.statmt.org/wmt14/ |
| **Weights & Biases** | Tracking de experimentos | https://wandb.ai/ |

---

## 🎓 Notas Finales

### Ideas Clave a Recordar

**Sobre las Ecuaciones Fundamentales:**

```
# RNN estándar (simple pero con gradiente desvaneciente)
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)

# LSTM (solución al gradiente desvaneciente)
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   ← "¿Qué olvidar?"
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   ← "¿Qué recordar nuevo?"
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
C_t = f_t * C_{t-1} + i_t * C̃_t       ← "Actualizar memoria"
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   ← "¿Qué exponer?"
h_t = o_t * tanh(C_t)

# GRU (simplificación eficiente)
r_t = σ(W_r · [h_{t-1}, x_t])          ← Reset gate
z_t = σ(W_z · [h_{t-1}, x_t])          ← Update gate
h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

**Decisiones de Diseño Prácticas:**

| Situación | Recomendación |
|-----------|---------------|
| Dependencias de largo plazo | LSTM > RNN |
| Recursos limitados (móvil/embebido) | GRU > LSTM |
| Necesitas procesar futuro y pasado | BiLSTM (si no es en tiempo real) |
| Secuencias de entrada y salida de distinta longitud | Seq2Seq |
| Secuencias muy largas (>500 tokens) | Transformers (Lab 12) |
| Baseline rápido | GRU de 1-2 capas |

**Buenas Prácticas en Entrenamiento:**

1. **Siempre usa Gradient Clipping** para RNNs  
   ```python
   nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Inicializa el forget gate con sesgo positivo** en LSTM:  
   ```python
   nn.init.constant_(lstm.bias_ih_l0[hidden_size:2*hidden_size], 1.0)
   ```

3. **Usa `batch_first=True`** en PyTorch para dimensiones intuitivas (batch, seq, features)

4. **Truncated BPTT** para secuencias largas:  
   Detach el estado oculto cada k pasos para controlar el uso de memoria

5. **Embedding Dropout** es más efectivo que Dropout estándar en NLP

### Conexión con Laboratorios Anteriores

- **Lab 01-02**: Las neuronas básicas y la arquitectura densa son la base de la celda RNN
- **Lab 03**: `tanh` y `sigmoid` son las activaciones clave en RNN/LSTM
- **Lab 04**: Cross-entropy loss para clasificación de secuencias; MSE para regresión
- **Lab 05**: BPTT es backpropagation estándar aplicado a través del tiempo
- **Lab 06**: Gradient clipping extiende las técnicas de entrenamiento vistas
- **Lab 07**: Las métricas (accuracy, F1, MSE) se aplican igual en modelos recurrentes
- **Lab 08**: PyTorch facilita enormemente la implementación de RNNs/LSTMs
- **Lab 09**: Las RNNs generativas (generación de texto/secuencias) son base de modelos IA generativa
- **Lab 10**: Las CNNs y LSTMs pueden combinarse (CNN extrae características, LSTM procesa secuencias)

### Próximo Paso: Lab 12 — Transformers

```
RNNs → Encoder-Decoder → Atención → Transformers

El mecanismo de ATENCIÓN que viste brevemente en el Ejercicio A3
es el corazón de los Transformers. En el Lab 12 aprenderás:

1. Self-Attention: cada token atiende a todos los demás
2. Multi-Head Attention: múltiples "perspectivas" de atención
3. Positional Encoding: cómo los Transformers codifican el orden
4. BERT, GPT y la revolución del pre-entrenamiento

Pregunta para reflexionar antes del Lab 12:
"Si pudieras hacer que cada token de una secuencia 'vea'
directamente todos los demás tokens, ¿por qué sería mejor
que el RNN que solo ve el pasado?"
```

---

## ✅ Checklist de Verificación

Antes de entregar, verifica que has completado todos los puntos:

### Comprensión Teórica
- [ ] Puedo explicar la ecuación de la RNN sin mirar notas
- [ ] Entiendo por qué el gradiente desaparece con tanh y muchos pasos de tiempo
- [ ] Sé el rol de cada puerta LSTM (forget, input, output) con un ejemplo propio
- [ ] Puedo comparar GRU y LSTM en términos de parámetros y capacidad
- [ ] Entiendo cuándo usar cada tipo de arquitectura (one-to-many, many-to-one, etc.)

### Implementaciones Completadas
- [ ] Celda RNN básica con NumPy (forward pass)
- [ ] BPTT completo en NumPy con gradient clipping
- [ ] Celda LSTM completa con NumPy (forward pass)
- [ ] Demostración del gradiente desvaneciente con gráfica
- [ ] LSTM para clasificación de texto con PyTorch
- [ ] LSTM para predicción de series de tiempo con PyTorch
- [ ] GRU implementado y comparado con LSTM
- [ ] BiLSTM implementado
- [ ] Arquitectura Encoder-Decoder (Seq2Seq) implementada

### Análisis y Documentación
- [ ] Benchmark comparativo RNN vs LSTM vs GRU con tabla y gráficas
- [ ] Análisis de las curvas de aprendizaje (¿hay overfitting?)
- [ ] Conclusiones escritas sobre cuándo usar cada arquitectura
- [ ] Reflexión sobre la conexión con el Lab 12 (Transformers)
- [ ] Código comentado y reproducible

### Entregables
- [ ] Todos los archivos `.py` entregados y funcionando
- [ ] Modelos guardados (`.pth`)
- [ ] Gráficas generadas y guardadas
- [ ] Reporte final completo (2-4 páginas)
- [ ] Respuestas a las preguntas de reflexión iniciales incluidas en el reporte

---

*Guía de Laboratorio: Redes Neuronales Recurrentes y LSTM — Lab 11*  
*Universidad — Curso de Deep Learning*  
*Conexión: ← Lab 10 (CNNs) | Lab 12 (Transformers) →*
