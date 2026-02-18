# Guía de Laboratorio: Backpropagation - El Corazón del Deep Learning

## 📋 Información del Laboratorio

**Título:** Fundamentos de Deep Learning - Backpropagation  
**Código:** Lab 05  
**Duración:** 2-3 horas  
**Nivel:** Intermedio  

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender profundamente la regla de la cadena y su aplicación en deep learning
2. Interpretar y construir grafos computacionales para operaciones matemáticas
3. Implementar el algoritmo de backpropagation desde cero en Python
4. Calcular gradientes analíticos para funciones de activación y pérdida
5. Verificar implementaciones mediante gradient checking (gradientes numéricos)
6. Identificar y guardar valores intermedios necesarios para backpropagation
7. Entrenar una red neuronal completa usando backpropagation
8. Diagnosticar problemas comunes: gradientes que explotan o desaparecen
9. Optimizar el código usando vectorización y operaciones matriciales

## 📚 Prerrequisitos

### Conocimientos

- Python intermedio (funciones, clases, NumPy)
- Cálculo diferencial (derivadas, regla de la cadena)
- Álgebra lineal (multiplicación matricial, transposición)
- Redes neuronales básicas (forward pass, funciones de activación)
- Funciones de pérdida (MSE, cross-entropy)

### Software

- Python 3.8+
- NumPy 1.19+
- Matplotlib (para visualizaciones)
- Jupyter Notebook (recomendado)

### Material de Lectura

Antes de comenzar, lee:
- `teoria.md` - Marco teórico completo sobre backpropagation
- `README.md` - Estructura del laboratorio y recursos disponibles
- Revisar Labs anteriores (01-04) sobre neuronas, funciones de activación y pérdida

## 📖 Introducción

### El Problema Fundamental del Aprendizaje

Hasta ahora hemos visto cómo las redes neuronales hacen predicciones mediante el **forward pass**: los datos fluyen de entrada a salida a través de capas de neuronas. Pero, ¿cómo aprende la red? ¿Cómo ajustamos los millones de parámetros (pesos y bias) para mejorar las predicciones?

La respuesta es **Backpropagation** - el algoritmo que revolucionó el deep learning.

### ¿Qué es Backpropagation?

**Backpropagation** (propagación hacia atrás) es un algoritmo eficiente para calcular gradientes de la función de pérdida con respecto a todos los parámetros de una red neuronal. Es la aplicación inteligente de la **regla de la cadena del cálculo diferencial**.

```
ENTRENAMIENTO DE UNA RED NEURONAL:

1. Forward Pass:
   Entrada → Capa1 → Capa2 → ... → Salida → Pérdida
   
2. Backward Pass (Backpropagation):
   ∂L/∂W₁ ← ∂L/∂W₂ ← ... ← ∂L/∂Wₙ ← Gradiente de Pérdida
   
3. Actualización de Parámetros:
   W_nuevo = W_viejo - α * ∂L/∂W
```

### Contexto Histórico

Aunque las redes neuronales fueron propuestas en los años 40-50, su verdadero potencial no se liberó hasta que backpropagation fue popularizado en 1986 por Rumelhart, Hinton y Williams. Este algoritmo:

- Hace posible entrenar redes con múltiples capas ocultas
- Calcula gradientes de manera eficiente (complejidad lineal)
- Es la base de todas las redes neuronales modernas
- Permite el aprendizaje automático de representaciones complejas

### La Regla de la Cadena: Fundamento Matemático

Backpropagation se basa en una idea simple del cálculo: **la regla de la cadena**.

**Para funciones compuestas:**
Si tenemos `y = f(g(x))`, entonces:
$$\frac{dy}{dx} = \frac{dy}{dg} \times \frac{dg}{dx}$$

**Ejemplo intuitivo:**

Imagina que conduces un auto:
- La distancia depende de la velocidad: `d = v * t`
- La velocidad depende del acelerador: `v = k * a`
- Por tanto: `d = (k * a) * t`

¿Cómo afecta el acelerador a la distancia? Usando la regla de la cadena:
$$\frac{∂d}{∂a} = \frac{∂d}{∂v} \times \frac{∂v}{∂a} = t \times k$$

En redes neuronales, la "distancia" es la pérdida, y el "acelerador" son los pesos.

### Grafos Computacionales

Una forma poderosa de entender backpropagation es mediante **grafos computacionales** - diagramas que muestran cómo se calculan las salidas a partir de las entradas.

**Ejemplo simple: z = (x + y) × w**

```
    x ───┐
         ├──→ [+] ──→ q ───┐
    y ───┘                 │
                           ├──→ [×] ──→ z
                       w ───┘
```

**Forward pass** (izquierda → derecha): Calcular z
- `q = x + y`
- `z = q × w`

**Backward pass** (derecha ← izquierda): Calcular gradientes
- `∂z/∂z = 1` (empezamos aquí)
- `∂z/∂q = w`, `∂z/∂w = q`
- `∂z/∂x = (∂z/∂q) × (∂q/∂x) = w × 1 = w`
- `∂z/∂y = (∂z/∂q) × (∂q/∂y) = w × 1 = w`

### ¿Por qué es Tan Importante?

**Sin backpropagation:**
- Calcular gradientes manualmente es tedioso y propenso a errores
- Complejidad computacional prohibitiva para redes grandes
- Imposible entrenar redes profundas eficientemente

**Con backpropagation:**
- Cálculo automático de todos los gradientes
- Un solo pase hacia atrás calcula todos los gradientes necesarios
- Complejidad lineal O(n) donde n es el número de parámetros
- Permite entrenar redes con millones de parámetros

### Aplicaciones Prácticas

Backpropagation es el motor detrás de:
- **Visión por Computadora**: Redes que reconocen objetos, rostros, escenas
- **PLN**: Modelos de lenguaje como GPT, BERT, traducción automática
- **Generación**: GANs que crean imágenes realistas, música, texto
- **Juegos**: AlphaGo, agentes que aprenden a jugar videojuegos
- **Ciencia**: Predicción de estructuras de proteínas, simulaciones físicas

Todos estos avances serían imposibles sin backpropagation.

## 🤔 Preguntas de Reflexión Iniciales

Antes de comenzar, reflexiona sobre estas preguntas:

1. ¿Por qué necesitamos calcular gradientes para entrenar una red neuronal?
2. ¿Qué significa "eficiencia computacional" en el contexto de cálculo de gradientes?
3. Si una red tiene 1 millón de parámetros, ¿cuántas derivadas parciales necesitamos calcular?
4. ¿Cómo podríamos verificar que nuestro cálculo de gradientes es correcto?
5. ¿Qué información del forward pass necesitamos guardar para backpropagation?

## 🔬 Parte 1: Regla de la Cadena y Grafos Computacionales (40 min)

### 1.1 Repaso de la Regla de la Cadena

Comencemos con ejemplos matemáticos simples antes de aplicarlo a redes neuronales.

**Ejemplo 1: Función compuesta simple**

```python
import numpy as np

# Función: y = (3x + 2)²
# Queremos: dy/dx

def forward_example1(x):
    """Forward pass: calcular y"""
    u = 3 * x + 2
    y = u ** 2
    return y, u  # Guardamos u para backprop

def backward_example1(x):
    """Backward pass: calcular dy/dx"""
    # Forward (calcular y guardar valores intermedios)
    u = 3 * x + 2
    y = u ** 2
    
    # Backward (aplicar regla de la cadena)
    dy_du = 2 * u      # ∂y/∂u = 2u
    du_dx = 3          # ∂u/∂x = 3
    dy_dx = dy_du * du_dx  # Regla de la cadena
    
    return dy_dx

# Probar
x = 5.0
y, u = forward_example1(x)
print(f"x = {x}")
print(f"u = 3x + 2 = {u}")
print(f"y = u² = {y}")

gradient = backward_example1(x)
print(f"dy/dx = {gradient}")
# Resultado: dy/dx = 2(3x + 2) * 3 = 6(17) = 102
```

**Actividad 1.1:** Implementa y verifica el gradiente de `y = sin(2x + 1)`

**Ejemplo 2: Múltiples variables**

```python
# Función: z = x² + y² + 2xy
# Queremos: ∂z/∂x, ∂z/∂y

def forward_example2(x, y):
    """Forward pass"""
    z = x**2 + y**2 + 2*x*y
    return z

def backward_example2(x, y):
    """Backward pass"""
    # Derivadas parciales
    dz_dx = 2*x + 2*y  # ∂z/∂x = 2x + 2y
    dz_dy = 2*y + 2*x  # ∂z/∂y = 2y + 2x
    
    return dz_dx, dz_dy

# Probar
x, y = 3.0, 4.0
z = forward_example2(x, y)
dz_dx, dz_dy = backward_example2(x, y)

print(f"z({x}, {y}) = {z}")
print(f"∂z/∂x = {dz_dx}")
print(f"∂z/∂y = {dz_dy}")
```

### 1.2 Grafos Computacionales

Los grafos computacionales son herramientas visuales poderosas para entender backpropagation.

**Ejemplo: z = (x + y) × w**

```python
class ComputationNode:
    """Nodo en un grafo computacional"""
    def __init__(self, name):
        self.name = name
        self.value = None
        self.grad = 0
    
    def __repr__(self):
        return f"{self.name}={self.value:.4f}, grad={self.grad:.4f}"

def forward_graph_example():
    """Forward pass con grafo computacional"""
    # Crear nodos
    x = ComputationNode('x')
    y = ComputationNode('y')
    w = ComputationNode('w')
    q = ComputationNode('q')  # q = x + y
    z = ComputationNode('z')  # z = q * w
    
    # Valores de entrada
    x.value = 2.0
    y.value = 3.0
    w.value = 4.0
    
    # Forward pass
    q.value = x.value + y.value  # q = 5.0
    z.value = q.value * w.value  # z = 20.0
    
    return x, y, w, q, z

def backward_graph_example(x, y, w, q, z):
    """Backward pass con grafo computacional"""
    # Inicializar gradiente de salida
    z.grad = 1.0  # dL/dz = 1 (asumimos L = z)
    
    # Backward pass (en orden inverso)
    # z = q * w
    q.grad += z.grad * w.value  # ∂z/∂q = w
    w.grad += z.grad * q.value  # ∂z/∂w = q
    
    # q = x + y
    x.grad += q.grad * 1.0  # ∂q/∂x = 1
    y.grad += q.grad * 1.0  # ∂q/∂y = 1

# Ejecutar
x, y, w, q, z = forward_graph_example()
print("Forward pass:")
print(x, y, w, q, z)

backward_graph_example(x, y, w, q, z)
print("\nBackward pass:")
print(x, y, w, q, z)
```

**Actividad 1.2:** Dibuja el grafo computacional para `f = (x + y) × (x - y)` y calcula todos los gradientes.

### 1.3 Operaciones Básicas y sus Gradientes

Tabla de referencia para operaciones comunes:

```python
class GradientOperations:
    """Colección de operaciones con sus gradientes"""
    
    @staticmethod
    def add_forward(x, y):
        return x + y
    
    @staticmethod
    def add_backward(dout, x, y):
        """∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1"""
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    
    @staticmethod
    def mul_forward(x, y):
        return x * y
    
    @staticmethod
    def mul_backward(dout, x, y):
        """∂(x*y)/∂x = y, ∂(x*y)/∂y = x"""
        dx = dout * y
        dy = dout * x
        return dx, dy
    
    @staticmethod
    def square_forward(x):
        return x ** 2
    
    @staticmethod
    def square_backward(dout, x):
        """∂(x²)/∂x = 2x"""
        dx = dout * 2 * x
        return dx
    
    @staticmethod
    def exp_forward(x):
        return np.exp(x)
    
    @staticmethod
    def exp_backward(dout, x):
        """∂(e^x)/∂x = e^x"""
        dx = dout * np.exp(x)
        return dx

# Ejemplo de uso
ops = GradientOperations()

# Forward
x, y = 3.0, 4.0
z = ops.mul_forward(x, y)  # z = 12
print(f"z = {z}")

# Backward (asumiendo dL/dz = 1)
dout = 1.0
dx, dy = ops.mul_backward(dout, x, y)
print(f"∂z/∂x = {dx}, ∂z/∂y = {dy}")  # ∂z/∂x = 4, ∂z/∂y = 3
```

**Actividad 1.3:** Implementa gradientes para división, exponencial y logaritmo.

## 🔬 Parte 2: Backpropagation en una Neurona (45 min)

### 2.1 Anatomía de una Neurona con Backpropagation

Implementemos una neurona que puede hacer forward y backward pass.

```python
class NeuronWithBackprop:
    """Neurona con capacidad de backpropagation"""
    
    def __init__(self, n_inputs):
        """Inicializar pesos y bias aleatoriamente"""
        self.w = np.random.randn(n_inputs) * 0.1
        self.b = np.random.randn() * 0.1
        
        # Cache para backpropagation
        self.cache = {}
        
        # Gradientes
        self.dw = np.zeros_like(self.w)
        self.db = 0
    
    def forward(self, x):
        """
        Forward pass: z = w·x + b
        Guardamos x para usarlo en backprop
        """
        self.cache['x'] = x
        z = np.dot(self.w, x) + self.b
        self.cache['z'] = z
        return z
    
    def backward(self, dz):
        """
        Backward pass
        Entrada: dz = ∂L/∂z (gradiente de la pérdida respecto a z)
        Salida: dx = ∂L/∂x (gradiente para propagar hacia atrás)
        
        Derivadas:
        ∂z/∂w = x  →  ∂L/∂w = ∂L/∂z * ∂z/∂w = dz * x
        ∂z/∂b = 1  →  ∂L/∂b = ∂L/∂z * ∂z/∂b = dz * 1
        ∂z/∂x = w  →  ∂L/∂x = ∂L/∂z * ∂z/∂x = dz * w
        """
        x = self.cache['x']
        
        # Calcular gradientes
        self.dw = dz * x  # ∂L/∂w
        self.db = dz      # ∂L/∂b
        dx = dz * self.w  # ∂L/∂x (para propagar)
        
        return dx
    
    def update(self, learning_rate=0.01):
        """Actualizar parámetros usando gradient descent"""
        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db

# Ejemplo de uso
neuron = NeuronWithBackprop(n_inputs=3)

# Forward pass
x = np.array([1.0, 2.0, 3.0])
z = neuron.forward(x)
print(f"Salida de la neurona: {z}")

# Backward pass (simulando dL/dz = 1)
dz = 1.0
dx = neuron.backward(dz)

print(f"Gradientes:")
print(f"  dL/dw = {neuron.dw}")
print(f"  dL/db = {neuron.db}")
print(f"  dL/dx = {dx}")

# Actualizar parámetros
neuron.update(learning_rate=0.1)
print(f"\nPesos actualizados: {neuron.w}")
print(f"Bias actualizado: {neuron.b}")
```

### 2.2 Neurona con Función de Activación

Agreguemos una función de activación (ReLU):

```python
class NeuronWithActivation:
    """Neurona con función de activación y backprop"""
    
    def __init__(self, n_inputs):
        self.w = np.random.randn(n_inputs) * 0.1
        self.b = np.random.randn() * 0.1
        self.cache = {}
        self.dw = np.zeros_like(self.w)
        self.db = 0
    
    def relu(self, z):
        """ReLU: max(0, z)"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivada de ReLU: 1 si z > 0, else 0"""
        return (z > 0).astype(float)
    
    def forward(self, x):
        """Forward pass: a = ReLU(w·x + b)"""
        self.cache['x'] = x
        
        # Suma ponderada
        z = np.dot(self.w, x) + self.b
        self.cache['z'] = z
        
        # Activación
        a = self.relu(z)
        self.cache['a'] = a
        
        return a
    
    def backward(self, da):
        """
        Backward pass con activación
        
        Entrada: da = ∂L/∂a
        
        Pasos:
        1. dz = da * ReLU'(z)  (gradiente local de ReLU)
        2. dw = dz * x
        3. db = dz
        4. dx = dz * w
        """
        x = self.cache['x']
        z = self.cache['z']
        
        # Gradiente a través de ReLU
        dz = da * self.relu_derivative(z)
        
        # Gradientes de parámetros
        self.dw = dz * x
        self.db = dz
        
        # Gradiente para propagar
        dx = dz * self.w
        
        return dx
    
    def update(self, lr=0.01):
        self.w -= lr * self.dw
        self.b -= lr * self.db

# Ejemplo
neuron = NeuronWithActivation(n_inputs=3)

# Forward
x = np.array([1.0, -2.0, 3.0])
a = neuron.forward(x)
print(f"Input: {x}")
print(f"z (before ReLU): {neuron.cache['z']}")
print(f"a (after ReLU): {a}")

# Backward
da = 1.0  # Gradiente de entrada
dx = neuron.backward(da)

print(f"\nGradientes:")
print(f"  dL/dw = {neuron.dw}")
print(f"  dL/db = {neuron.db}")
print(f"  dL/dx = {dx}")
```

**Actividad 2.1:** Implementa `NeuronWithSigmoid` que use función sigmoid en lugar de ReLU.

### 2.3 Ejemplo Completo: Entrenar una Neurona

Entrenemos una neurona para aprender la función AND:

```python
# Datos: AND lógico
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_train = np.array([0, 0, 0, 1])  # AND

# Crear neurona
neuron = NeuronWithActivation(n_inputs=2)

# Función de pérdida MSE
def mse_loss(pred, target):
    return 0.5 * (pred - target) ** 2

def mse_derivative(pred, target):
    return pred - target

# Entrenar
learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    total_loss = 0
    
    for x, y_true in zip(X_train, y_train):
        # Forward
        y_pred = neuron.forward(x)
        loss = mse_loss(y_pred, y_true)
        total_loss += loss
        
        # Backward
        dloss = mse_derivative(y_pred, y_true)
        neuron.backward(dloss)
        
        # Update
        neuron.update(learning_rate)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Evaluar
print("\nResultados finales:")
for x, y_true in zip(X_train, y_train):
    y_pred = neuron.forward(x)
    print(f"Input: {x}, Predicción: {y_pred:.4f}, Real: {y_true}")
```

**Actividad 2.2:** Entrena una neurona para aprender OR y XOR. ¿Qué observas con XOR?

## 🔬 Parte 3: Backpropagation en Redes Multicapa (60 min)

### 3.1 Red de 2 Capas con Backpropagation

Implementemos una red completa con dos capas:

```python
class Layer:
    """Capa densa con backpropagation"""
    
    def __init__(self, n_inputs, n_neurons, activation='relu'):
        # Inicialización He para ReLU
        self.W = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros((1, n_neurons))
        self.activation = activation
        self.cache = {}
        
    def forward(self, X):
        """
        Forward pass
        X: (batch_size, n_inputs)
        Salida: (batch_size, n_neurons)
        """
        self.cache['X'] = X
        
        # Z = X @ W + b
        Z = np.dot(X, self.W) + self.b
        self.cache['Z'] = Z
        
        # Activación
        if self.activation == 'relu':
            A = np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
        elif self.activation == 'linear':
            A = Z
        
        self.cache['A'] = A
        return A
    
    def backward(self, dA):
        """
        Backward pass
        dA: gradiente de la pérdida respecto a A (salida de esta capa)
        Retorna: dX (gradiente para propagar a capa anterior)
        """
        X = self.cache['X']
        Z = self.cache['Z']
        m = X.shape[0]  # batch size
        
        # Gradiente de la activación
        if self.activation == 'relu':
            dZ = dA * (Z > 0)
        elif self.activation == 'sigmoid':
            A = self.cache['A']
            dZ = dA * A * (1 - A)
        elif self.activation == 'linear':
            dZ = dA
        
        # Gradientes de parámetros
        self.dW = (1/m) * np.dot(X.T, dZ)
        self.db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        
        # Gradiente para propagar
        dX = np.dot(dZ, self.W.T)
        
        return dX
    
    def update(self, lr):
        """Actualizar parámetros"""
        self.W -= lr * self.dW
        self.b -= lr * self.db


class TwoLayerNetwork:
    """Red neuronal de 2 capas con backpropagation"""
    
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.layer1 = Layer(n_inputs, n_hidden, activation='relu')
        self.layer2 = Layer(n_hidden, n_outputs, activation='sigmoid')
    
    def forward(self, X):
        """Forward pass a través de ambas capas"""
        A1 = self.layer1.forward(X)
        A2 = self.layer2.forward(A1)
        return A2
    
    def backward(self, dA2):
        """Backward pass a través de ambas capas"""
        dA1 = self.layer2.backward(dA2)
        dX = self.layer1.backward(dA1)
        return dX
    
    def update(self, lr):
        """Actualizar parámetros de ambas capas"""
        self.layer1.update(lr)
        self.layer2.update(lr)
    
    def train_step(self, X, y, lr=0.01):
        """Un paso de entrenamiento completo"""
        # Forward
        y_pred = self.forward(X)
        
        # Calcular pérdida (Binary Cross-Entropy)
        m = y.shape[0]
        loss = -np.mean(y * np.log(y_pred + 1e-8) + (1-y) * np.log(1-y_pred + 1e-8))
        
        # Backward
        dA2 = y_pred - y  # Gradiente de BCE con sigmoid
        self.backward(dA2)
        
        # Update
        self.update(lr)
        
        return loss

# Ejemplo: Entrenar en XOR (¡ahora sí funciona!)
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_train = np.array([[0], [1], [1], [0]])  # XOR

# Crear red
net = TwoLayerNetwork(n_inputs=2, n_hidden=4, n_outputs=1)

# Entrenar
print("Entrenando en XOR...")
for epoch in range(5000):
    loss = net.train_step(X_train, y_train, lr=0.5)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluar
print("\nResultados finales:")
predictions = net.forward(X_train)
for i, (x, y_true, y_pred) in enumerate(zip(X_train, y_train, predictions)):
    print(f"Input: {x}, Predicción: {y_pred[0]:.4f}, Real: {y_true[0]}")
```

### 3.2 Visualización de Gradientes

Es útil visualizar cómo fluyen los gradientes:

```python
import matplotlib.pyplot as plt

def visualize_gradients(network, X, y):
    """Visualiza magnitudes de gradientes en cada capa"""
    # Forward
    y_pred = network.forward(X)
    
    # Backward
    dA2 = y_pred - y
    network.backward(dA2)
    
    # Recopilar magnitudes de gradientes
    grad_layer1 = np.mean(np.abs(network.layer1.dW))
    grad_layer2 = np.mean(np.abs(network.layer2.dW))
    
    # Graficar
    layers = ['Layer 1', 'Layer 2']
    gradients = [grad_layer1, grad_layer2]
    
    plt.figure(figsize=(10, 5))
    plt.bar(layers, gradients, color=['blue', 'red'])
    plt.ylabel('Magnitud Promedio del Gradiente')
    plt.title('Flujo de Gradientes en la Red')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Gradiente capa 1: {grad_layer1:.6f}")
    print(f"Gradiente capa 2: {grad_layer2:.6f}")

# Usar
visualize_gradients(net, X_train, y_train)
```

**Actividad 3.1:** Crea una red de 3 capas y entrénala en un dataset de clasificación simple.

## 🔬 Parte 4: Verificación de Gradientes (30 min)

### 4.1 Gradient Checking

La verificación numérica de gradientes es CRUCIAL para asegurar que backpropagation esté implementado correctamente.

```python
def numerical_gradient(f, x, epsilon=1e-5):
    """
    Calcula gradiente numérico usando diferencias finitas
    
    f: función que toma x y retorna un escalar
    x: punto donde calcular el gradiente
    epsilon: pequeño valor para la diferencia finita
    """
    grad = np.zeros_like(x)
    
    # Iterar sobre cada dimensión
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # f(x + epsilon)
        x[idx] = old_value + epsilon
        fxplus = f(x)
        
        # f(x - epsilon)
        x[idx] = old_value - epsilon
        fxminus = f(x)
        
        # Gradiente numérico
        grad[idx] = (fxplus - fxminus) / (2 * epsilon)
        
        # Restaurar valor
        x[idx] = old_value
        it.iternext()
    
    return grad

def gradient_check(network, X, y, epsilon=1e-5):
    """
    Verifica que los gradientes analíticos coincidan con los numéricos
    """
    # Forward y backward para obtener gradientes analíticos
    y_pred = network.forward(X)
    loss_initial = -np.mean(y * np.log(y_pred + 1e-8) + (1-y) * np.log(1-y_pred + 1e-8))
    
    dA2 = y_pred - y
    network.backward(dA2)
    
    # Gradientes analíticos
    analytical_dW1 = network.layer1.dW.copy()
    analytical_dW2 = network.layer2.dW.copy()
    
    # Función de pérdida para gradient checking
    def loss_function(params):
        # Desempaquetar parámetros
        W1, b1, W2, b2 = params
        
        # Forward temporal
        A1 = np.maximum(0, np.dot(X, W1) + b1)
        A2 = 1 / (1 + np.exp(-(np.dot(A1, W2) + b2)))
        
        # Pérdida
        loss = -np.mean(y * np.log(A2 + 1e-8) + (1-y) * np.log(1-A2 + 1e-8))
        return loss
    
    # Calcular gradiente numérico solo para W1 (por simplicidad)
    print("Verificando gradientes de W1...")
    
    numerical_dW1 = np.zeros_like(network.layer1.W)
    
    for i in range(network.layer1.W.shape[0]):
        for j in range(network.layer1.W.shape[1]):
            # Perturbar W1[i,j]
            network.layer1.W[i, j] += epsilon
            loss_plus = loss_function([network.layer1.W, network.layer1.b, 
                                      network.layer2.W, network.layer2.b])
            
            network.layer1.W[i, j] -= 2 * epsilon
            loss_minus = loss_function([network.layer1.W, network.layer1.b,
                                       network.layer2.W, network.layer2.b])
            
            # Gradiente numérico
            numerical_dW1[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restaurar
            network.layer1.W[i, j] += epsilon
    
    # Comparar
    diff = np.linalg.norm(analytical_dW1 - numerical_dW1) / (
           np.linalg.norm(analytical_dW1) + np.linalg.norm(numerical_dW1))
    
    print(f"\nDiferencia relativa: {diff:.10f}")
    
    if diff < 1e-7:
        print("✓ ¡Gradientes correctos!")
    elif diff < 1e-5:
        print("⚠ Gradientes probablemente correctos (diferencia pequeña)")
    else:
        print("✗ ERROR: Gradientes incorrectos")
    
    return diff

# Ejecutar gradient checking
print("=== GRADIENT CHECKING ===")
difference = gradient_check(net, X_train, y_train)
```

### 4.2 Consejos para Debugging

```python
def debug_backprop(network, X, y):
    """Herramienta de debugging para backpropagation"""
    
    print("=== DEBUG BACKPROPAGATION ===\n")
    
    # Forward
    print("1. FORWARD PASS")
    A1 = network.layer1.forward(X)
    A2 = network.layer2.forward(A1)
    print(f"   Salida capa 1: shape={A1.shape}, min={A1.min():.4f}, max={A1.max():.4f}")
    print(f"   Salida capa 2: shape={A2.shape}, min={A2.min():.4f}, max={A2.max():.4f}")
    
    # Pérdida
    loss = -np.mean(y * np.log(A2 + 1e-8) + (1-y) * np.log(1-A2 + 1e-8))
    print(f"   Pérdida: {loss:.6f}")
    
    # Backward
    print("\n2. BACKWARD PASS")
    dA2 = A2 - y
    print(f"   Gradiente inicial (dA2): shape={dA2.shape}, mean={np.mean(np.abs(dA2)):.6f}")
    
    dA1 = network.layer2.backward(dA2)
    print(f"   Gradiente capa 2 -> 1 (dA1): shape={dA1.shape}, mean={np.mean(np.abs(dA1)):.6f}")
    print(f"   Gradiente W2: mean={np.mean(np.abs(network.layer2.dW)):.6f}")
    
    dX = network.layer1.backward(dA1)
    print(f"   Gradiente capa 1 -> entrada: mean={np.mean(np.abs(dX)):.6f}")
    print(f"   Gradiente W1: mean={np.mean(np.abs(network.layer1.dW)):.6f}")
    
    # Verificar NaN o Inf
    print("\n3. VERIFICACIONES")
    has_nan = np.isnan(network.layer1.dW).any() or np.isnan(network.layer2.dW).any()
    has_inf = np.isinf(network.layer1.dW).any() or np.isinf(network.layer2.dW).any()
    
    if has_nan:
        print("   ✗ ¡ADVERTENCIA! Gradientes contienen NaN")
    if has_inf:
        print("   ✗ ¡ADVERTENCIA! Gradientes contienen Inf")
    if not has_nan and not has_inf:
        print("   ✓ Gradientes son valores numéricos válidos")

# Usar
debug_backprop(net, X_train, y_train)
```

**Actividad 4.1:** Introduce un bug intencional en tu código de backpropagation y usa gradient checking para encontrarlo.

## 📊 Análisis Final de Rendimiento

### Comparación: Antes vs Después de Backpropagation

```python
# Sin entrenar (pesos aleatorios)
net_untrained = TwoLayerNetwork(n_inputs=2, n_hidden=4, n_outputs=1)
pred_before = net_untrained.forward(X_train)

# Entrenar
net_trained = TwoLayerNetwork(n_inputs=2, n_hidden=4, n_outputs=1)
losses = []

for epoch in range(5000):
    loss = net_trained.train_step(X_train, y_train, lr=0.5)
    losses.append(loss)

pred_after = net_trained.forward(X_train)

# Comparar
print("=== ANTES DEL ENTRENAMIENTO ===")
for i, (x, y_true, y_pred) in enumerate(zip(X_train, y_train, pred_before)):
    print(f"Input: {x}, Pred: {y_pred[0]:.4f}, Real: {y_true[0]}")

print("\n=== DESPUÉS DEL ENTRENAMIENTO ===")
for i, (x, y_true, y_pred) in enumerate(zip(X_train, y_train, pred_after)):
    print(f"Input: {x}, Pred: {y_pred[0]:.4f}, Real: {y_true[0]}")

# Graficar curva de aprendizaje
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Curva de Aprendizaje - Backpropagation en Acción')
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nPérdida inicial: {losses[0]:.6f}")
print(f"Pérdida final: {losses[-1]:.6f}")
print(f"Mejora: {(1 - losses[-1]/losses[0]) * 100:.2f}%")
```

### Problemas Comunes y Soluciones

```python
class BackpropDiagnostics:
    """Herramientas para diagnosticar problemas en backpropagation"""
    
    @staticmethod
    def check_vanishing_gradients(network, threshold=1e-7):
        """Detecta gradientes que desaparecen"""
        grad1 = np.mean(np.abs(network.layer1.dW))
        grad2 = np.mean(np.abs(network.layer2.dW))
        
        print("=== DIAGNÓSTICO: VANISHING GRADIENTS ===")
        print(f"Gradiente promedio capa 1: {grad1:.10f}")
        print(f"Gradiente promedio capa 2: {grad2:.10f}")
        
        if grad1 < threshold or grad2 < threshold:
            print("⚠ ¡ADVERTENCIA! Posible vanishing gradients")
            print("Soluciones:")
            print("  - Usar ReLU en lugar de sigmoid/tanh")
            print("  - Reducir número de capas")
            print("  - Usar batch normalization")
            print("  - Mejor inicialización de pesos")
        else:
            print("✓ Gradientes en rango saludable")
    
    @staticmethod
    def check_exploding_gradients(network, threshold=1.0):
        """Detecta gradientes que explotan"""
        grad1 = np.mean(np.abs(network.layer1.dW))
        grad2 = np.mean(np.abs(network.layer2.dW))
        
        print("\n=== DIAGNÓSTICO: EXPLODING GRADIENTS ===")
        
        if grad1 > threshold or grad2 > threshold:
            print("⚠ ¡ADVERTENCIA! Posible exploding gradients")
            print("Soluciones:")
            print("  - Reducir learning rate")
            print("  - Usar gradient clipping")
            print("  - Mejor inicialización de pesos")
        else:
            print("✓ Gradientes bajo control")

# Usar diagnósticos
diag = BackpropDiagnostics()
diag.check_vanishing_gradients(net_trained)
diag.check_exploding_gradients(net_trained)
```

## 🎯 EJERCICIOS PROPUESTOS

### Nivel Básico

**Ejercicio 1:** Implementación de Operaciones Básicas
```
Implementa una clase para cada operación (suma, multiplicación, división)
con métodos forward() y backward(). Verifica con gradient checking.
```

**Ejercicio 2:** Grafo Computacional Manual
```
Para la función f(x,y,z) = (x + y) * z:
a) Dibuja el grafo computacional
b) Calcula el forward pass con x=2, y=3, z=4
c) Calcula el backward pass manualmente
d) Verifica con código
```

**Ejercicio 3:** Funciones de Activación
```
Implementa forward y backward para:
- Sigmoid
- Tanh
- Leaky ReLU (con α=0.01)
Verifica cada una con gradient checking.
```

### Nivel Intermedio

**Ejercicio 4:** Red de 3 Capas
```
Implementa una red con arquitectura: 4 → 8 → 4 → 2
- Usa ReLU en capas ocultas, softmax en salida
- Entrena en un dataset de clasificación multiclase
- Visualiza la evolución de los gradientes
```

**Ejercicio 5:** Regularización L2
```
Agrega regularización L2 a tu red:
- Modifica la función de pérdida: L = L_data + λ||W||²
- Implementa el gradiente correspondiente
- Compara resultados con y sin regularización
```

**Ejercicio 6:** Mini-batch SGD
```
Implementa entrenamiento con mini-batches:
- Divide los datos en batches de tamaño 32
- Implementa un epoch completo iterando sobre batches
- Compara tiempo de entrenamiento vs batch completo
```

### Nivel Avanzado

**Ejercicio 7:** Arquitectura Profunda
```
Crea una red de 5+ capas:
- Implementa desde cero (no usar frameworks)
- Entrena en MNIST
- Diagnostica y soluciona vanishing gradients
- Usa diferentes inicializaciones (He, Xavier)
```

**Ejercicio 8:** Gradient Checking Completo
```
Implementa gradient checking para:
- Todos los parámetros (W y b) de todas las capas
- Diferentes funciones de pérdida
- Crea un informe detallado de diferencias
```

**Ejercicio 9:** Optimizador con Momentum
```
Implementa backpropagation con momentum:
- v = β*v + (1-β)*gradiente
- W = W - α*v
- Compara convergencia con SGD vanilla
```

## 📝 Entregables

### 1. Código Fuente
- `backprop.py`: Implementación de backpropagation
- `layers.py`: Clases de capas con forward/backward
- `network.py`: Red neuronal completa
- `gradient_check.py`: Verificación de gradientes
- `experiments.ipynb`: Notebook con experimentos

### 2. Documentación
- README explicando tu implementación
- Comentarios detallados en el código
- Diagramas de grafos computacionales

### 3. Resultados
- Curvas de aprendizaje
- Resultados de gradient checking
- Comparación de diferentes configuraciones
- Análisis de errores

### 4. Reporte Técnico (2-3 páginas)
Incluir:
- Explicación de tu implementación
- Decisiones de diseño
- Resultados experimentales
- Dificultades encontradas y soluciones
- Conclusiones

## 🎯 Criterios de Evaluación (CDIO)

### Conceive (Concebir) - 25%
- [ ] Comprensión profunda de la regla de la cadena
- [ ] Identificación correcta de gradientes necesarios
- [ ] Diseño apropiado de la arquitectura de código
- [ ] Planificación de estrategia de verificación

### Design (Diseñar) - 25%
- [ ] Implementación correcta del algoritmo de backpropagation
- [ ] Código modular y reutilizable
- [ ] Manejo apropiado de dimensiones matriciales
- [ ] Implementación eficiente (vectorización)

### Implement (Implementar) - 30%
- [ ] Código funcional sin errores
- [ ] Gradient checking pasa (diferencia < 1e-7)
- [ ] Red neuronal entrena correctamente
- [ ] Resultados reproducibles

### Operate (Operar) - 20%
- [ ] Experimentación con diferentes configuraciones
- [ ] Análisis crítico de resultados
- [ ] Identificación y solución de problemas
- [ ] Documentación clara y completa

## 📋 Rúbrica de Evaluación

| Criterio | Excelente (90-100%) | Bueno (75-89%) | Satisfactorio (60-74%) | Insuficiente (<60%) |
|----------|-------------------|--------------|---------------------|------------------|
| **Implementación** | Backprop perfecto, gradient check < 1e-9 | Backprop correcto, < 1e-7 | Backprop funciona, < 1e-5 | Errores en implementación |
| **Comprensión** | Explica detalladamente cada paso | Explica conceptos principales | Explica parcialmente | Comprensión limitada |
| **Código** | Muy limpio, modular, documentado | Bien estructurado | Funcional pero básico | Desorganizado o con errores |
| **Experimentos** | Análisis profundo, múltiples experimentos | Buenos experimentos | Experimentos básicos | Experimentos insuficientes |
| **Documentación** | Excelente, clara, completa | Buena documentación | Documentación básica | Documentación pobre |

## 📚 Referencias Adicionales

### Artículos Fundamentales
1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors". Nature.
2. LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition"
3. Glorot, X., & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks"

### Recursos Online
- **CS231n Stanford**: http://cs231n.stanford.edu/ (especialmente módulo sobre backpropagation)
- **Deep Learning Book** (Goodfellow): Capítulo 6 - Deep Feedforward Networks
- **3Blue1Brown**: Serie de videos sobre backpropagation (muy visual)
- **Andrej Karpathy**: "Yes you should understand backprop" (blog post)

### Herramientas
- NumPy documentation: https://numpy.org/doc/
- Matplotlib para visualización
- autograd (para verificar gradientes automáticamente)

### Papers Adicionales
- "Delving Deep into Rectifiers" (He et al., 2015) - Sobre inicialización
- "Batch Normalization" (Ioffe & Szegedy, 2015)
- "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, 2010)

## 🎓 Notas Finales

### Consejos para el Éxito

1. **Siempre verifica tus gradientes**: Gradient checking es tu mejor amigo. Un error pequeño en backprop puede arruinar todo el entrenamiento.

2. **Dibuja grafos computacionales**: Antes de programar, dibuja el grafo. Te ayudará a visualizar el flujo de gradientes.

3. **Empieza simple**: Implementa y verifica operaciones simples antes de construir redes complejas.

4. **Usa dimensiones explícitas**: Siempre conoce las dimensiones de tus tensores. Muchos bugs vienen de errores de dimensión.

5. **Guarda valores intermedios**: En forward pass, guarda todo lo que necesitarás en backward pass.

### Errores Comunes

❌ **Olvidar transponer matrices**: `dW = X.T @ dZ` (no `X @ dZ`)
❌ **No sumar gradientes**: Al calcular `db`, hay que sumar sobre el batch
❌ **Confundir `*` y `@`**: `*` es elemento-wise, `@` es producto matricial
❌ **No inicializar gradientes a cero**: Acumular gradientes sin limpiar
❌ **Gradient checking con batch grande**: Usa batches pequeños para verificar

### Reflexión Final

**Backpropagation es el corazón del deep learning**. Es la diferencia entre "redes neuronales interesantes en teoría" y "redes neuronales que revolucionan la tecnología".

Dominar backpropagation te da:
- Comprensión profunda de cómo aprenden las redes
- Capacidad de debuggear problemas de entrenamiento
- Habilidad para implementar arquitecturas personalizadas
- Base sólida para frameworks modernos (PyTorch, TensorFlow)

**Un mensaje importante**: Una vez que entiendas backpropagation a fondo, probablemente nunca lo implementarás manualmente otra vez. Los frameworks modernos lo hacen automáticamente. Pero ese conocimiento profundo te hará un practicante mucho mejor de deep learning.

### Próximos Pasos

En el siguiente laboratorio (Lab 06), usaremos backpropagation para:
- Implementar loops de entrenamiento completos
- Trabajar con datasets reales
- Implementar técnicas avanzadas (early stopping, regularización)
- Optimizar el rendimiento del entrenamiento

¡Ahora tienes la herramienta más poderosa en deep learning. Úsala sabiamente! 🚀

---

**"Understanding backpropagation is the difference between being a machine learning user and a machine learning practitioner."** - Andrej Karpathy

**¡Backpropagation es la magia del deep learning! ✨**
