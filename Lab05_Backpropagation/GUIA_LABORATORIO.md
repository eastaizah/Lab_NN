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

**¿Qué hacemos?** Revisamos la regla de la cadena del cálculo diferencial y la aplicamos a funciones compuestas como las que aparecen en redes neuronales.

**¿Por qué lo hacemos?** Una red neuronal es esencialmente una función compuesta de muchas capas: `L = f_n(f_{n-1}(... f_1(x)))`. Para ajustar cualquier parámetro, necesitamos calcular `∂L/∂w`. La regla de la cadena nos permite **descomponer** esa derivada compleja en una cadena de derivadas locales simples:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdots \frac{\partial a_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial w}$$

Sin la regla de la cadena, calcular gradientes en una red de 100 capas sería matemáticamente intratable. Con ella, basta con que cada capa conozca su **gradiente local** y sepa multiplicarlo por el gradiente que llega desde las capas superiores.

**¿Cómo lo hacemos?** Introducimos la notación de variable intermedia `u`:

$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}$$

**Analogía del termostato:** Imagina que el consumo eléctrico `E` depende de la temperatura de la habitación `T`, y `T` depende de la posición del termostato `p`. Para saber cuánto afecta el termostato al consumo —es decir, `∂E/∂p`— multiplicamos "cuánto cambia el consumo por grado" (`∂E/∂T`) por "cuánto cambia la temperatura por posición" (`∂T/∂p`). Eso es exactamente la regla de la cadena aplicada a una neurona.

**¿Qué resultados esperar?** Gradientes que coincidan exactamente con los calculados por diferenciación analítica directa. La regla de la cadena no es una aproximación: es matemáticamente exacta.

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

**¿Qué hacemos?** Representamos cálculos matemáticos como un grafo dirigido donde los nodos son operaciones y las aristas son el flujo de datos (y de gradientes).

**¿Por qué lo hacemos?** Los grafos computacionales convierten backpropagation en un procedimiento **sistemático y automático**. En lugar de derivar manualmente una función monolítica compleja, cada nodo del grafo solo necesita conocer su operación local y aplicar la regla de la cadena hacia atrás:

```
Forward pass  →→→→→→→  (izquierda a derecha): calcular salidas
Backward pass ←←←←←←  (derecha a izquierda): propagar gradientes
```

Esta separación limpia es la razón por la que frameworks como PyTorch o TensorFlow pueden calcular gradientes automáticamente para cualquier arquitectura: construyen el grafo en el forward pass y lo recorren en reversa durante el backward pass.

**¿Cómo lo hacemos?** Cada nodo almacena:
1. Su **valor** (calculado en el forward pass)
2. Su **gradiente acumulado** (calculado en el backward pass)
3. Cómo propagar el gradiente hacia sus entradas (la "puerta local")

**¿Qué resultados esperar?** Al final del backward pass, cada nodo tendrá el gradiente correcto `∂L/∂nodo`, que es exactamente lo que necesitamos para actualizar los parámetros.

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

**¿Qué hacemos?** Catalogamos las operaciones primitivas más frecuentes en redes neuronales junto con sus gradientes locales.

**¿Por qué lo hacemos?** Cualquier función compleja —por ejemplo `sigmoid(w·x + b)`— puede descomponerse en una cadena de operaciones primitivas (suma, multiplicación, exponencial). Si conocemos el gradiente local de cada primitiva, podemos calcular el gradiente de cualquier composición simplemente multiplicando los gradientes locales (regla de la cadena).

**Tabla de gradientes de operaciones primitivas:**

| Operación | Forward: `z = f(x, y)` | Gradiente `∂z/∂x` | Gradiente `∂z/∂y` | Notas |
|-----------|------------------------|-------------------|-------------------|-------|
| Suma      | `z = x + y`            | `1`               | `1`               | Distribuye el gradiente igual a ambas entradas |
| Resta     | `z = x - y`            | `1`               | `-1`              | Invierte el signo hacia la segunda entrada |
| Multiplicación | `z = x * y`       | `y`               | `x`               | Cada entrada recibe el valor de la otra |
| División  | `z = x / y`            | `1/y`             | `-x/y²`           | Asimétrico: la entrada denominador tiene gradiente negativo |
| Cuadrado  | `z = x²`               | `2x`              | —                 | Requiere guardar `x` en caché |
| Exponencial | `z = eˣ`             | `eˣ`              | —                 | La derivada es ella misma; requiere guardar `z` en caché |
| Logaritmo | `z = ln(x)`            | `1/x`             | —                 | Solo válido para `x > 0`; gradiente explota cerca de 0 |
| ReLU      | `z = max(0, x)`        | `1 si x>0, 0 si x≤0` | —             | Corta el gradiente para activaciones negativas |
| Sigmoid   | `z = σ(x)`             | `σ(x)(1-σ(x))`   | —                 | Se satura en extremos → gradiente ≈ 0 |

**¿Cómo lo hacemos?** Implementamos cada operación como una clase con métodos `forward()` y `backward()`. Esto nos permite componerlas libremente para construir funciones arbitrariamente complejas.

**¿Qué resultados esperar?** Para cada operación, los gradientes numéricos y analíticos deben coincidir con una diferencia relativa menor a `1e-7`.

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

**¿Qué hacemos?** Implementamos una neurona que, además de realizar el forward pass (`z = w·x + b`), puede ejecutar el backward pass para calcular gradientes con respecto a sus parámetros.

**¿Por qué lo hacemos?** Una neurona tiene tres tipos de parámetros que necesitan gradientes:
- `∂L/∂w` → para actualizar los pesos y mejorar la predicción
- `∂L/∂b` → para actualizar el bias
- `∂L/∂x` → para **propagar** el gradiente hacia las capas anteriores (esta neurona no es la primera)

**La clave del caché:** Durante el forward pass, debemos guardar los valores intermedios que necesitaremos en el backward pass. Para la operación `z = w·x + b`:
- Necesitamos `x` para calcular `∂L/∂w = ∂L/∂z · x`
- Necesitamos `w` para calcular `∂L/∂x = ∂L/∂z · w`

Si no guardamos `x` durante el forward pass, no podemos calcular `∂L/∂w` durante el backward pass.

**¿Cómo lo hacemos?** Usamos un diccionario `cache` para almacenar los valores intermedios del forward pass. El backward pass recibe `dz = ∂L/∂z` (el gradiente que llega desde la capa siguiente) y calcula los tres gradientes usando la regla de la cadena:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = dz \cdot x$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = dz \cdot 1 = dz$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x} = dz \cdot w$$

**¿Qué resultados esperar?** Los gradientes calculados deben coincidir con los gradientes numéricos con precisión de al menos `1e-7`.

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

**¿Qué hacemos?** Extendemos la neurona para incluir una función de activación no lineal (ReLU) en el forward pass y su derivada en el backward pass.

**¿Por qué lo hacemos?** La función de activación introduce **no-linealidad** en la red, pero también crea una "compuerta" por la que debe pasar el gradiente. Sin considerar la activación en el backward pass, los gradientes serían incorrectos.

La cadena completa para una neurona con activación es:

$$\text{Forward:} \quad z = w \cdot x + b \xrightarrow{\text{ReLU}} a = \max(0, z)$$

$$\text{Backward:} \quad \frac{\partial L}{\partial a} \xrightarrow{\cdot \text{ReLU}'(z)} \frac{\partial L}{\partial z} \xrightarrow{\text{neurona}} \frac{\partial L}{\partial w}, \frac{\partial L}{\partial b}, \frac{\partial L}{\partial x}$$

El paso clave es `dz = da * ReLU'(z)`, donde `ReLU'(z) = 1 si z > 0, 0 si z ≤ 0`. Esto significa que cuando `z ≤ 0`, el gradiente se **bloquea completamente** (la neurona está "muerta" y no aprende). Para `z > 0`, el gradiente fluye sin modificación.

**Impacto en el flujo de gradientes:**
- **ReLU**: Flujo binario (0 o 1). Puede causar neuronas muertas, pero evita saturación.
- **Sigmoid**: Flujo suavizado (`σ(1-σ)`). Para valores extremos de `z`, el gradiente se acerca a cero → **gradiente desvaneciente**.
- **Tanh**: Similar a sigmoid pero con mejor simetría; aún puede saturarse.

**¿Qué resultados esperar?** Cuando `z < 0`, los gradientes `dw`, `db` y `dx` deben ser cero porque ReLU bloqueó el flujo. Cuando `z > 0`, el comportamiento debe ser idéntico al de la neurona sin activación.

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

**¿Qué hacemos?** Usamos nuestra neurona con backpropagation para aprender la función lógica AND mediante descenso por gradiente.

**¿Por qué AND y no XOR?** Una neurona individual —incluso con función de activación— solo puede aprender problemas **linealmente separables**: aquellos donde las clases pueden separarse con un hiperplano (una línea en 2D). Esto es una limitación fundamental:

```
AND: linealmente separable        XOR: NO linealmente separable
(0,0)→0  (0,1)→0                  (0,0)→0  (0,1)→1
(1,0)→0  (1,1)→1                  (1,0)→1  (1,1)→0

  y                                 y
  1 | . .                           1 | . x
  0 | . x          /línea/          0 | x .
     ------                            ------
     0  1  x                           0  1  x

Leyenda: x = clase 1, . = clase 0

✓ Una línea puede separar           ✗ Ninguna línea puede separar
  el "1" de los "0"                   los "1" de los "0"
```

La neurona con AND aprenderá correctamente. Con XOR, la pérdida nunca llegará a cero y las predicciones serán incorrectas. Esto demuestra por qué necesitamos **múltiples capas**: para aprender fronteras de decisión no lineales.

**¿Cómo lo hacemos?** Realizamos descenso por gradiente estocástico (SGD): para cada ejemplo de entrenamiento, ejecutamos forward pass, calculamos la pérdida MSE, ejecutamos backward pass y actualizamos los parámetros.

**¿Qué resultados esperar?**
- Para **AND**: La pérdida debe decrecer y los outputs deben acercarse a 0 para `[0,0]`, `[0,1]`, `[1,0]` y a 1 para `[1,1]`.
- Para **XOR** (Actividad 2.2): La pérdida se estancará y las predicciones serán imprecisas, evidenciando la limitación de las neuronas simples.

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

**¿Qué hacemos?** Implementamos una red neuronal con dos capas densas, cada una con su propio forward y backward pass, formando una cadena completa de backpropagation.

**¿Por qué múltiples capas?** Una capa oculta transforma los datos a un **nuevo espacio de representación** donde el problema puede volverse linealmente separable. Geométricamente:
- Una capa = un hiperplano (frontera lineal)
- Dos capas = múltiples hiperplanos combinados (regiones convexas)
- Tres o más capas = fronteras arbitrariamente complejas

Por esto la red de 2 capas puede aprender XOR (imposible para una sola neurona): la primera capa transforma el espacio, y la segunda separa linealmente la representación resultante.

**Inicialización He: ¿Por qué importa?**

La inicialización correcta de los pesos es crítica para evitar problemas desde el inicio del entrenamiento:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

Si los pesos son demasiado **pequeños** (ej. todos cero): todas las neuronas aprenden lo mismo (simetría perfecta), los gradientes son idénticos, y la red no puede aprender representaciones diversas.

Si los pesos son demasiado **grandes**: las activaciones se saturan desde el primer forward pass, los gradientes desaparecen o explotan antes de que empiece el entrenamiento.

La inicialización **He** (también conocida como **inicialización Kaiming**, propuesta por Kaiming He et al., 2015) está diseñada específicamente para ReLU: el factor `√(2/n_in)` compensa que ReLU desactiva aproximadamente la mitad de las neuronas, manteniendo la varianza de las activaciones constante a lo largo de la red durante las primeras iteraciones.

**¿Cómo lo hacemos?** El backward pass de la red sigue el orden inverso de las capas. El gradiente fluye de salida a entrada:

```
Forward:  X → [Capa1] → A1 → [Capa2] → A2 → L
Backward: dX ← [Capa1] ← dA1 ← [Capa2] ← dA2 ← dL
```

Cada capa calcula `dW`, `db` para actualizar sus propios parámetros, y `dX` para pasarlo a la capa anterior.

**¿Qué resultados esperar?** Con 5000 épocas en XOR, la pérdida debe bajar por debajo de 0.01 y las predicciones deben ser inequívocas: cerca de 0 para entradas iguales y cerca de 1 para entradas distintas.

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

**¿Qué hacemos?** Medimos y graficamos la magnitud promedio de los gradientes en cada capa de la red para diagnosticar el estado del proceso de aprendizaje.

**¿Por qué lo hacemos?** La magnitud del gradiente es un **indicador de salud** del entrenamiento. Nos dice cuánto está "aprendiendo" cada capa:

| Magnitud del gradiente | Diagnóstico | Causa probable |
|------------------------|-------------|----------------|
| `~1e-1` a `~1e-3` | ✅ Saludable | Aprendizaje activo en todas las capas |
| `< 1e-7` (capas profundas) | ⚠️ Vanishing gradients | Activaciones saturadas (sigmoid/tanh), red muy profunda |
| `> 10` | ⚠️ Exploding gradients | Learning rate muy alto, pesos mal inicializados |
| `NaN` o `Inf` | ❌ Colapso numérico | Overflow, división por cero, log de negativo |

**Patrón esperado en redes saludables:** Los gradientes deben ser **similares en magnitud** en todas las capas. Si la capa 1 tiene gradientes 1000 veces más pequeños que la capa 2, la red solo está aprendiendo en las capas cercanas a la salida, y las capas profundas están prácticamente congeladas.

**¿Cómo lo hacemos?** Ejecutamos un forward+backward pass y luego inspeccionamos las magnitudes promedio de `dW` en cada capa usando `np.mean(np.abs(dW))`.

**¿Qué resultados esperar?** En una red bien inicializada con ReLU entrenando XOR, ambas capas deben mostrar gradientes no nulos de magnitud comparable, y estos deben decrecer suavemente a medida que la red converge.

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

**¿Qué hacemos?** Verificamos matemáticamente que nuestra implementación analítica de backpropagation es correcta, comparándola con gradientes calculados numéricamente.

**¿Por qué lo hacemos?** Los bugs en backpropagation son insidiosos: la red puede seguir entrenando, la pérdida puede incluso bajar, pero los gradientes incorrectos llevan a un aprendizaje subóptimo o a fallos sutiles. El gradient checking es la única forma confiable de garantizar que la implementación es correcta.

**El método de diferencias finitas centradas:**

La derivada en un punto `x` se puede aproximar numéricamente usando la fórmula:

$$f'(x) \approx \frac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon}$$

Esta es más precisa que la diferencia hacia adelante `[f(x+ε) - f(x)] / ε` porque el error de aproximación es `O(ε²)` vs `O(ε)`. Para `ε = 1e-5`, el error es del orden de `1e-10`, mucho más pequeño que las diferencias que observaríamos en un bug real.

**La métrica de diferencia relativa:**

No comparamos la diferencia absoluta `|grad_analítico - grad_numérico|` porque los gradientes pueden tener magnitudes muy diferentes. Usamos:

$$\text{diferencia relativa} = \frac{\|g_{\text{analítico}} - g_{\text{numérico}}\|_2}{\|g_{\text{analítico}}\|_2 + \|g_{\text{numérico}}\|_2}$$

Interpretación:
- `< 1e-7` → ✅ Implementación correcta
- `1e-7` a `1e-5` → ⚠️ Probablemente correcto (puede ser error numérico)
- `> 1e-5` → ❌ Hay un bug en backpropagation

**¿Qué resultados esperar?** Con una implementación correcta, la diferencia relativa debe ser menor a `1e-7`. Si introduces un bug intencional (como olvidar trasponer una matriz), la diferencia subirá a `1e-3` o mayor.

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

**¿Qué hacemos?** Aplicamos una estrategia sistemática para encontrar y corregir bugs en implementaciones de backpropagation.

**¿Por qué lo hacemos?** Backpropagation tiene varios puntos de falla comunes que son difíciles de detectar a simple vista porque el código puede ejecutarse sin errores pero producir gradientes incorrectos. Conocer los bugs más frecuentes acelera enormemente el proceso de depuración.

**Los bugs más comunes en backpropagation:**

1. **Transpuesta incorrecta:** En capas densas, `dW = X.T @ dZ` (no `X @ dZ`). Un fallo de dimensiones a veces se "resuelve" transponiéndola en el lugar equivocado.

2. **No dividir por batch size:** Los gradientes deben promediar sobre el batch: `dW = (1/m) * X.T @ dZ`. Sin este factor, el learning rate efectivo escala con el tamaño del batch.

3. **No sumar sobre el batch en bias:** `db = (1/m) * np.sum(dZ, axis=0)`. Olvidar el `sum` produce dimensiones incorrectas o gradientes escalonados.

4. **Confundir `*` y `@`:** `*` es elemento-a-elemento (Hadamard), `@` es multiplicación matricial. Intercambiarlos produce resultados con dimensiones incorrectas o incorrectos silenciosamente.

5. **No guardar el caché:** Si en el forward pass no guardas `x` o `z`, no puedes calcular los gradientes correctos en el backward pass.

6. **Olvidar el gradiente de la activación:** Para una capa con ReLU, el backward pass es `dZ = dA * relu_prime(Z)`, no simplemente `dZ = dA`.

**Estrategia de debugging recomendada:**
1. Verifica las **formas (shapes)** de todos los tensores en forward y backward
2. Comprueba que no hay `NaN` ni `Inf` en ningún punto
3. Ejecuta gradient checking con un batch pequeño (4-8 ejemplos)
4. Si falla, aísla la capa problemática verificando capa por capa
5. Usa prints de la magnitud media (`np.mean(np.abs(tensor))`) para detectar valores anómalos

**¿Qué resultados esperar?** La herramienta de debugging debe mostrar formas consistentes, valores sin NaN/Inf, y magnitudes de gradiente en rango razonable (ni cercanas a 0 ni superiores a 10 en las primeras iteraciones).

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

**¿Qué hacemos?** Comparamos el comportamiento de una red con pesos aleatorios (sin entrenar) contra la misma red después del proceso de backpropagation + descenso por gradiente.

**¿Por qué lo hacemos?** Esta comparación cuantifica directamente el **valor del aprendizaje**: transforma una función aleatoria e inútil en una función que modela correctamente el patrón en los datos. Las métricas que observamos son:

- **Pérdida inicial vs final**: Mide cuánto mejoró la función de predicción. Una reducción del 95%+ indica aprendizaje exitoso.
- **Predicciones antes/después**: Confirma que la red pasó de respuestas aleatorias a respuestas correctas.
- **Curva de aprendizaje**: Revela la dinámica del entrenamiento — ¿baja suavemente? ¿tiene mesetas? ¿oscila?

**Interpretación de la curva de aprendizaje:**

| Forma de la curva | Diagnóstico |
|-------------------|-------------|
| Descenso suave y estable | ✅ Learning rate adecuado |
| Descenso en "escalones" (plateaus) | Posible mínimo local o learning rate muy pequeño |
| Oscilaciones grandes | Learning rate demasiado alto |
| Descenso rápido inicial, luego estancamiento | Red convergiendo a un mínimo, puede necesitar más capacidad |
| Pérdida constante (no baja) | Bug en backpropagation o arquitectura insuficiente |

**¿Qué resultados esperar?** Para XOR con la arquitectura 2→4→1, esperamos:
- Pérdida inicial: ~0.25 (equivalente a predicciones aleatorias para clasificación binaria)
- Pérdida final: < 0.01 después de ~5000 épocas
- Mejora porcentual: > 95%

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
