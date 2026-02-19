# Guía de Laboratorio: Funciones de Activación

## 📋 Información del Laboratorio

**Título:** Funciones de Activación en Redes Neuronales  
**Código:** Lab 03  
**Duración:** 2-3 horas  
**Nivel:** Básico-Intermedio  

---

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender el rol fundamental de las funciones de activación en redes neuronales
2. Implementar ReLU, Sigmoid, Tanh, Leaky ReLU y Softmax desde cero usando NumPy
3. Calcular las derivadas de cada función de activación (necesarias para backpropagation)
4. Visualizar y comparar el comportamiento de diferentes activaciones
5. Elegir la función de activación apropiada para cada tipo de problema y capa
6. Reconocer y demostrar el problema del gradiente que desaparece (vanishing gradient)
7. Identificar y diagnosticar el problema de neuronas muertas en ReLU
8. Integrar funciones de activación en la arquitectura de red del Lab 02
9. Entender por qué la no-linealidad es esencial para el aprendizaje profundo

---

## 📚 Prerrequisitos

### Conocimientos

- **Labs 01 y 02 completados**: neuronas, forward propagation, clases `CapaDensa` y `RedNeuronal`
- Python intermedio: clases, lambdas, funciones de orden superior
- Álgebra lineal básica: vectores, matrices, broadcasting
- Cálculo diferencial básico: derivadas de funciones elementales (chain rule)

### Software

- Python 3.8+
- NumPy 1.19+
- Matplotlib 3.0+
- Jupyter Notebook (recomendado)

### Material de Lectura

Antes de comenzar este laboratorio:
- `teoria.md` — Marco teórico completo sobre funciones de activación
- `README.md` — Visión general del laboratorio
- Repasa la Parte 4 del Lab 02 (limitaciones sin activación no lineal)

---

## 📖 Introducción

En el Lab 02 demostramos matemáticamente que una red neuronal sin funciones de activación no lineal es equivalente a una sola transformación lineal, sin importar cuántas capas tenga. Esta es la limitación fundamental que las **funciones de activación** vienen a resolver.

### Contexto del Problema

¿Por qué necesitamos no-linealidad? Considera estos problemas del mundo real:

**Problema del XOR** (no linealmente separable):
```
 (0,0)→0    (1,1)→0
 (0,1)→1    (1,0)→1
```
No existe ninguna línea recta que separe estas clases. Una red lineal falla; una con activaciones no lineales lo resuelve.

**Reconocimiento de dígitos escritos a mano:**
La relación entre 784 píxeles y el dígito 0-9 es altamente no lineal. Ninguna función lineal puede capturar estos patrones complejos.

### Enfoque con Funciones de Activación

Las funciones de activación se aplican elemento a elemento después de la transformación lineal de cada capa:

```
                 Capa 1                          Capa 2
X  →  [z = X·W₁ + b₁]  →  [a = f(z)]  →  [z = a·W₂ + b₂]  →  [a = g(z)]  →  Y
         transformación         ACTIVACIÓN      transformación         ACTIVACIÓN
         lineal                 NO LINEAL       lineal                 NO LINEAL
```

Las funciones de activación más importantes:

| Función | Ecuación | Rango | Uso típico |
|---------|----------|-------|------------|
| **ReLU** | $\max(0, x)$ | $[0, +\infty)$ | Capas ocultas (estándar) |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | Salida binaria |
| **Tanh** | $\tanh(x)$ | $(-1, 1)$ | RNNs, capas ocultas |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | $(0,1)$, suma=1 | Salida multiclase |
| **Leaky ReLU** | $\max(\alpha x, x)$ | $(-\infty, +\infty)$ | Alternativa a ReLU |

### Conceptos Fundamentales

**1. No-linealidad como capacidad representacional:**

Con activaciones no lineales, la red puede aproximar cualquier función continua (Teorema de Aproximación Universal). Sin ellas, solo puede aprender funciones lineales.

**2. Derivadas — por qué importan:**

Durante el entrenamiento, el algoritmo de backpropagation necesita calcular la derivada de la función de pérdida respecto a cada parámetro. Esto requiere las derivadas de las activaciones:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

donde $\frac{\partial a^{(l)}}{\partial z^{(l)}} = f'(z^{(l)})$ es la derivada de la activación.

**3. El problema del gradiente que desaparece:**

Sigmoid y Tanh "saturan" para valores extremos de $x$: su derivada se acerca a 0. Al multiplicar muchos gradientes pequeños durante backpropagation (una por cada capa), el gradiente se hace exponencialmente pequeño. Las capas iniciales no aprenden.

### Aplicaciones Prácticas

La elección correcta de activación determina el éxito del entrenamiento:
- Clasificar correos como spam/no-spam → Sigmoid en la salida
- Clasificar imágenes en 1000 categorías → Softmax en la salida
- Red profunda para reconocimiento de voz → ReLU en capas ocultas
- Red recurrente LSTM para texto → Tanh en las puertas internas

### Motivación Histórica

Las primeras redes (años 80-90) usaban Sigmoid y Tanh. El problema del gradiente desvaneciente fue identificado por Hochreiter (1991). Décadas después, Nair y Hinton (2010) propusieron ReLU como alternativa eficiente, lo que desbloqueó el entrenamiento de redes muy profundas y desencadenó el "renacimiento" del deep learning moderno.

---

## 🔬 Parte 1: Implementación de Funciones de Activación (40 min)

### 1.1 Introducción Conceptual: Forward y Backward

**¿Qué hacemos?** Implementar cada función de activación junto con su derivada.

**¿Por qué lo hacemos?** Para el forward pass necesitamos la función $f(x)$. Para el backward pass (backpropagation) necesitamos su derivada $f'(x)$. En este lab implementamos ambas para estar preparados para Lab 05 (backpropagation).

**¿Cómo lo hacemos?** Cada función opera elemento a elemento sobre arrays NumPy, aprovechando broadcasting.

**¿Qué resultados esperar?** Funciones que toman un array de cualquier shape y devuelven otro array del mismo shape con los valores transformados.

### 1.2 ReLU (Rectified Linear Unit)

ReLU es la función de activación más usada en la actualidad. Su simplicidad la hace computacionalmente eficiente y su derivada no-cero en la región positiva evita el problema del gradiente desvaneciente.

**Intuición:** ReLU "apaga" las neuronas que reciben señal negativa y deja pasar sin cambios las que reciben señal positiva. Esto crea sparsity: en promedio, la mitad de las neuronas están activas en cada forward pass.

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{si } x > 0 \\ 0 & \text{si } x \leq 0 \end{cases}$$

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{si } x > 0 \\ 0 & \text{si } x \leq 0 \end{cases}$$

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    """
    Rectified Linear Unit (ReLU).
    
    Aplica max(0, x) elemento a elemento.
    Ventajas: simple, eficiente, evita gradiente desvaneciente.
    Desventajas: neuronas muertas para x <= 0.
    
    Args:
        x: Array NumPy de cualquier shape
    Returns:
        Array del mismo shape con max(0, x)
    """
    return np.maximum(0, x)


def relu_derivada(x):
    """
    Derivada de ReLU.
    
    1 donde x > 0, 0 donde x <= 0.
    Nota: técnicamente no diferenciable en x=0, pero en práctica
    se usa 0 o 1 en ese punto sin consecuencias.
    
    Args:
        x: Array NumPy (valores ANTES de aplicar ReLU)
    Returns:
        Array del mismo shape con la derivada
    """
    return (x > 0).astype(float)


# Prueba y verificación
print("=" * 50)
print("PRUEBA DE ReLU")
print("=" * 50)

x_prueba = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

print(f"\nEntrada:         {x_prueba}")
print(f"ReLU(x):         {relu(x_prueba)}")
print(f"ReLU'(x):        {relu_derivada(x_prueba)}")

# Verificación matemática
print(f"\nVerificaciones:")
print(f"  ReLU(-5) = 0:  {relu(np.array([-5.0]))[0] == 0}")
print(f"  ReLU(5)  = 5:  {relu(np.array([5.0]))[0] == 5.0}")
print(f"  ReLU'(-5) = 0: {relu_derivada(np.array([-5.0]))[0] == 0}")
print(f"  ReLU'(5) = 1:  {relu_derivada(np.array([5.0]))[0] == 1.0}")

# ReLU con arrays multidimensionales
X = np.random.randn(4, 3)
print(f"\nArray (4,3):\n{X}")
print(f"\nReLU(array):\n{relu(X)}")
print(f"  Porcentaje activado: {(relu(X) > 0).mean():.1%}")
```

**Actividad 1.1**: Verifica que `relu(-100)` = 0 y `relu(100)` = 100. ¿Por qué ReLU es tan eficiente computacionalmente comparada con Sigmoid?

**Actividad 1.2**: Calcula el porcentaje de neuronas activas (salida > 0) cuando la entrada sigue una distribución normal. ¿Qué porcentaje esperas teóricamente?

### 1.3 Sigmoid (Sigmoide)

Sigmoid fue históricamente la primera función de activación ampliamente usada. Su salida en el rango (0, 1) la hace ideal para modelar probabilidades.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

**Intuición:** Sigmoid "comprime" cualquier valor real al rango (0, 1). Para $x$ muy negativo → 0, para $x$ muy positivo → 1, para $x=0$ → 0.5. El problema: para valores extremos, la derivada $\sigma'(x) \approx 0$ (saturación).

```python
def sigmoid(x):
    """
    Función Sigmoide.
    
    Comprime valores al rango (0, 1).
    Ideal para probabilidades en clasificación binaria.
    
    NOTA: Para valores muy negativos puede generar overflow.
    Se usa la versión numéricamente estable.
    
    Args:
        x: Array NumPy de cualquier shape
    Returns:
        Array del mismo shape con valores en (0, 1)
    """
    # Versión numéricamente estable para evitar overflow
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def sigmoid_derivada(x):
    """
    Derivada de Sigmoid.
    
    σ'(x) = σ(x) * (1 - σ(x))
    
    ⚠️ SATURACIÓN: Para |x| > 5, la derivada es ≈ 0
    Esto causa el problema del gradiente desvaneciente.
    
    Args:
        x: Array NumPy (valores ANTES de aplicar sigmoid)
    Returns:
        Array del mismo shape con la derivada
    """
    s = sigmoid(x)
    return s * (1 - s)


# Prueba y verificación
print("=" * 50)
print("PRUEBA DE SIGMOID")
print("=" * 50)

x_prueba = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])

print(f"\nEntrada:           {x_prueba}")
print(f"Sigmoid(x):        {sigmoid(x_prueba).round(4)}")
print(f"Sigmoid'(x):       {sigmoid_derivada(x_prueba).round(6)}")

print(f"\nPropiedades:")
print(f"  σ(0) = 0.5:    {abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10}")
print(f"  Rango en (-∞):  {sigmoid(np.array([-100.0]))[0]:.10f}")
print(f"  Rango en (+∞):  {sigmoid(np.array([100.0]))[0]:.10f}")

print(f"\n⚠️ Saturación del gradiente:")
for val in [-10, -5, 0, 5, 10]:
    grad = sigmoid_derivada(np.array([float(val)]))[0]
    print(f"  σ'({val:3d}) = {grad:.8f}")
```

**Actividad 1.3**: Verifica numéricamente que la derivada `sigmoid_derivada(x)` es correcta comparándola con una aproximación numérica:
```
f'(x) ≈ (f(x+h) - f(x-h)) / (2h)  con h = 1e-5
```

**Actividad 1.4**: Grafica `sigmoid(x)` y `sigmoid_derivada(x)` para $x \in [-6, 6]$. ¿En qué rango la derivada es significativa?

### 1.4 Tanh (Tangente Hiperbólica)

Tanh es similar a Sigmoid pero con salidas en $(-1, 1)$ y centrada en cero. Esto facilita el aprendizaje porque las activaciones positivas y negativas se balancean.

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

$$\tanh'(x) = 1 - \tanh^2(x)$$

```python
def tanh(x):
    """
    Tangente Hiperbólica (Tanh).
    
    Similar a Sigmoid pero con rango (-1, 1) y centrada en 0.
    Convergencia más rápida que Sigmoid en la práctica.
    Aún sufre de saturación para |x| grande.
    
    Args:
        x: Array NumPy de cualquier shape
    Returns:
        Array del mismo shape con valores en (-1, 1)
    """
    return np.tanh(x)  # NumPy tiene tanh optimizado


def tanh_derivada(x):
    """
    Derivada de Tanh.
    
    tanh'(x) = 1 - tanh²(x)
    
    Rango de la derivada: (0, 1]
    Máximo en x=0: tanh'(0) = 1
    
    Args:
        x: Array NumPy (valores ANTES de aplicar tanh)
    Returns:
        Array del mismo shape con la derivada
    """
    return 1 - np.tanh(x) ** 2


# Prueba
print("=" * 50)
print("PRUEBA DE TANH")
print("=" * 50)

x_prueba = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
print(f"\nEntrada:     {x_prueba}")
print(f"Tanh(x):     {tanh(x_prueba).round(4)}")
print(f"Tanh'(x):    {tanh_derivada(x_prueba).round(6)}")

print(f"\nPropiedades:")
print(f"  Antisimétrica: tanh(-x) = -tanh(x)")
t1 = tanh(np.array([2.0]))[0]
t2 = tanh(np.array([-2.0]))[0]
print(f"  tanh(2) = {t1:.4f}, tanh(-2) = {t2:.4f}, suma = {t1+t2:.10f}")

print(f"\nComparación Sigmoid vs Tanh (derivadas):")
print(f"{'x':>6} | {'σ(x)':<10} | {'σ\'(x)':<12} | {'tanh(x)':<10} | {'tanh\'(x)'}")
print("-" * 60)
for v in [-3, -1, 0, 1, 3]:
    sv = sigmoid(np.array([float(v)]))[0]
    sdv = sigmoid_derivada(np.array([float(v)]))[0]
    tv = tanh(np.array([float(v)]))[0]
    tdv = tanh_derivada(np.array([float(v)]))[0]
    print(f"{v:>6} | {sv:<10.4f} | {sdv:<12.6f} | {tv:<10.4f} | {tdv:.6f}")
```

**Actividad 1.5**: ¿Por qué Tanh converge más rápido que Sigmoid en la práctica? Pista: relaciona con las salidas centradas en cero.

### 1.5 Softmax

Softmax es especial: opera sobre un vector completo (no elemento a elemento) y produce una distribución de probabilidad válida (todos los valores en $(0,1)$ y suma = 1).

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Truco de estabilidad numérica:** Restar el máximo antes de exponenciar evita overflow sin cambiar el resultado:

$$\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

```python
def softmax(x):
    """
    Función Softmax.
    
    Convierte un vector de scores en una distribución de probabilidad.
    La suma de todas las salidas es exactamente 1.
    
    Usa estabilización numérica restando el máximo para evitar overflow.
    
    Args:
        x: Array (batch_size, n_clases) o (n_clases,)
    Returns:
        Array del mismo shape con probabilidades que suman 1
    """
    # Estabilización numérica: restar el máximo no cambia el resultado
    # pero previene overflow con valores grandes
    x_stable = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Prueba
print("=" * 55)
print("PRUEBA DE SOFTMAX")
print("=" * 55)

# Caso 1: Un vector simple
z = np.array([[1.0, 2.0, 3.0, 4.0]])  # Un batch, 4 clases
probs = softmax(z)
print(f"\nScores:             {z[0]}")
print(f"Probabilidades:     {probs[0].round(4)}")
print(f"Suma:               {probs.sum():.10f}")
print(f"Clase predicha:     {np.argmax(probs)}")

# Caso 2: Batch de 3 muestras con 5 clases
batch_z = np.random.randn(3, 5)
batch_probs = softmax(batch_z)
print(f"\nBatch scores:\n{batch_z.round(3)}")
print(f"\nBatch probabilidades:\n{batch_probs.round(4)}")
print(f"Sumas por muestra: {batch_probs.sum(axis=1)}")

# Caso 3: Efecto de temperatura
print("\n🌡️  Efecto de temperatura en Softmax:")
z_temp = np.array([[1.0, 2.0, 5.0]])
for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    p = softmax(z_temp / T)
    print(f"  T={T:4.1f}: {p[0].round(3)}")
print("  → T pequeño: más 'confiado'; T grande: más uniforme")
```

**Actividad 1.6**: Demuestra que restar el máximo antes de exponenciar no cambia el resultado del softmax. Verifica con un ejemplo numérico concreto con y sin la estabilización.

**Actividad 1.7**: Implementa `softmax` sin estabilización numérica y demuestra cuándo falla (overflow) con valores grandes como `np.array([[1000., 2000., 3000.]])`.

### 1.6 Leaky ReLU

Leaky ReLU soluciona el problema de las "neuronas muertas" de ReLU: en lugar de hacer 0 en la región negativa, permite un gradiente pequeño $\alpha$.

$$\text{LeakyReLU}(x) = \max(\alpha x, x) = \begin{cases} x & \text{si } x > 0 \\ \alpha x & \text{si } x \leq 0 \end{cases}$$

```python
def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU.
    
    Soluciona el problema de neuronas muertas de ReLU
    permitiendo un pequeño gradiente en la región negativa.
    
    Args:
        x: Array NumPy de cualquier shape
        alpha: Pendiente en la región negativa (default=0.01)
    Returns:
        Array del mismo shape
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivada(x, alpha=0.01):
    """Derivada de Leaky ReLU."""
    return np.where(x > 0, 1.0, alpha)


# Prueba
x_prueba = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print("Leaky ReLU (alpha=0.01):")
print(f"  Entrada: {x_prueba}")
print(f"  Salida:  {leaky_relu(x_prueba).round(4)}")
print(f"  Derivada:{leaky_relu_derivada(x_prueba)}")
```

**Actividad 1.8**: Implementa ELU (Exponential Linear Unit):
$$\text{ELU}(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha(e^x - 1) & \text{si } x \leq 0 \end{cases}$$

### Actividades de Verificación

**Actividad 1.9**: Crea una función `verificar_derivada(func, func_deriv, x, h=1e-5)` que compare la derivada analítica con la aproximación numérica usando diferencias finitas. Verifica todas las funciones implementadas.

**Actividad 1.10**: Implementa todas las funciones de activación y sus derivadas en un diccionario para acceso fácil:
```python
ACTIVACIONES = {
    'relu': (relu, relu_derivada),
    'sigmoid': (sigmoid, sigmoid_derivada),
    'tanh': (tanh, tanh_derivada),
    'leaky_relu': (leaky_relu, leaky_relu_derivada),
}
```

### Preguntas de Reflexión

**Pregunta 1.1 (Concebir):** ¿Por qué no usamos funciones de activación polinomiales (e.g., $f(x) = x^2$) a pesar de ser no lineales?

**Pregunta 1.2 (Diseñar):** ¿Qué función de activación usarías para la capa oculta de una red que debe predecir una probabilidad de lluvia? ¿Y para la capa de salida?

**Pregunta 1.3 (Implementar):** ¿Por qué el "truco de estabilidad numérica" en Softmax (restar el máximo) produce exactamente el mismo resultado matemático?

**Pregunta 1.4 (Operar):** Si observas que en producción el modelo predice siempre la misma clase (probabilidad muy alta para una clase, muy baja para otras), ¿qué podría estar pasando con la temperatura del Softmax?

---

## 🔬 Parte 2: Integración con la Arquitectura de Red (40 min)

### 2.1 Introducción Conceptual: Capas de Activación

**¿Qué hacemos?** Integrar las funciones de activación en la arquitectura modular del Lab 02.

**¿Por qué lo hacemos?** Las funciones de activación son operaciones separadas en el grafo computacional. Separarlas en clases propias facilita:
- Agregar cualquier activación a cualquier capa sin modificar `CapaDensa`
- Implementar backpropagation de forma modular
- Experimentar con diferentes combinaciones de capas y activaciones

**Analogía:** En electrónica, los componentes (resistores, capacitores) son separados y se conectan según el diseño del circuito. Del mismo modo, `CapaDensa` y `CapaActivacion` son componentes que se combinan libremente.

**¿Qué resultados esperar?** Una clase `CapaActivacion` que aplica cualquier función de activación y puede calcular gradientes para backpropagation.

### 2.2 Clase CapaActivacion

```python
class CapaActivacion:
    """
    Capa de activación independiente.
    
    Aplica una función de activación elemento a elemento.
    Guarda la entrada para calcular gradientes en backpropagation.
    
    Args:
        funcion: Función de activación f(x)
        derivada: Derivada de la función f'(x)
        nombre: Nombre descriptivo (para display)
    """
    
    def __init__(self, funcion, derivada, nombre="activacion"):
        self.funcion = funcion
        self.derivada = derivada
        self.nombre = nombre
        self.entradas = None
        self.salida = None
    
    def forward(self, entradas):
        """
        Aplica la función de activación.
        
        Guarda la entrada para usar en backward pass.
        
        Args:
            entradas: Array (batch_size, n_neuronas) — salida de CapaDensa
        Returns:
            salida: Array del mismo shape con activación aplicada
        """
        self.entradas = entradas.copy()
        self.salida = self.funcion(entradas)
        return self.salida
    
    def backward(self, grad_salida):
        """
        Backpropagation a través de la activación.
        
        Multiplica el gradiente entrante por la derivada local.
        (Regla de la cadena)
        
        Args:
            grad_salida: Gradiente de la pérdida respecto a la salida
        Returns:
            grad_entrada: Gradiente respecto a la entrada de esta capa
        """
        return grad_salida * self.derivada(self.entradas)
    
    def contar_parametros(self):
        """Las capas de activación no tienen parámetros aprendibles."""
        return 0
    
    def __repr__(self):
        return f"CapaActivacion({self.nombre})"


# Instancias predefinidas
ActivacionReLU = lambda: CapaActivacion(relu, relu_derivada, "ReLU")
ActivacionSigmoid = lambda: CapaActivacion(sigmoid, sigmoid_derivada, "Sigmoid")
ActivacionTanh = lambda: CapaActivacion(tanh, tanh_derivada, "Tanh")
ActivacionLeakyReLU = lambda: CapaActivacion(leaky_relu, leaky_relu_derivada, "LeakyReLU")


# Ejemplo de uso
print("=" * 50)
print("PRUEBA DE CapaActivacion")
print("=" * 50)

capa_relu = ActivacionReLU()
X = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

salida = capa_relu.forward(X)
print(f"\nEntrada:  {X[0]}")
print(f"ReLU:     {salida[0]}")

# Simular un gradiente del backward pass
grad = np.ones_like(salida)  # Gradiente = 1 para todos
grad_entrada = capa_relu.backward(grad)
print(f"Gradiente entrada (backprop): {grad_entrada[0]}")
```

### 2.3 Red Neuronal con Activaciones

```python
class RedNeuronalConActivaciones:
    """
    Red neuronal que intercala capas densas y capas de activación.
    
    Permite construir redes con cualquier combinación de
    capas y funciones de activación.
    
    Ejemplo:
        red = RedNeuronalConActivaciones(
            arquitectura=[10, 20, 15, 1],
            activaciones=['relu', 'relu', 'sigmoid']
        )
    """
    
    ACTIVACIONES_DISPONIBLES = {
        'relu':       (relu, relu_derivada),
        'sigmoid':    (sigmoid, sigmoid_derivada),
        'tanh':       (tanh, tanh_derivada),
        'leaky_relu': (leaky_relu, leaky_relu_derivada),
        'lineal':     (lambda x: x, lambda x: np.ones_like(x)),
    }
    
    def __init__(self, arquitectura, activaciones, seed=None):
        """
        Args:
            arquitectura: Lista de neuronas [n_in, n1, n2, ..., n_out]
            activaciones: Lista de nombres de activaciones, una por capa
                         len(activaciones) == len(arquitectura) - 1
        """
        assert len(activaciones) == len(arquitectura) - 1, \
            "Necesitas una activación por capa densa"
        
        self.capas = []
        
        for i in range(len(arquitectura) - 1):
            n_in = arquitectura[i]
            n_out = arquitectura[i + 1]
            nombre_act = activaciones[i]
            
            # Capa densa
            self.capas.append(CapaDensa(n_in, n_out, seed=seed))
            
            # Capa de activación
            func, deriv = self.ACTIVACIONES_DISPONIBLES[nombre_act]
            self.capas.append(CapaActivacion(func, deriv, nombre_act))
        
        self.arquitectura = arquitectura
        self.activaciones = activaciones
    
    def forward(self, X):
        """Forward pass a través de todas las capas."""
        activacion = X
        for capa in self.capas:
            activacion = capa.forward(activacion)
        return activacion
    
    def resumen(self):
        """Imprime la arquitectura con activaciones."""
        print("\n" + "=" * 65)
        print("ARQUITECTURA DE LA RED")
        print("=" * 65)
        total_params = 0
        for i, capa in enumerate(self.capas):
            params = capa.contar_parametros()
            total_params += params
            print(f"  Capa {i+1}: {capa!r:<40} | {params:>10,} parámetros")
        print("-" * 65)
        print(f"  TOTAL:                                       {total_params:>10,} parámetros")
        print("=" * 65)
    
    def contar_parametros(self):
        return sum(c.contar_parametros() for c in self.capas)


# Ejemplos de uso para diferentes problemas
print("=" * 60)
print("REDES PARA DIFERENTES PROBLEMAS")
print("=" * 60)

# Clasificación binaria (spam)
print("\n1. Clasificación Binaria (spam/no-spam):")
red_binaria = RedNeuronalConActivaciones(
    arquitectura=[100, 64, 32, 1],
    activaciones=['relu', 'relu', 'sigmoid']
)
red_binaria.resumen()

X_spam = np.random.randn(16, 100)
pred = red_binaria.forward(X_spam)
print(f"   Entrada: {X_spam.shape} → Salida: {pred.shape}")
print(f"   Rango de salida: [{pred.min():.4f}, {pred.max():.4f}] (esperado: [0,1])")

# Clasificación multiclase (MNIST)
print("\n2. Clasificación Multiclase (MNIST):")
red_multiclase = RedNeuronalConActivaciones(
    arquitectura=[784, 256, 128, 10],
    activaciones=['relu', 'relu', 'lineal']  # Softmax se aplica aparte
)
X_mnist = np.random.randn(32, 784)
logits = red_multiclase.forward(X_mnist)
probs = softmax(logits)
print(f"   Logits: {logits.shape} → Probs: {probs.shape}")
print(f"   Sumas de probabilidades: {probs.sum(axis=1)[:3].round(4)} (todas deben ser 1)")

# Regresión
print("\n3. Regresión (predicción de precios):")
red_regresion = RedNeuronalConActivaciones(
    arquitectura=[20, 64, 32, 1],
    activaciones=['relu', 'relu', 'lineal']  # Sin activación en salida
)
X_reg = np.random.randn(8, 20)
pred_reg = red_regresion.forward(X_reg)
print(f"   Salida: {pred_reg.shape}, sin restricción de rango")
```

**Actividad 2.1**: Crea una red `[10, 20, 15, 5]` usando todas las combinaciones de activaciones: (relu, relu, relu), (tanh, tanh, sigmoid), (leaky_relu, relu, lineal). Compara las distribuciones de salida.

**Actividad 2.2**: Implementa el método `analizar_activaciones(X)` en `RedNeuronalConActivaciones` que analice las estadísticas de cada capa densa y de activación por separado.

**Actividad 2.3**: Verifica que `red.backward()` funciona correctamente para la capa de activación: el gradiente multiplicado por la derivada debe ser correcto.

### Preguntas de Reflexión

**Pregunta 2.1 (Concebir):** ¿Por qué Softmax no se incluye como capa de activación en la red sino que se aplica después de los logits?

**Pregunta 2.2 (Diseñar):** ¿Qué ventajas tiene separar `CapaDensa` de `CapaActivacion` frente a fusionarlas en una sola clase?

**Pregunta 2.3 (Implementar):** En el método `backward()` de `CapaActivacion`, ¿por qué multiplicamos `grad_salida * derivada(entradas)` (regla de la cadena)?

**Pregunta 2.4 (Operar):** Si al inferir en producción obtienes probabilidades de Softmax siempre muy uniformes (~0.1 para 10 clases), ¿qué podría indicar esto sobre el estado del modelo?

---

## 🔬 Parte 3: Visualización y Análisis Comparativo (35 min)

### 3.1 Introducción Conceptual: Visualizar para Comprender

**¿Qué hacemos?** Graficar las funciones de activación y sus derivadas, y comparar su comportamiento.

**¿Por qué lo hacemos?** Las gráficas revelan intuitivamente:
- En qué rangos satura cada función (derivada ≈ 0)
- Cómo se distribuyen los gradientes
- Por qué ReLU mitiga el gradiente desvaneciente
- El efecto de cada función sobre la distribución de activaciones

**¿Qué resultados esperar?** Gráficas claras que muestren las curvas de cada función y sus derivadas, con anotaciones que expliquen el comportamiento en regiones clave.

### 3.2 Comparación Visual de Funciones y Derivadas

```python
import matplotlib.pyplot as plt

def graficar_funciones_activacion():
    """
    Crea una visualización completa de todas las funciones
    de activación y sus derivadas.
    """
    x = np.linspace(-5, 5, 300)
    
    funciones = [
        ('ReLU',        relu(x),          relu_derivada(x),         'steelblue'),
        ('Sigmoid',     sigmoid(x),        sigmoid_derivada(x),      'darkorange'),
        ('Tanh',        tanh(x),           tanh_derivada(x),         'green'),
        ('Leaky ReLU',  leaky_relu(x),     leaky_relu_derivada(x),   'red'),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Funciones de Activación y sus Derivadas', 
                 fontsize=15, fontweight='bold', y=1.01)
    
    for i, (nombre, f_x, df_x, color) in enumerate(funciones):
        # Función
        ax = axes[0, i]
        ax.plot(x, f_x, color=color, linewidth=2.5)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(f'{nombre}', fontsize=12, fontweight='bold', color=color)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
        
        # Derivada
        ax = axes[1, i]
        ax.plot(x, df_x, color=color, linewidth=2.5, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(f"Derivada de {nombre}", fontsize=11, color=color)
        ax.set_xlabel('x')
        ax.set_ylabel("f'(x)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
    
    plt.tight_layout()
    plt.savefig('activaciones_comparacion.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico guardado como 'activaciones_comparacion.png'")

graficar_funciones_activacion()
```

### 3.3 El Problema del Gradiente que Desaparece

```python
def demostrar_gradiente_desaparece(n_capas=15):
    """
    Demuestra cómo el gradiente se desvanece al propagarse hacia atrás
    en redes profundas con activaciones saturantes.
    
    Args:
        n_capas: Número de capas a simular
    """
    print("=" * 65)
    print("DEMOSTRACIÓN: GRADIENTE QUE DESAPARECE")
    print("=" * 65)
    
    # Simular backpropagation con diferentes activaciones
    # Si el gradiente en cada capa es d, después de n capas: gradiente ≈ d^n
    
    gradientes_sigmoid = []
    gradientes_tanh = []
    gradientes_relu = []
    
    # Punto de operación: x=0 (donde los gradientes son más favorables)
    x = np.array([0.0])
    
    grad_sigmoid = sigmoid_derivada(x)[0]   # ≈ 0.25 (máximo)
    grad_tanh = tanh_derivada(x)[0]         # = 1.0 (máximo)
    grad_relu = relu_derivada(np.array([1.0]))[0]  # = 1.0
    
    print(f"\nGradiente por capa (en punto óptimo x=0):")
    print(f"  Sigmoid:  σ'(0) = {grad_sigmoid:.4f}")
    print(f"  Tanh:    tanh'(0) = {grad_tanh:.4f}")
    print(f"  ReLU:   ReLU'(1) = {grad_relu:.4f}")
    
    print(f"\n{'Capa':<6} | {'Sigmoid':<15} | {'Tanh':<15} | {'ReLU':<15}")
    print("-" * 55)
    
    g_sig = 1.0
    g_tanh = 1.0
    g_relu = 1.0
    
    for capa in range(1, n_capas + 1):
        g_sig  *= grad_sigmoid
        g_tanh *= grad_tanh
        g_relu *= grad_relu
        
        gradientes_sigmoid.append(g_sig)
        gradientes_tanh.append(g_tanh)
        gradientes_relu.append(g_relu)
        
        if capa <= 10 or capa == n_capas:
            print(f"{capa:<6} | {g_sig:<15.2e} | {g_tanh:<15.2e} | {g_relu:<15.2e}")
    
    print(f"\n⚠️  CONCLUSIÓN:")
    print(f"  Sigmoid tras {n_capas} capas: gradiente = {g_sig:.2e}")
    print(f"  → ¡{n_capas} órdenes de magnitud más pequeño!")
    print(f"  → Las primeras capas CASI NO APRENDEN")
    print(f"\n  ReLU mantiene gradiente = {g_relu:.2e}")
    print(f"  → Todas las capas aprenden a la misma tasa")
    
    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    capas = list(range(1, n_capas + 1))
    ax1.semilogy(capas, gradientes_sigmoid, 'o-', color='darkorange', 
                 linewidth=2, label='Sigmoid')
    ax1.semilogy(capas, gradientes_tanh, 's-', color='green', 
                 linewidth=2, label='Tanh')
    ax1.semilogy(capas, gradientes_relu, '^-', color='steelblue', 
                 linewidth=2, label='ReLU')
    ax1.set_xlabel('Número de capa (desde la salida)')
    ax1.set_ylabel('Magnitud del gradiente (escala log)')
    ax1.set_title('Gradiente que Desaparece', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Comparación de derivadas
    x_range = np.linspace(-4, 4, 200)
    ax2.plot(x_range, sigmoid_derivada(x_range), '--', color='darkorange', 
             linewidth=2, label=f"Sigmoid' (max={sigmoid_derivada(np.array([0.]))[0]:.3f})")
    ax2.plot(x_range, tanh_derivada(x_range), '--', color='green', 
             linewidth=2, label=f"Tanh' (max={tanh_derivada(np.array([0.]))[0]:.3f})")
    ax2.plot(x_range, relu_derivada(x_range), '-', color='steelblue', 
             linewidth=2, label=f"ReLU' (max=1.0)")
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.set_title('Comparación de Derivadas', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig('saturacion_gradientes.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico guardado como 'saturacion_gradientes.png'")

demostrar_gradiente_desaparece()
```

### 3.4 Neuronas Muertas en ReLU

```python
def analizar_neuronas_muertas(red, X, umbral=0.01):
    """
    Detecta y cuantifica neuronas muertas en redes con ReLU.
    
    Una neurona "muerta" es aquella cuya activación es cero
    para TODAS las muestras del dataset. Una vez muerta,
    no puede aprender porque su gradiente es siempre 0.
    
    Args:
        red: RedNeuronalConActivaciones con ReLU
        X: Datos de entrada (batch_size, n_entradas)
        umbral: Porcentaje máximo de activaciones > 0 para considerar "muerta"
    
    Returns:
        dict: Estadísticas de neuronas muertas por capa
    """
    print("=" * 60)
    print("ANÁLISIS DE NEURONAS MUERTAS (ReLU)")
    print("=" * 60)
    
    stats = {}
    activacion = X
    
    for i, capa in enumerate(red.capas):
        activacion = capa.forward(activacion)
        
        if isinstance(capa, CapaActivacion) and capa.nombre == "ReLU":
            # Verificar cuántas neuronas tienen activación 0 en TODAS las muestras
            activa_por_neurona = (activacion > umbral).mean(axis=0)  # Shape: (n_neuronas,)
            muertas = (activa_por_neurona == 0).sum()
            total = activacion.shape[1]
            
            print(f"\n  Capa ReLU #{i+1}:")
            print(f"    Total neuronas:    {total}")
            print(f"    Neuronas muertas:  {muertas} ({muertas/total:.1%})")
            print(f"    Activadas siempre: {(activa_por_neurona == 1).sum()} "
                  f"({(activa_por_neurona == 1).sum()/total:.1%})")
            print(f"    Activación media:  {activa_por_neurona.mean():.1%} de las muestras")
            
            stats[f'relu_{i}'] = {
                'muertas': muertas,
                'total': total,
                'porcentaje': muertas / total
            }
    
    return stats


# Demostración con inicialización que causa muchas neuronas muertas
print("\n--- Con inicialización muy negativa (bias negativo grande) ---")
np.random.seed(42)
red_mala_init = RedNeuronalConActivaciones(
    arquitectura=[20, 50, 30, 5],
    activaciones=['relu', 'relu', 'relu']
)
# Forzar bias negativos grandes para crear neuronas muertas
for capa in red_mala_init.capas:
    if isinstance(capa, CapaDensa):
        capa.biases = np.full_like(capa.biases, -5.0)  # Bias muy negativo

X_test = np.random.randn(500, 20)
stats = analizar_neuronas_muertas(red_mala_init, X_test)

print("\n--- Con inicialización estándar ---")
red_buena_init = RedNeuronalConActivaciones(
    arquitectura=[20, 50, 30, 5],
    activaciones=['relu', 'relu', 'relu'],
    seed=42
)
stats2 = analizar_neuronas_muertas(red_buena_init, X_test)
```

**Actividad 3.1**: Grafica las 4 funciones de activación en un mismo gráfico. ¿Cuál tiene el rango más amplio de valores? ¿Cuál tiene la derivada más simple?

**Actividad 3.2**: Ejecuta `demostrar_gradiente_desaparece()` con 20 capas. ¿Cuántas capas puede soportar Sigmoid antes de que el gradiente sea menor que $10^{-10}$?

**Actividad 3.3**: Experimenta con diferentes bias negativos y mide cuántas neuronas mueren. ¿A partir de qué valor de bias aparecen neuronas muertas?

**Actividad 3.4**: Compara las distribuciones de activaciones de ReLU vs Leaky ReLU en las capas intermedias de una red profunda.

### Preguntas de Reflexión

**Pregunta 3.1 (Concebir):** ¿Por qué el problema del gradiente desvaneciente es más severo en redes muy profundas que en redes poco profundas?

**Pregunta 3.2 (Diseñar):** Si debes usar Sigmoid o Tanh (por requisitos del dominio), ¿qué estrategias complementarias podrías usar para mitigar el vanishing gradient?

**Pregunta 3.3 (Implementar):** ¿Cómo podrías modificar la inicialización de biases para reducir la aparición de neuronas muertas al inicio del entrenamiento?

**Pregunta 3.4 (Operar):** En un sistema en producción, ¿cómo detectarías en tiempo real si tu red está sufriendo de neuronas muertas o gradiente desvaneciente?

---

## 🔬 Parte 4: Casos de Uso y Selección de Activaciones (30 min)

### 4.1 Introducción Conceptual: La Elección Correcta Importa

**¿Qué hacemos?** Estudiar qué función de activación corresponde a cada tipo de problema y posición en la red.

**¿Por qué lo hacemos?** Una elección incorrecta puede hacer que la red no converja, produzca salidas sin sentido (probabilidades > 1), o aprenda muy lentamente.

**Reglas generales:**

| Posición en la red | Problema | Activación recomendada |
|-------------------|----------|----------------------|
| Capas ocultas | Cualquiera | **ReLU** (o Leaky ReLU) |
| Capa de salida | Clasificación binaria | **Sigmoid** |
| Capa de salida | Clasificación multiclase | **Softmax** |
| Capa de salida | Regresión | **Lineal** (sin activación) |
| Capas ocultas | RNNs, LSTMs | **Tanh** |
| Capas ocultas | Generativas (GANs) | **Leaky ReLU**, Tanh |

**¿Qué resultados esperar?** Redes cuyas salidas tienen el rango e interpretación correctos para cada tipo de problema.

### 4.2 Comparación Experimental de Combinaciones

```python
def comparar_configuraciones():
    """
    Compara experimentalmente diferentes combinaciones de activaciones
    y analiza su impacto en las distribuciones de salida.
    """
    print("=" * 65)
    print("COMPARACIÓN EXPERIMENTAL DE CONFIGURACIONES")
    print("=" * 65)
    
    np.random.seed(42)
    X = np.random.randn(200, 10)
    
    configuraciones = [
        {
            'nombre': 'Solo Sigmoid (problemático en capas ocultas)',
            'arq': [10, 20, 15, 5],
            'acts': ['sigmoid', 'sigmoid', 'sigmoid']
        },
        {
            'nombre': 'Solo ReLU (bueno para capas ocultas)',
            'arq': [10, 20, 15, 5],
            'acts': ['relu', 'relu', 'relu']
        },
        {
            'nombre': 'ReLU ocultas + Sigmoid salida (clasificación binaria)',
            'arq': [10, 20, 15, 1],
            'acts': ['relu', 'relu', 'sigmoid']
        },
        {
            'nombre': 'ReLU ocultas + Lineal salida (regresión)',
            'arq': [10, 20, 15, 1],
            'acts': ['relu', 'relu', 'lineal']
        },
        {
            'nombre': 'Tanh ocultas + Lineal salida',
            'arq': [10, 20, 15, 5],
            'acts': ['tanh', 'tanh', 'lineal']
        },
    ]
    
    for config in configuraciones:
        red = RedNeuronalConActivaciones(
            arquitectura=config['arq'],
            activaciones=config['acts'],
            seed=42
        )
        salida = red.forward(X)
        
        print(f"\n📊 {config['nombre']}")
        print(f"   Shape salida: {salida.shape}")
        print(f"   Media: {salida.mean():.4f} | Std: {salida.std():.4f}")
        print(f"   Min:   {salida.min():.4f} | Max: {salida.max():.4f}")
        
        # Para salida con Softmax
        if config['acts'][-1] == 'lineal' and salida.shape[1] == 5:
            probs = softmax(salida)
            print(f"   (Softmax) Suma probs: {probs.sum(axis=1).mean():.6f}")

comparar_configuraciones()
```

### 4.3 Guía de Selección Práctica

```python
def recomendar_activacion(tipo_problema, posicion_capa, info_adicional=None):
    """
    Recomienda la función de activación apropiada.
    
    Args:
        tipo_problema: 'binaria', 'multiclase', 'regresion', 'rnn'
        posicion_capa: 'oculta', 'salida'
        info_adicional: dict con contexto extra
    
    Returns:
        str: Nombre de la activación recomendada y justificación
    """
    reglas = {
        ('salida', 'binaria'):     ('sigmoid',  'Salida en (0,1) → interpretable como probabilidad'),
        ('salida', 'multiclase'):  ('softmax',  'Distribución de prob. que suma 1'),
        ('salida', 'regresion'):   ('lineal',   'Sin restricción de rango para valores continuos'),
        ('oculta', 'general'):     ('relu',     'Eficiente, evita vanishing gradient'),
        ('oculta', 'rnn'):         ('tanh',     'Centrada en 0, gradientes más estables en RNNs'),
        ('oculta', 'profunda'):    ('relu',     'Ideal para redes muy profundas'),
    }
    
    key = (posicion_capa, tipo_problema)
    if key in reglas:
        act, justif = reglas[key]
        print(f"✅ Recomendación: {act.upper()}")
        print(f"   Justificación: {justif}")
        return act
    else:
        print("⚠️  Situación no contemplada. Usa ReLU como punto de partida.")
        return 'relu'


# Ejemplos de uso
print("🤔 ¿Qué activación usar?\n")
print("Caso 1: Capa oculta en clasificación de imágenes")
recomendar_activacion('general', 'oculta')

print("\nCaso 2: Capa de salida para clasificar 10 dígitos")
recomendar_activacion('multiclase', 'salida')

print("\nCaso 3: Capa de salida para predecir temperatura")
recomendar_activacion('regresion', 'salida')

print("\nCaso 4: Capa oculta en red recurrente (RNN)")
recomendar_activacion('rnn', 'oculta')
```

**Actividad 4.1**: Diseña e implementa redes para los siguientes escenarios. Justifica cada elección de activación:
- Detector de fraude (binario): input=30 features, output=probabilidad de fraude
- Clasificador de sentimientos (5 clases): input=1000 features de texto
- Predictor de temperatura (regresión): input=10 variables meteorológicas

**Actividad 4.2**: Implementa un experimento que compare el tiempo de convergencia de una red con Sigmoid vs ReLU en capas ocultas usando gradiente descendente manual.

**Actividad 4.3**: Diseña una red para el problema XOR con activaciones no lineales. ¿Resuelve el problema que una red lineal no podía?

### Preguntas de Reflexión

**Pregunta 4.1 (Concebir):** ¿Por qué usar Sigmoid en capas ocultas de redes profundas (más de 5 capas) es generalmente una mala práctica?

**Pregunta 4.2 (Diseñar):** Si tienes una red para predicción de ratings (1-5 estrellas), ¿qué activación usarías en la capa de salida? ¿Cómo representarías el problema?

**Pregunta 4.3 (Implementar):** ¿Cómo implementarías una función de activación personalizada que solo uses en un problema específico?

**Pregunta 4.4 (Operar):** En un modelo de producción para clasificación multiclase, ¿cuándo usarías las probabilidades del Softmax directamente vs la clase predicha (`argmax`)?

---

## 📊 Análisis Final de Rendimiento

### Benchmark: Velocidad de Funciones de Activación

Las funciones de activación se aplican millones de veces durante el entrenamiento. Su velocidad de ejecución importa significativamente.

**Fundamento:** ReLU es simplemente `np.maximum(0, x)`, una operación elementwise extremadamente rápida. Sigmoid y Tanh requieren cálculo de exponenciales, que es más costoso. Este benchmark te mostrará el impacto práctico.

```python
import time

def benchmark_activaciones(n=10_000_000, repeticiones=5):
    """
    Compara el tiempo de ejecución de cada función de activación.
    
    Args:
        n: Número de elementos en el array
        repeticiones: Número de mediciones para promediar
    """
    print("\n" + "=" * 65)
    print(f"BENCHMARK: VELOCIDAD DE FUNCIONES DE ACTIVACIÓN")
    print(f"Array de {n:,} elementos, {repeticiones} repeticiones")
    print("=" * 65)
    
    x = np.random.randn(n)
    
    funciones_test = {
        'ReLU':        relu,
        'Sigmoid':     sigmoid,
        'Tanh':        tanh,
        'Leaky ReLU':  leaky_relu,
        'Softmax':     lambda x_: softmax(x_.reshape(100, -1)).ravel(),
    }
    
    tiempos = {}
    for nombre, func in funciones_test.items():
        mediciones = []
        for _ in range(repeticiones):
            start = time.perf_counter()
            _ = func(x)
            mediciones.append(time.perf_counter() - start)
        
        t_med = np.mean(mediciones[1:])  # Descartar primera medición (cold start)
        tiempos[nombre] = t_med
    
    # Normalizar respecto a ReLU
    t_relu = tiempos['ReLU']
    
    print(f"\n{'Función':<15} | {'Tiempo (ms)':<15} | {'Relativo a ReLU'}")
    print("-" * 50)
    for nombre, t in sorted(tiempos.items(), key=lambda x: x[1]):
        print(f"{nombre:<15} | {t*1000:<15.3f} | {t/t_relu:.2f}x")
    
    print(f"\n💡 ReLU es la más rápida por ser solo max(0,x)")
    print(f"   Sigmoid/Tanh son ~{tiempos['Sigmoid']/t_relu:.1f}x más lentas por las exponenciales")

benchmark_activaciones()
```

### Análisis del Impacto en Distribución de Activaciones

```python
def analizar_impacto_activaciones_en_red():
    """
    Analiza cómo cada activación afecta la distribución de
    activaciones a través de una red profunda (10 capas).
    """
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    n_capas = 10
    n_neuronas = 100
    X = np.random.randn(1000, n_neuronas)
    
    activaciones_test = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
    
    fig, axes = plt.subplots(len(activaciones_test), n_capas, 
                              figsize=(n_capas * 2.5, len(activaciones_test) * 2.5))
    
    for row, act_nombre in enumerate(activaciones_test):
        arquitectura = [n_neuronas] + [n_neuronas] * n_capas
        activaciones = [act_nombre] * n_capas
        red = RedNeuronalConActivaciones(arquitectura, activaciones, seed=42)
        
        activacion = X
        for col, capa in enumerate(red.capas):
            activacion = capa.forward(activacion)
            if isinstance(capa, CapaActivacion):
                idx_capa = col // 2  # Cada 2 capas hay una activación
                ax = axes[row, idx_capa]
                ax.hist(activacion.ravel(), bins=30, alpha=0.7, 
                       color=plt.cm.tab10(row), edgecolor='black', linewidth=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                if idx_capa == 0:
                    ax.set_ylabel(act_nombre.upper(), fontsize=10, fontweight='bold')
                if row == 0:
                    ax.set_title(f'Capa {idx_capa+1}', fontsize=9)
    
    plt.suptitle('Distribución de Activaciones por Capa y Función\n'
                 '(Cada columna = una capa, cada fila = una activación)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('distribucion_activaciones.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico guardado como 'distribucion_activaciones.png'")

analizar_impacto_activaciones_en_red()
```

---

## 🎯 EJERCICIOS PROPUESTOS

### Ejercicio 1: Verificación Numérica de Derivadas (Básico)

**Objetivo:** Verificar matemáticamente que las derivadas implementadas son correctas.

**Contexto:** La verificación numérica de gradientes (gradient checking) es una técnica estándar en deep learning para detectar bugs en la implementación de backpropagation.

**Tareas:**
1. Implementa `gradient_check(func, func_deriv, x, h=1e-5)` que compare derivada analítica vs numérica
2. Verifica las 4 funciones principales: ReLU, Sigmoid, Tanh, Leaky ReLU
3. Reporta el error relativo: `|df_analitica - df_numerica| / max(|df_analitica|, |df_numerica|, ε)`
4. Identifica en qué puntos el gradient check puede fallar (discontinuidades)

```python
def gradient_check(func, func_deriv, x, h=1e-5, verbose=True):
    """
    Verifica la derivada analítica contra la aproximación numérica.
    
    Usa diferencias centradas: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    
    Args:
        func: Función de activación f(x)
        func_deriv: Derivada analítica f'(x)
        x: Puntos donde verificar
        h: Paso para diferencias finitas
        verbose: Si imprimir resultados detallados
    
    Returns:
        error_relativo: Error relativo máximo
    """
    # Tu código aquí
    pass

# Verifica todas las funciones
x_test = np.array([-3.0, -1.0, -0.1, 0.1, 1.0, 3.0])
for nombre, (func, deriv) in ACTIVACIONES.items():
    print(f"\n{nombre}:")
    error = gradient_check(func, deriv, x_test)
```

### Ejercicio 2: Análisis de Saturación (Intermedio)

**Objetivo:** Cuantificar y visualizar el problema de saturación en redes profundas.

**Contexto:** La saturación ocurre cuando las activaciones se concentran en los extremos de la función, donde la derivada es prácticamente cero. Esto "mata" los gradientes.

**Tareas:**
1. Para cada función de activación, define el rango "activo" donde `|f'(x)| > 0.01`
2. Genera datos con diferentes distribuciones (normal, uniforme, sesgada)
3. Mide el porcentaje de activaciones en la zona saturada
4. Grafica la relación entre la escala de entrada y el porcentaje de saturación

```python
def analizar_saturacion(func_deriv, nombre, umbral_grad=0.01):
    """
    Mide el porcentaje de saturación para una función de activación.
    
    Args:
        func_deriv: Derivada de la función
        nombre: Nombre para mostrar
        umbral_grad: Umbral mínimo de gradiente para considerar "activo"
    
    Returns:
        dict: Estadísticas de saturación
    """
    # Tu código aquí
    pass

# Analiza las 4 funciones
for nombre, (func, deriv) in ACTIVACIONES.items():
    analizar_saturacion(deriv, nombre)
```

### Ejercicio 3: Red para el Problema XOR con Activaciones (Intermedio)

**Objetivo:** Demostrar que las funciones de activación no lineales permiten resolver problemas que una red lineal no puede.

**Contexto:** En Lab 02 demostramos que sin activaciones, una red multicapa es lineal. Ahora verificaremos que con activaciones, una red puede resolver XOR.

**Tareas:**
1. Crear los datos XOR: 4 puntos con etiquetas 0 y 1
2. Implementar entrenamiento manual (gradiente descendente básico, 1000 iteraciones)
3. Comparar la red lineal (sin activación) vs la red con ReLU o Sigmoid
4. Visualizar la frontera de decisión aprendida

```python
# Datos XOR
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)

def entrenar_paso(red, X, y, lr=0.1):
    """
    Un paso de entrenamiento simplificado (sin backprop completo).
    Esta es una aproximación educativa, no producción.
    """
    # Tu código aquí (puede ser entrenamiento numérico de gradiente)
    pass

# Prueba con red lineal vs red con activación
# ¿Cuál converge para XOR?
```

### Ejercicio 4: Softmax con Temperatura (Avanzado)

**Objetivo:** Entender el efecto de la temperatura en Softmax y su uso en modelos generativos.

**Contexto:** La temperatura $T$ en Softmax controla la "confianza" del modelo. Se usa en modelos de lenguaje (GPT, ChatGPT) para controlar la creatividad vs coherencia de las respuestas.

**Tareas:**
1. Implementa `softmax_temperatura(x, T)` = `softmax(x/T)`
2. Grafica la distribución para temperaturas: T = 0.1, 0.5, 1.0, 2.0, 10.0
3. Calcula la entropía de la distribución resultante para cada T
4. Explica: ¿qué temperatura usarías para un asistente de código preciso? ¿Y para escritura creativa?

```python
def softmax_temperatura(x, T=1.0):
    """
    Softmax con temperatura.
    
    T → 0: distribución más concentrada (greedy)
    T = 1: softmax estándar
    T → ∞: distribución uniforme (aleatoria)
    
    Args:
        x: Logits (batch_size, n_clases)
        T: Temperatura (escalar positivo)
    """
    return softmax(x / T)


def entropia(probs):
    """
    Calcula la entropía de Shannon de una distribución.
    H(p) = -sum(p * log(p))
    """
    # Tu código aquí
    pass

# Análisis completo con visualización
```

### Ejercicio 5: Función de Activación Personalizada — GELU (Proyecto)

**Objetivo:** Implementar y analizar GELU, la función de activación usada en GPT y BERT.

**Contexto:** GELU (Gaussian Error Linear Unit) fue propuesta en 2016 y se ha convertido en el estándar para transformers. Es una versión suavizada de ReLU que pondera cada activación por su probabilidad bajo una distribución gaussiana.

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

donde $\Phi(x)$ es la CDF de la distribución normal estándar.

**Tareas:**
1. Implementa GELU usando la aproximación de tanh
2. Implementa su derivada (puede ser numérica o analítica)
3. Compara con ReLU: rango, saturación, suavidad
4. Implementa y prueba una red usando GELU en todas las capas ocultas
5. Integra GELU en `ACTIVACIONES` y en `RedNeuronalConActivaciones`

```python
def gelu(x):
    """
    Gaussian Error Linear Unit (GELU).
    Usada en GPT-2, GPT-3, BERT, ViT, etc.
    
    Aproximación con tanh (más eficiente que usar scipy.stats):
    GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    
    Args:
        x: Array NumPy de cualquier shape
    Returns:
        Array del mismo shape
    """
    # Tu código aquí
    pass


def gelu_derivada(x):
    """Derivada de GELU (usa gradient_check para verificar)."""
    # Tu código aquí (puede ser aproximación numérica)
    pass

# Integración en el sistema
# Agrega GELU a ACTIVACIONES y prueba una red completa
```

---

## 📝 Entregables

### 1. Código Implementado (60%)

**Requisitos mínimos:**
- Todas las funciones de activación y sus derivadas: `relu`, `sigmoid`, `tanh`, `softmax`, `leaky_relu`
- Clase `CapaActivacion` con `forward()` y `backward()`
- Clase `RedNeuronalConActivaciones` con soporte para múltiples activaciones
- Al menos 3 ejercicios propuestos implementados y verificados
- Tests con `gradient_check` para verificar derivadas

**Criterios de calidad:**
- Código limpio, PEP8, con docstrings completos
- Manejo de casos borde (overflow en sigmoid, distribuciones extremas en softmax)
- Tests que verifican shapes, rangos y propiedades matemáticas

### 2. Notebook de Experimentación (25%)

**Debe incluir:**
- Todas las actividades de las partes 1-4 ejecutadas y analizadas
- Visualizaciones de funciones, derivadas, y distribuciones de activaciones
- Comparativa experimental de configuraciones (diferentes activaciones en capas ocultas)
- Demostración del problema del gradiente desvaneciente con gráficas
- Análisis de neuronas muertas con diferentes inicializaciones
- Respuestas escritas a todas las Preguntas de Reflexión

### 3. Reporte Técnico (15%)

**Secciones requeridas:**
1. Introducción: por qué las activaciones son esenciales
2. Marco teórico: descripción matemática de cada función y sus derivadas
3. Metodología: experimentos diseñados y realizados
4. Resultados: tablas comparativas, gráficas, gradient checks
5. Análisis y discusión: ventajas y limitaciones de cada activación
6. Conclusiones y recomendaciones: guía personal de selección de activaciones

**Extensión:** 3-5 páginas, formato PDF

### Formato de Entrega

```
Lab03_Entrega_NombreApellido/
├── codigo/
│   ├── activaciones.py          # Funciones de activación y derivadas
│   ├── red_con_activaciones.py  # Clases CapaActivacion y RedNeuronalConActivaciones
│   └── tests.py                 # Gradient checks y tests de propiedades
├── notebooks/
│   └── experimentos.ipynb
├── reporte/
│   └── reporte_lab03.pdf
└── README.md
```

---

## 🎯 Criterios de Evaluación (CDIO)

### Concebir (25%)

**Comprensión conceptual:**
- ✅ Explica por qué la no-linealidad es necesaria en redes profundas
- ✅ Comprende el problema del gradiente desvaneciente y su causa
- ✅ Distingue cuándo usar cada función de activación
- ✅ Entiende la relación entre activaciones y las derivadas en backpropagation

**Evidencia:** Respuestas a preguntas de reflexión, introducción del reporte

### Diseñar (25%)

**Planificación:**
- ✅ Diseña redes con activaciones apropiadas para cada tipo de problema
- ✅ Planifica experimentos para comparar funciones de activación
- ✅ Propone soluciones para neuronas muertas y vanishing gradient
- ✅ Considera stabilidad numérica en implementaciones

**Evidencia:** Ejercicios 1-5, sección de metodología del reporte

### Implementar (30%)

**Construcción:**
- ✅ Funciones de activación correctas (verificadas con gradient check)
- ✅ `CapaActivacion` con forward y backward funcionales
- ✅ `RedNeuronalConActivaciones` extensible y correcta
- ✅ Código documentado, limpio, con manejo de errores

**Evidencia:** Código fuente, resultados de tests

### Operar (20%)

**Validación y análisis:**
- ✅ Ejecuta benchmarks comparativos de velocidad
- ✅ Analiza distribuciones de activaciones en redes profundas
- ✅ Diagnostica y cuantifica neuronas muertas
- ✅ Extrae conclusiones prácticas sobre selección de activaciones

**Evidencia:** Notebook de experimentos, sección de resultados del reporte

### Rúbrica Detallada

| Criterio | Excelente (90-100%) | Bueno (75-89%) | Satisfactorio (60-74%) | Insuficiente (<60%) |
|----------|-------------------|----------------|----------------------|-------------------|
| **Implementación** | Todas las funciones correctas, gradient check <1e-6, código impecable | Funciones correctas, documentación básica | Mayoria de funciones correctas, errores menores | Funciones incorrectas o incompletas |
| **Comprensión teórica** | Explica intuición, derivadas, limitaciones con detalle | Correcto, aplica bien | Comprensión básica | Comprensión incorrecta o ausente |
| **Experimentación** | Experimentos creativos, hipótesis, conclusiones profundas | Todos los experimentos requeridos | Experimentos básicos | Experimentos incompletos |
| **Documentación** | Excelente: clara, matemáticamente rigurosa | Buena y completa | Básica | Pobre o ausente |

---

## 📚 Referencias Adicionales

### Libros

1. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*
   - Capítulo 6, Sección 6.3: Hidden Units (funciones de activación)
   - Capítulo 8, Sección 8.1: Vanishing and Exploding Gradients
   - http://www.deeplearningbook.org

2. **Nielsen, M.** (2015). *Neural Networks and Deep Learning*
   - Capítulo 1: Sigmoid neurons y el problema de activaciones
   - http://neuralnetworksanddeeplearning.com

3. **Zhang, A. et al.** (2023). *Dive into Deep Learning*
   - Capítulo 5: Multilayer Perceptrons (con activaciones)
   - https://d2l.ai

### Artículos Académicos

1. **Hochreiter, S.** (1991). "Untersuchungen zu dynamischen neuronalen Netzen"
   - Primera documentación del problema del gradiente desvaneciente

2. **Nair, V., & Hinton, G.E.** (2010). "Rectified linear units improve restricted Boltzmann machines"
   - Introducción de ReLU como función de activación práctica
   - *Proceedings of ICML*

3. **Glorot, X., Bordes, A., & Bengio, Y.** (2011). "Deep sparse rectifier neural networks"
   - Análisis de por qué ReLU funciona mejor que Sigmoid
   - *Proceedings of AISTATS*

4. **Hendrycks, D., & Gimpel, K.** (2016). "Gaussian Error Linear Units (GELUs)"
   - Propuesta de GELU, usada en GPT y BERT
   - arXiv:1606.08415

5. **Klambauer, G. et al.** (2017). "Self-Normalizing Neural Networks"
   - Propuesta de SELU, una alternativa auto-normalizante
   - *Proceedings of NeurIPS*

### Recursos Online

1. **3Blue1Brown — "Neural Networks" series (Capítulo 2)**
   - Gradiente descendente y backpropagation visualizados
   - https://www.youtube.com/watch?v=IHZwWFHWa-w

2. **CS231n: Activation Functions**
   - Análisis detallado de activaciones con visualizaciones
   - https://cs231n.github.io/neural-networks-1/#actfun

3. **Distill.pub — "Visualizing Neural Networks"**
   - Artículos interactivos de alta calidad
   - https://distill.pub

### Tutoriales Interactivos

1. **TensorFlow Playground**
   - Experimenta con activaciones en tiempo real
   - https://playground.tensorflow.org

2. **Seeing Theory — Probability and Statistics**
   - Para entender la base estadística de las funciones de activación
   - https://seeing-theory.brown.edu

### Documentación Técnica

- **NumPy**: https://numpy.org/doc/stable/reference/ufuncs.html — Operaciones elementwise
- **SciPy**: https://docs.scipy.org/doc/scipy/reference/special.html — Funciones especiales (erf, erfcinv)
- **Python**: https://docs.python.org/3/library/math.html — math.tanh, math.exp

---

## 🎓 Notas Finales

### Conceptos Clave para Recordar

1. **Sin no-linealidad = regresión lineal**
   - Cualquier red sin activaciones se reduce a $Y = XW + b$
   - Las activaciones son lo que diferencia el deep learning de la álgebra lineal

2. **ReLU: el estándar moderno** — `max(0, x)`
   - Extremadamente eficiente (solo una comparación)
   - Derivada constante (1) en región positiva → sin vanishing gradient
   - Genera sparsity (neuronas inactivas ≈ computación gratuita)
   - Problema: neuronas muertas (solucionable con Leaky ReLU o He init)

3. **Sigmoid** — `1/(1+e^(-x))`
   - Solo para la capa de salida de clasificación binaria
   - Nunca en capas ocultas de redes profundas (vanishing gradient)
   - Salida en (0,1): directamente interpretable como probabilidad

4. **Softmax** — normalización exponencial
   - Siempre en la capa de salida de clasificación multiclase
   - Produce distribución de probabilidad válida (suma = 1)
   - Sensible a escala: usa estabilización numérica (restar máximo)

5. **Tanh** — versión simétrica de Sigmoid
   - Mejor que Sigmoid para capas ocultas (centrada en 0)
   - Aún sufre vanishing gradient en redes muy profundas
   - Preferida en RNNs y LSTMs

6. **Las derivadas importan**: deben ser correctas para que backpropagation funcione
   - Verifica siempre con `gradient_check` antes de entrenar

7. **La elección de activación afecta**:
   - Velocidad de convergencia
   - Estabilidad del entrenamiento
   - Capacidad representacional
   - Interpretabilidad de las salidas

8. **Temperatura en Softmax**: parámetro clave en modelos generativos
   - T < 1: predicciones más deterministas
   - T > 1: predicciones más diversas

### Preparación para el Siguiente Lab

**Lab 04: Funciones de Pérdida** te enseñará cómo medir el error de la red y cómo usarlo para ajustar los parámetros.

Aprenderás:
- **MSE** (Mean Squared Error): para regresión
- **MAE** (Mean Absolute Error): más robusto a outliers
- **Binary Cross-Entropy**: para clasificación binaria (con Sigmoid)
- **Categorical Cross-Entropy**: para clasificación multiclase (con Softmax)
- Por qué Sigmoid + Binary Cross-Entropy funcionan juntos naturalmente

**Para prepararte:**
1. Revisa logaritmos naturales y sus derivadas: $\frac{d}{dx}\ln(x) = \frac{1}{x}$
2. Piensa en qué significa "medir el error" entre probabilidades
3. Reflexiona: ¿por qué `argmax(softmax(x)) == argmax(x)`?
4. Investiga qué es "cross-entropy" en teoría de la información

### Consejos de Estudio

1. **Verifica tus derivadas**: usa `gradient_check` siempre
2. **Visualiza todo**: grafica funciones y sus derivadas juntas
3. **Experimenta con temperatura**: observa cómo cambia la distribución
4. **Diagnostica activamente**: ejecuta `analizar_neuronas_muertas` en tus redes
5. **Lee código de otros**: TensorFlow y PyTorch tienen implementaciones de referencia
6. **Comprende las discontinuidades**: ¿en qué puntos ReLU no es diferenciable?
7. **Usa `np.clip` con sabiduría**: evita overflow en Sigmoid con valores muy extremos

### Solución de Problemas Comunes

**Problema: `RuntimeWarning: overflow encountered in exp`**
- **Causa**: Sigmoid aplicada a valores muy grandes (e.g., 1000)
- **Diagnóstico**: Verificar rango de valores de entrada: `print(x.min(), x.max())`
- **Solución**: Usar la implementación numéricamente estable con `np.where`

**Problema: Softmax devuelve `nan` o `inf`**
- **Causa**: Overflow en `np.exp(x)` para valores grandes
- **Diagnóstico**: `np.max(x)` es muy grande
- **Solución**: Aplicar estabilización numérica: restar `np.max(x, axis=-1, keepdims=True)`

**Problema: Muchas neuronas muertas (activación siempre 0)**
- **Causa**: Biases negativos grandes o learning rate muy alto
- **Diagnóstico**: `analizar_neuronas_muertas(red, X)`
- **Solución 1**: Usar Leaky ReLU o ELU en lugar de ReLU
- **Solución 2**: Reducir learning rate
- **Solución 3**: Inicialización He para biases positivos

**Problema: El gradient check falla para ReLU en x=0**
- **Causa**: ReLU no es diferenciable en x=0
- **Diagnóstico**: El punto de evaluación está exactamente en 0
- **Solución**: Es un comportamiento esperado; el gradient check es válido para x ≠ 0

**Problema: Entrenamiento muy lento (sospecha de vanishing gradient)**
- **Diagnóstico**: `demostrar_gradiente_desaparece(n_capas=len(red.capas)//2)`
- **Solución 1**: Cambiar Sigmoid/Tanh por ReLU en capas ocultas
- **Solución 2**: Reducir número de capas
- **Solución 3**: Usar técnicas avanzadas: Batch Normalization, residual connections

### Comunidad y Soporte

- **Foro del curso**: Para preguntas conceptuales sobre activaciones
- **Horas de oficina**: Para revisión de implementaciones y gradient check
- **Papers with Code**: Implementaciones de referencia para todas las activaciones
  - https://paperswithcode.com/methods/category/activation-functions
- **Stack Overflow**: Para errores específicos de NumPy/overflow

### Certificación de Completitud

Has completado exitosamente el Lab 03 cuando puedas:

- [ ] Explicar intuitivamente por qué sin activaciones una red es lineal
- [ ] Implementar ReLU, Sigmoid, Tanh, Softmax y Leaky ReLU desde cero
- [ ] Verificar las derivadas con gradient check (error < 1e-5)
- [ ] Demostrar el vanishing gradient con Sigmoid en 10+ capas
- [ ] Identificar neuronas muertas en una red con ReLU
- [ ] Elegir la activación correcta para la capa de salida según el tipo de problema
- [ ] Integrar activaciones en la arquitectura modular del Lab 02
- [ ] Comparar la velocidad de ejecución de diferentes activaciones
- [ ] Implementar Softmax numéricamente estable y explicar por qué

---

**¡Felicitaciones por completar el Lab 03!** Ahora tus redes neuronales tienen la capacidad de aprender patrones no lineales complejos.

**Siguiente parada**: Lab 04 — Funciones de Pérdida 🚀

---

*Versión: 2.0 | Actualizado: 2024 | Licencia: MIT — Uso educativo*
