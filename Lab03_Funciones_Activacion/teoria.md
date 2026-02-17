# Teoría: Funciones de Activación

## Introducción

Las funciones de activación son componentes esenciales en las redes neuronales que introducen **no linealidad** en el modelo. Sin ellas, una red neuronal profunda sería equivalente a una simple regresión lineal, sin importar cuántas capas tenga.

## ¿Por qué necesitamos funciones de activación?

Imagina que queremos que nuestra red neuronal aprenda patrones complejos como:
- Reconocer si una imagen contiene un gato o un perro
- Predecir si un estudiante aprobará basándose en múltiples factores
- Clasificar correos electrónicos como spam o no spam

Estos problemas no son lineales. La relación entre las entradas y salidas no puede representarse con una simple línea recta. Las funciones de activación permiten que la red aprenda estas relaciones complejas.

## Principales Funciones de Activación

### 1. Sigmoid (Sigmoide)

**Ecuación:**
```
σ(x) = 1 / (1 + e^(-x))
```

**Derivada:**
```
σ'(x) = σ(x) * (1 - σ(x))
```

**Características:**
- Rango de salida: (0, 1)
- Forma de "S" suave
- Útil para probabilidades

**Ventajas:**
- Salida interpretable como probabilidad
- Suave y diferenciable en todo su dominio
- Históricamente importante

**Desventajas:**
- **Problema del gradiente que desaparece**: Para valores muy grandes o muy pequeños, la derivada se acerca a 0
- Salidas no centradas en cero
- Computacionalmente costosa (exponencial)

**Cuándo usar:**
- Capa de salida en clasificación binaria
- Cuando necesitas salidas en rango (0, 1)

### 2. Tanh (Tangente Hiperbólica)

**Ecuación:**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

O también:
```
tanh(x) = 2σ(2x) - 1
```

**Derivada:**
```
tanh'(x) = 1 - tanh²(x)
```

**Características:**
- Rango de salida: (-1, 1)
- Centrada en cero
- Similar a sigmoid pero mejorada

**Ventajas:**
- Salidas centradas en cero (mejor que sigmoid)
- Gradientes más fuertes que sigmoid
- Convergencia más rápida en la práctica

**Desventajas:**
- Todavía sufre de gradiente que desaparece
- Computacionalmente costosa

**Cuándo usar:**
- Capas ocultas en redes recurrentes (RNN, LSTM)
- Cuando necesitas salidas centradas en cero

### 3. ReLU (Rectified Linear Unit)

**Ecuación:**
```
ReLU(x) = max(0, x)
```

**Derivada:**
```
ReLU'(x) = 1 si x > 0
          = 0 si x ≤ 0
```

**Características:**
- Rango de salida: [0, ∞)
- Extremadamente simple
- No saturación en región positiva

**Ventajas:**
- **Muy eficiente computacionalmente**
- Mitiga el problema del gradiente que desaparece
- Convergencia más rápida que sigmoid/tanh
- Genera sparsity (algunas neuronas se "apagan")

**Desventajas:**
- **Problema de "neuronas muertas"**: Si una neurona siempre recibe valores negativos, su gradiente es siempre 0
- No centrada en cero
- No acotada superiormente

**Cuándo usar:**
- **Opción por defecto** para capas ocultas en redes profundas
- CNNs (Redes Convolucionales)
- La mayoría de arquitecturas modernas

### 4. Leaky ReLU

**Ecuación:**
```
LeakyReLU(x) = max(αx, x)  donde α es pequeño (típicamente 0.01)
```

O también:
```
LeakyReLU(x) = x si x > 0
              = αx si x ≤ 0
```

**Derivada:**
```
LeakyReLU'(x) = 1 si x > 0
               = α si x ≤ 0
```

**Ventajas:**
- Soluciona el problema de neuronas muertas de ReLU
- Permite gradientes pequeños cuando x < 0

**Cuándo usar:**
- Alternativa a ReLU cuando se observan muchas neuronas muertas

### 5. Softmax

**Ecuación (para un vector de entrada z):**
```
Softmax(z_i) = e^(z_i) / Σ(e^(z_j)) para j = 1 a n
```

**Características:**
- Convierte un vector de valores reales en probabilidades
- La suma de todas las salidas es 1
- Cada salida está en rango (0, 1)

**Ventajas:**
- Interpretación probabilística clara
- Diferenciable
- Considera todas las clases simultáneamente

**Desventajas:**
- Computacionalmente más costosa
- Sensible a valores muy grandes (overflow)

**Cuándo usar:**
- **Capa de salida en clasificación multiclase**
- Cuando necesitas probabilidades para múltiples clases

## Comparación Visual

```
Sigmoid:     ___/‾‾‾        (0 a 1, forma S)
Tanh:       __/‾‾‾          (-1 a 1, forma S centrada)
ReLU:       __/‾‾‾          (0 a ∞, codo en 0)
LeakyReLU:  _/‾‾‾           (pendiente pequeña en negativos)
```

## Reglas Generales de Uso

| Situación | Función Recomendada |
|-----------|---------------------|
| Capas ocultas (general) | ReLU o Leaky ReLU |
| Clasificación binaria (salida) | Sigmoid |
| Clasificación multiclase (salida) | Softmax |
| Regresión (salida) | Lineal (sin activación) |
| RNNs/LSTMs | Tanh |
| GANs | Leaky ReLU, Tanh |

## El Problema del Gradiente que Desaparece

Cuando usamos funciones como sigmoid o tanh en redes profundas:

1. Durante backpropagation, multiplicamos muchos gradientes pequeños
2. Estos gradientes se vuelven exponencialmente pequeños
3. Las capas iniciales aprenden muy lentamente
4. El entrenamiento se estanca

**Solución:** Usar ReLU y sus variantes, que mantienen gradientes constantes en la región positiva.

## Elección de la Función de Activación

**Proceso recomendado:**

1. **Empieza con ReLU** para capas ocultas
2. Si observas neuronas muertas → prueba Leaky ReLU
3. Para clasificación binaria → Sigmoid en la salida
4. Para clasificación multiclase → Softmax en la salida
5. Para RNNs → Tanh en puertas internas
6. Experimenta y ajusta según los resultados

## Implementación Matemática

### Ejemplo: Forward y Backward de ReLU

**Forward:**
```python
def relu_forward(x):
    return np.maximum(0, x)
```

**Backward:**
```python
def relu_backward(dout, cache):
    x = cache
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx
```

## Conceptos Clave para Recordar

1. **No linealidad es esencial**: Sin activación, la red es lineal
2. **ReLU es el estándar**: Simple, eficiente, efectiva
3. **Softmax para multiclase**: Convierte scores en probabilidades
4. **Sigmoid para binaria**: Salida entre 0 y 1
5. **Gradientes importan**: Elige funciones que no saturen

## Ejercicios de Reflexión

1. ¿Qué pasaría si usáramos solo funciones lineales?
2. ¿Por qué ReLU es tan popular a pesar de su simpleza?
3. ¿En qué situaciones Sigmoid sigue siendo útil?
4. ¿Cómo afecta la elección de activación al tiempo de entrenamiento?

## Próximos Pasos

En el siguiente laboratorio implementaremos estas funciones desde cero y las visualizaremos para entender mejor su comportamiento y derivadas.
