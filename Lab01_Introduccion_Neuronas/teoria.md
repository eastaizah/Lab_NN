# Lab 01: Introducción a las Neuronas

## Fundamentos Teóricos

### ¿Qué es una Neurona Artificial?

Una neurona artificial es la unidad básica de procesamiento en una red neuronal, inspirada en las neuronas biológicas del cerebro humano. Aunque mucho más simple que su contraparte biológica, la neurona artificial es capaz de aprender patrones complejos cuando se combina con otras neuronas.

### Componentes de una Neurona Artificial

#### 1. Entradas (Inputs)
- Las entradas son los datos que la neurona recibe
- Pueden ser características de un conjunto de datos (ej: edad, altura, peso)
- Se representan como un vector: **x** = [x₁, x₂, x₃, ..., xₙ]

#### 2. Pesos (Weights)
- Cada entrada tiene un peso asociado
- Los pesos determinan la importancia de cada entrada
- Se representan como: **w** = [w₁, w₂, w₃, ..., wₙ]
- Los pesos son los parámetros que la red "aprende" durante el entrenamiento

#### 3. Bias (Sesgo)
- Es un valor adicional que se suma a la combinación lineal
- Permite a la neurona ajustar su salida independientemente de las entradas
- Se representa como: **b**
- El bias también es un parámetro aprendible

#### 4. Función de Suma Ponderada
La neurona calcula una suma ponderada de sus entradas:

```
z = (x₁ × w₁) + (x₂ × w₂) + ... + (xₙ × wₙ) + b
```

En notación vectorial:
```
z = x · w + b
```

Donde "·" representa el producto punto (dot product) entre los vectores.

#### 5. Función de Activación
- Transforma la suma ponderada en la salida final de la neurona
- Introduce no-linealidad en el modelo
- Por ahora, usaremos la función identidad (la salida es igual a la entrada)

### Ejemplo Conceptual

Imaginemos una neurona que decide si vamos a una fiesta basándose en tres factores:

1. **Clima** (x₁): 0 = malo, 1 = bueno
2. **Cansancio** (x₂): 0 = descansado, 1 = cansado
3. **Amigos que van** (x₃): número de amigos (0-10)

Pesos que indican importancia:
- w₁ = 0.3 (el clima es algo importante)
- w₂ = -0.5 (el cansancio es muy importante, negativamente)
- w₃ = 0.8 (los amigos son muy importantes)

Bias:
- b = -2.0 (en general, preferimos quedarnos en casa)

Si:
- Clima bueno (x₁ = 1)
- Estamos descansados (x₂ = 0)
- Van 5 amigos (x₃ = 5)

Cálculo:
```
z = (1 × 0.3) + (0 × -0.5) + (5 × 0.8) + (-2.0)
z = 0.3 + 0 + 4.0 - 2.0
z = 2.3
```

Un valor positivo alto sugiere que deberíamos ir a la fiesta.

### Representación Matemática Completa

Para una neurona con n entradas:

**Entrada**: x = [x₁, x₂, ..., xₙ]  
**Pesos**: w = [w₁, w₂, ..., wₙ]  
**Bias**: b  
**Suma ponderada**: z = Σᵢ(xᵢ × wᵢ) + b  
**Salida**: y = f(z)

Donde f es la función de activación (por ahora, f(z) = z)

### Implementación en Código

En Python con NumPy, el producto punto se calcula eficientemente:

```python
import numpy as np

# Entradas
inputs = np.array([1.0, 2.0, 3.0])

# Pesos
weights = np.array([0.2, 0.8, -0.5])

# Bias
bias = 2.0

# Cálculo de la salida
output = np.dot(inputs, weights) + bias
```

### Por qué las Neuronas son Poderosas

1. **Flexibilidad**: Con los pesos y bias adecuados, pueden representar diversas funciones
2. **Aprendizaje**: Los pesos se ajustan automáticamente usando datos
3. **Escalabilidad**: Se pueden combinar múltiples neuronas para problemas complejos
4. **No-linealidad**: Con funciones de activación, pueden modelar relaciones complejas

### Limitaciones de una Sola Neurona

- Solo puede aprender patrones lineales (o casi lineales)
- No puede resolver problemas como XOR
- Necesita combinarse con otras neuronas para tareas complejas

### Próximos Pasos

En el siguiente laboratorio:
- Combinaremos múltiples neuronas en capas
- Crearemos nuestra primera red neuronal
- Aprenderemos sobre diferentes arquitecturas

## Conceptos Clave para Recordar

1. **Neurona = Suma Ponderada + Bias + Activación**
2. **Pesos**: Parámetros aprendibles que determinan importancia
3. **Bias**: Parámetro aprendible que ajusta el umbral
4. **Producto Punto**: Operación fundamental para calcular la suma ponderada
5. **NumPy**: Librería esencial para operaciones eficientes con vectores y matrices

## Ejercicios de Reflexión

1. ¿Qué sucede si todos los pesos son cero?
2. ¿Cómo afecta un bias muy grande a la neurona?
3. ¿Por qué usamos el producto punto en lugar de sumar manualmente?
4. ¿Puede una neurona simple resolver cualquier problema? ¿Por qué sí o no?
