# Lab 02: Primera Red Neuronal

## Fundamentos Teóricos

### De Neuronas a Redes Neuronales

En el laboratorio anterior aprendimos sobre neuronas individuales. Ahora veremos cómo conectar múltiples neuronas para crear redes neuronales capaces de resolver problemas complejos.

### ¿Qué es una Red Neuronal?

Una red neuronal es un conjunto de neuronas organizadas en capas que trabajan juntas para transformar datos de entrada en predicciones útiles.

### Arquitectura de una Red Neuronal

#### 1. Capa de Entrada (Input Layer)
- No es realmente una capa de neuronas, sino los datos de entrada
- Cada característica del dato es un nodo de entrada
- Ejemplo: para imágenes 28x28, tendríamos 784 entradas

#### 2. Capas Ocultas (Hidden Layers)
- Capas intermedias entre entrada y salida
- Cada capa puede tener cualquier número de neuronas
- Extraen características progresivamente más complejas
- Una red con 2+ capas ocultas se considera "deep learning"

#### 3. Capa de Salida (Output Layer)
- Produce la predicción final
- El número de neuronas depende del problema:
  - Clasificación binaria: 1 o 2 neuronas
  - Clasificación multiclase: 1 neurona por clase
  - Regresión: típicamente 1 neurona

### Ejemplo de Arquitectura

Consideremos una red para clasificar dígitos escritos a mano (0-9):

```
[Entrada: 784 píxeles] → [Capa Oculta 1: 128 neuronas] → [Capa Oculta 2: 64 neuronas] → [Salida: 10 neuronas]
```

### Conexiones entre Capas

#### Forward Propagation (Propagación hacia Adelante)

El proceso de calcular la salida de la red:

1. **Entrada → Capa Oculta 1**:
   ```
   H1 = X · W1 + B1
   ```
   - X: datos de entrada (batch_size, 784)
   - W1: pesos de la primera capa (784, 128)
   - B1: biases de la primera capa (128)
   - H1: salida de la primera capa (batch_size, 128)

2. **Capa Oculta 1 → Capa Oculta 2**:
   ```
   H2 = H1 · W2 + B2
   ```
   - H1: entrada desde la capa anterior (batch_size, 128)
   - W2: pesos de la segunda capa (128, 64)
   - B2: biases de la segunda capa (64)
   - H2: salida de la segunda capa (batch_size, 64)

3. **Capa Oculta 2 → Salida**:
   ```
   Y = H2 · W3 + B3
   ```
   - H2: entrada desde la capa anterior (batch_size, 64)
   - W3: pesos de la capa de salida (64, 10)
   - B3: biases de la capa de salida (10)
   - Y: predicciones finales (batch_size, 10)

### Dimensiones de las Matrices

Entender las dimensiones es crucial:

```python
# Para un batch de 32 muestras con 784 características cada una:
X.shape = (32, 784)     # Entrada
W1.shape = (784, 128)   # Pesos capa 1
H1.shape = (32, 128)    # Salida capa 1

# Regla general:
# Si entrada es (n, m) y pesos son (m, k)
# Entonces salida es (n, k)
```

### Notación Matemática

Para una red con L capas:

**Capa l (l = 1, 2, ..., L)**:
- **W^(l)**: matriz de pesos de la capa l
- **b^(l)**: vector de biases de la capa l
- **a^(l)**: activaciones (salidas) de la capa l
- **z^(l)**: valores pre-activación de la capa l

**Forward pass para la capa l**:
```
z^(l) = a^(l-1) · W^(l) + b^(l)
a^(l) = f(z^(l))
```

Donde:
- a^(0) = X (la entrada)
- f es la función de activación (por ahora, identidad)

### ¿Por qué Múltiples Capas?

#### Teorema de Aproximación Universal
Una red neuronal con una sola capa oculta puede aproximar cualquier función continua. Entonces, ¿por qué usar múltiples capas?

**Ventajas de Capas Múltiples**:

1. **Eficiencia**: 
   - Capas profundas pueden representar funciones complejas con menos neuronas
   - Una función que requiere 2^n neuronas en 1 capa, puede requerir solo O(n) neuronas en varias capas

2. **Jerarquía de Características**:
   - Capas tempranas: características simples (bordes, texturas)
   - Capas medias: características compuestas (formas)
   - Capas finales: conceptos abstractos (objetos completos)

3. **Mejor Generalización**:
   - La estructura profunda actúa como una forma de regularización
   - Aprende representaciones más robustas

### Ejemplo Conceptual

**Red para Reconocimiento de Rostros**:

```
Entrada: Imagen 64x64x3 (RGB)
    ↓
Capa 1 (128 neuronas): Detecta bordes básicos
    ↓
Capa 2 (64 neuronas): Combina bordes en formas (ojos, nariz, boca)
    ↓
Capa 3 (32 neuronas): Combina formas en partes de rostro
    ↓
Salida (1 neurona): ¿Es un rostro? (0 o 1)
```

### Número de Parámetros

Para una red, el número total de parámetros aprendibles es:

```
Para cada capa l:
  Parámetros = (n_entradas × n_neuronas) + n_neuronas
               \_____________pesos_____________/   \__bias__/
```

**Ejemplo**:
- Capa 1: (784 × 128) + 128 = 100,480 parámetros
- Capa 2: (128 × 64) + 64 = 8,256 parámetros
- Capa 3: (64 × 10) + 10 = 650 parámetros
- **Total**: 109,386 parámetros

### Inicialización de Pesos

¿Por qué no inicializar todos los pesos a cero?

**Problema de Simetría**:
- Si todos los pesos son iguales, todas las neuronas aprenden lo mismo
- La red se comporta como si tuviera una sola neurona por capa

**Soluciones Comunes**:

1. **Random pequeño**: `W = np.random.randn(n, m) * 0.01`
   - Simple, funciona bien para redes pequeñas

2. **Xavier/Glorot**: `W = np.random.randn(n, m) * np.sqrt(1/n)`
   - Mantiene la varianza estable entre capas

3. **He**: `W = np.random.randn(n, m) * np.sqrt(2/n)`
   - Recomendado para redes con ReLU

### Implementación en Código

```python
import numpy as np

class RedNeuronal:
    def __init__(self, arquitectura):
        """
        Args:
            arquitectura: lista con número de neuronas por capa
                         ej: [784, 128, 64, 10]
        """
        self.capas = []
        
        # Crear capas
        for i in range(len(arquitectura) - 1):
            n_entradas = arquitectura[i]
            n_neuronas = arquitectura[i + 1]
            
            capa = {
                'pesos': np.random.randn(n_entradas, n_neuronas) * 0.01,
                'biases': np.zeros(n_neuronas)
            }
            self.capas.append(capa)
    
    def forward(self, X):
        """Calcula la salida de la red."""
        activacion = X
        
        for capa in self.capas:
            z = np.dot(activacion, capa['pesos']) + capa['biases']
            activacion = z  # Por ahora, sin función de activación
        
        return activacion
```

### Flujo de Datos en una Red

```
Entrada (X)
    ↓
[Operación Lineal: X·W1 + b1]
    ↓
Activación 1 (a1)
    ↓
[Operación Lineal: a1·W2 + b2]
    ↓
Activación 2 (a2)
    ↓
[Operación Lineal: a2·W3 + b3]
    ↓
Salida (Y)
```

### Limitaciones Sin Funciones de Activación

**Problema**: Sin funciones de activación no lineales, cualquier red profunda es equivalente a una red de una sola capa.

**Demostración**:
```
H1 = X·W1 + b1
H2 = H1·W2 + b2 = (X·W1 + b1)·W2 + b2
   = X·(W1·W2) + (b1·W2 + b2)
   = X·W + b    (donde W = W1·W2, b = b1·W2 + b2)
```

Por lo tanto, en el siguiente laboratorio introduciremos funciones de activación.

### Conceptos Clave para Recordar

1. **Red Neuronal = Capas de Neuronas Conectadas**
2. **Forward Propagation**: Cálculo de la salida desde entrada hasta salida
3. **Dimensiones**: Clave para entender el flujo de datos
4. **Múltiples Capas**: Permiten aprender jerarquías de características
5. **Inicialización**: Los pesos iniciales afectan el aprendizaje
6. **Sin Activación No Lineal**: La red es esencialmente lineal

### Próximos Pasos

En el Lab 03:
- Introduciremos funciones de activación (ReLU, Sigmoid, Softmax)
- Veremos cómo añaden no linealidad
- Entenderemos cuándo usar cada una

## Ejercicios de Reflexión

1. ¿Cuántos parámetros tiene una red [100, 50, 10]?
2. ¿Qué forma tienen las matrices de pesos entre cada par de capas?
3. ¿Por qué una red sin funciones de activación es equivalente a una regresión lineal?
4. ¿Cómo cambiarían las dimensiones si procesamos un batch de 64 muestras?
