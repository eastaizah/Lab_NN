# Guía de Laboratorio: Primera Red Neuronal Multicapa

## 📋 Información del Laboratorio

**Título:** Primera Red Neuronal Multicapa  
**Código:** Lab 02  
**Duración:** 2-3 horas  
**Nivel:** Básico-Intermedio  

---

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender la arquitectura de redes neuronales multicapa y el papel de cada tipo de capa
2. Implementar forward propagation desde cero siguiendo el flujo de datos capa a capa
3. Diseñar arquitecturas apropiadas para diferentes tipos de problemas
4. Calcular el número de parámetros aprendibles en cualquier red
5. Entender y rastrear el flujo de dimensiones (shapes) a través de las capas
6. Implementar redes neuronales usando programación orientada a objetos
7. Visualizar y analizar las activaciones intermedias de cada capa
8. Reconocer y demostrar las limitaciones sin funciones de activación no lineal
9. Aplicar buenas prácticas de inicialización de pesos

---

## 📚 Prerrequisitos

### Conocimientos

- **Lab 01 completado**: Neuronas individuales, producto punto, vectorización con NumPy
- Python intermedio: clases, métodos, herencia, comprensión de listas
- Álgebra lineal básica: vectores, matrices, multiplicación matricial
- NumPy básico: arrays, operaciones matriciales, broadcasting

### Software

- Python 3.8+
- NumPy 1.19+
- Matplotlib 3.0+
- Jupyter Notebook (recomendado)

### Material de Lectura

Antes de comenzar este laboratorio:
- `teoria.md` — Marco teórico completo sobre arquitecturas de redes neuronales
- `README.md` — Visión general del laboratorio y estructura de archivos
- Repasa la sección 2 de Lab01 (capas de neuronas y operaciones matriciales)

---

## 📖 Introducción

En el Lab 01 aprendiste a construir **neuronas individuales**: unidades computacionales que calculan una suma ponderada de sus entradas. Ahora daremos el siguiente paso natural: conectar múltiples neuronas en **capas** y múltiples capas en **redes neuronales profundas**.

### Contexto del Problema

Las neuronas individuales, aunque poderosas para operaciones simples, tienen una limitación fundamental: solo pueden aprender **patrones linealmente separables**. En otras palabras, solo pueden resolver problemas donde los datos de distintas clases se separan con una línea recta (o hiperplano).

Esta limitación se ilustra perfectamente con el problema **XOR**:

```
 (0,0)→0    (1,1)→0  ← Clase 0 (puntos en diagonal)
 (0,1)→1    (1,0)→1  ← Clase 1 (puntos en diagonal)
```

No existe ninguna línea recta que separe perfectamente estas dos clases. Se necesita una **red neuronal multicapa** para resolverlo.

### Enfoque con Redes Neuronales

Una red neuronal multicapa organiza las neuronas en capas conectadas:

```
DATOS           CAPA OCULTA 1    CAPA OCULTA 2    SALIDA
[x₁]  ──┐       [h₁₁]           [h₂₁]            [y₁]
[x₂]  ──┼──→   [h₁₂]  ──→     [h₂₂]  ──→       [y₂]
[x₃]  ──┘       [h₁₃]           [h₂₃]            ...
                [h₁₄]
```

Arquitectura típica para clasificación de dígitos MNIST:
```
[784 píxeles] → [128 neuronas] → [64 neuronas] → [10 clases]
```

Cada flecha `→` representa una capa de conexiones ponderadas (pesos + biases).

### Conceptos Fundamentales

**1. Forward Propagation (Propagación hacia adelante):**

El proceso de calcular la salida de la red, capa por capa:

$$\mathbf{z}^{(l)} = \mathbf{a}^{(l-1)} \cdot \mathbf{W}^{(l)} + \mathbf{b}^{(l)}$$

$$\mathbf{a}^{(l)} = f\left(\mathbf{z}^{(l)}\right)$$

Donde:
- $\mathbf{a}^{(0)} = \mathbf{X}$ (datos de entrada)
- $\mathbf{W}^{(l)}$ es la matriz de pesos de la capa $l$
- $\mathbf{b}^{(l)}$ es el vector de biases de la capa $l$
- $f$ es la función de activación (en este lab, identidad)

**2. Dimensiones de Tensores:**

Para un batch de $N$ muestras con $d$ características:

| Tensor | Shape | Descripción |
|--------|-------|-------------|
| X (entrada) | $(N, d)$ | N muestras, d características cada una |
| W (pesos) | $(d_{in}, d_{out})$ | Conexiones entre capas |
| b (biases) | $(d_{out},)$ | Un bias por neurona |
| a (activación) | $(N, d_{out})$ | N muestras, $d_{out}$ activaciones |

**Regla fundamental**: Si la entrada tiene shape $(N, m)$ y los pesos son $(m, k)$, la salida tiene shape $(N, k)$.

**3. Número de Parámetros:**

Para cada capa densa:

$$\text{parámetros} = (n_{entradas} \times n_{neuronas}) + n_{neuronas}$$

Para la red MNIST completa:
- Capa 1: $(784 \times 128) + 128 = 100{,}480$
- Capa 2: $(128 \times 64) + 64 = 8{,}256$
- Capa 3: $(64 \times 10) + 10 = 650$
- **Total: 109,386 parámetros aprendibles**

### Aplicaciones Prácticas

Las redes neuronales multicapa son la base de:
- **Visión por computadora**: ResNet, VGG, EfficientNet (reconocimiento de imágenes)
- **Procesamiento de lenguaje**: BERT, GPT (traducción, resumen, chatbots)
- **Sistemas de recomendación**: Netflix, Spotify, YouTube
- **Diagnóstico médico**: detección de tumores, clasificación de radiografías
- **Finanzas**: predicción de mercados, detección de fraude

### Motivación Histórica

El perceptrón simple de Rosenblatt (1958) era una sola neurona. En 1969, Minsky y Papert demostraron matemáticamente que no podía resolver XOR. Esto provocó el primer "invierno de la IA". En los 80s, Rumelhart et al. desarrollaron el algoritmo de backpropagation, que permitió entrenar redes multicapa y superar las limitaciones del perceptrón simple — dando inicio a la era del deep learning.

---

## 🔬 Parte 1: Construyendo Tu Primera Red Multicapa (35 min)

### 1.1 Introducción Conceptual: ¿Cómo se conectan las capas?

**¿Qué hacemos?** Conectar múltiples capas de neuronas de forma secuencial.

**¿Por qué lo hacemos?** Cada capa transforma la representación de los datos. Las capas tempranas aprenden características simples (bordes, colores) y las capas posteriores combinan estas en conceptos complejos (formas, objetos).

**Analogía:** Imagina un equipo de análisis de texto:
- **Capa 1** (analista de palabras): identifica palabras individuales
- **Capa 2** (analista de frases): agrupa palabras en frases con significado
- **Capa 3** (analista de sentimiento): determina si el mensaje es positivo o negativo

Cada nivel depende del nivel anterior y agrega una nueva capa de comprensión.

**¿Qué resultados esperar?** Un tensor de salida cuyo shape depende de la arquitectura definida.

### 1.2 Dos Capas Conectadas Manualmente

Empecemos con lo más básico: dos capas conectadas sin clases:

```python
import numpy as np

# Arquitectura: 3 entradas → 4 neuronas ocultas → 2 salidas
print("=" * 50)
print("RED DE DOS CAPAS: 3 → 4 → 2")
print("=" * 50)

# Datos de entrada (1 muestra con 3 características)
X = np.array([[1.0, 2.0, 3.0]])  # Shape: (1, 3)

# Capa 1: 3 entradas → 4 neuronas
W1 = np.random.randn(3, 4) * 0.01  # Shape: (3, 4)
b1 = np.zeros(4)                    # Shape: (4,)
a1 = X @ W1 + b1                   # Shape: (1, 4)

print(f"\n📐 Capa 1:")
print(f"   Entrada X: {X.shape}")
print(f"   Pesos W1: {W1.shape}")
print(f"   Salida a1: {a1.shape}")

# Capa 2: 4 neuronas → 2 salidas
W2 = np.random.randn(4, 2) * 0.01  # Shape: (4, 2)
b2 = np.zeros(2)                    # Shape: (2,)
salida = a1 @ W2 + b2              # Shape: (1, 2)

print(f"\n📐 Capa 2:")
print(f"   Entrada a1: {a1.shape}")
print(f"   Pesos W2: {W2.shape}")
print(f"   Salida final: {salida.shape}")
print(f"\n🔢 Resultado: {salida}")
```

**Actividad 1.1**: Ejecuta el código anterior. Verifica manualmente que las shapes de cada operación son correctas. ¿Qué ocurre si intentas usar W1 con shape `(4, 3)` en vez de `(3, 4)`?

**Actividad 1.2**: Modifica el código para una arquitectura `[5, 8, 6, 3]`. Traza las shapes en cada paso.

### 1.3 Red Completa para MNIST

Ahora implementemos la arquitectura clásica para clasificar dígitos:

```python
# Red [784, 128, 64, 10] — Arquitectura para MNIST
np.random.seed(42)  # Reproducibilidad

print("=" * 60)
print("RED NEURONAL PARA MNIST: [784, 128, 64, 10]")
print("=" * 60)

# Simular un batch de 32 imágenes 28x28 aplanadas
X = np.random.randn(32, 784)  # Shape: (32, 784)

# Capa 1: 784 → 128
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros(128)
a1 = X @ W1 + b1  # Shape: (32, 128)

# Capa 2: 128 → 64
W2 = np.random.randn(128, 64) * 0.01
b2 = np.zeros(64)
a2 = a1 @ W2 + b2  # Shape: (32, 64)

# Capa 3: 64 → 10
W3 = np.random.randn(64, 10) * 0.01
b3 = np.zeros(10)
salida = a2 @ W3 + b3  # Shape: (32, 10)

print(f"\n{'Tensor':<15} {'Shape':<15} {'Descripción'}")
print("-" * 55)
print(f"{'X':<15} {str(X.shape):<15} {'32 imágenes, 784 píxeles cada una'}")
print(f"{'a1':<15} {str(a1.shape):<15} {'32 activaciones de 128 neuronas'}")
print(f"{'a2':<15} {str(a2.shape):<15} {'32 activaciones de 64 neuronas'}")
print(f"{'salida':<15} {str(salida.shape):<15} {'32 vectores de 10 scores de clase'}")

# Contar parámetros
params_c1 = 784 * 128 + 128
params_c2 = 128 * 64 + 64
params_c3 = 64 * 10 + 10
print(f"\n📊 Parámetros:")
print(f"   Capa 1: {params_c1:,}")
print(f"   Capa 2: {params_c2:,}")
print(f"   Capa 3: {params_c3:,}")
print(f"   TOTAL:  {params_c1 + params_c2 + params_c3:,}")
```

**Actividad 1.3**: Crea y traza la red `[10, 20, 15, 5]`. Verifica dimensiones paso a paso.

**Actividad 1.4**: Calcula manualmente el número de parámetros de `[784, 256, 128, 10]` y verifica con código.

**Actividad 1.5**: Experimenta con diferentes batch sizes (1, 8, 32, 64). ¿Cambia el número de parámetros?

**Actividad 1.6**: ¿Qué sucede si el batch size es 1? Verifica que la red funciona igual para una sola muestra.

### Preguntas de Reflexión

**Pregunta 1.1 (Concebir):** ¿Por qué conectamos capas en secuencia en lugar de conectar todas las neuronas directamente a la salida?

**Pregunta 1.2 (Diseñar):** Para un problema con 100 características de entrada y 5 clases de salida, ¿cómo diseñarías la arquitectura? ¿Qué factores considerarías?

**Pregunta 1.3 (Implementar):** ¿Por qué la shape de los pesos W entre dos capas debe ser `(n_capa_anterior, n_capa_siguiente)` y no al revés?

**Pregunta 1.4 (Operar):** Si la red tiene 32 millones de parámetros, ¿cuánta memoria RAM necesita solo para almacenar los pesos (en MB), asumiendo float32 (4 bytes por número)?

---

## 🔬 Parte 2: Programación Orientada a Objetos (40 min)

### 2.1 Introducción Conceptual: ¿Por qué usar clases?

**¿Qué hacemos?** Encapsular la lógica de capas y redes en clases reutilizables.

**¿Por qué lo hacemos?** El código procedimental (como en la Parte 1) se vuelve inmanejable para redes grandes. Las clases permiten:
- **Encapsulamiento**: cada capa maneja sus propios parámetros
- **Reutilización**: crear cualquier arquitectura con las mismas clases
- **Mantenibilidad**: modificar una capa sin afectar el resto
- **Extensibilidad**: agregar nuevas funcionalidades fácilmente

**Analogía:** Piensa en construir con bloques LEGO. Cada `CapaDensa` es un tipo de bloque estandarizado que puedes apilar en cualquier configuración, y la `RedNeuronal` es el conjunto ensamblado.

**¿Qué resultados esperar?** Clases que generalizan el proceso de forward propagation para cualquier arquitectura.

### 2.2 Clase CapaDensa

```python
import numpy as np

class CapaDensa:
    """
    Capa densa (fully connected) de neuronas artificiales.
    
    Una capa densa conecta cada neurona de la capa anterior
    con cada neurona de esta capa a través de pesos aprendibles.
    
    Args:
        n_entradas: Número de características de entrada
        n_neuronas: Número de neuronas en esta capa
        seed: Semilla aleatoria para reproducibilidad
    """
    
    def __init__(self, n_entradas, n_neuronas, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Validaciones
        assert n_entradas > 0, "n_entradas debe ser positivo"
        assert n_neuronas > 0, "n_neuronas debe ser positivo"
        
        self.n_entradas = n_entradas
        self.n_neuronas = n_neuronas
        
        # Inicialización de pesos: valores pequeños aleatorios
        # Multiplicamos por 0.01 para evitar saturación en activaciones
        self.pesos = np.random.randn(n_entradas, n_neuronas) * 0.01
        self.biases = np.zeros(n_neuronas)
        
        # Almacén para forward pass (útil para debugging y backprop)
        self.entradas = None
        self.salida = None
        
        print(f"✅ CapaDensa creada: {n_entradas} → {n_neuronas} "
              f"({self.contar_parametros():,} parámetros)")
    
    def forward(self, entradas):
        """
        Propagación hacia adelante.
        
        Calcula: salida = entradas @ pesos + biases
        
        Args:
            entradas: Array (batch_size, n_entradas)
        
        Returns:
            salida: Array (batch_size, n_neuronas)
        """
        # Validar dimensiones de entrada
        assert entradas.shape[1] == self.n_entradas, \
            f"Shape esperado: (batch, {self.n_entradas}), recibido: {entradas.shape}"
        
        self.entradas = entradas
        self.salida = np.dot(entradas, self.pesos) + self.biases
        return self.salida
    
    def contar_parametros(self):
        """Retorna el número total de parámetros aprendibles."""
        return self.pesos.size + self.biases.size
    
    def resumen(self):
        """Imprime información detallada de la capa."""
        print(f"\n📋 Capa Densa:")
        print(f"   Forma de pesos: {self.pesos.shape}")
        print(f"   Forma de biases: {self.biases.shape}")
        print(f"   Parámetros totales: {self.contar_parametros():,}")
        print(f"   Media de pesos: {self.pesos.mean():.6f}")
        print(f"   Std de pesos: {self.pesos.std():.6f}")
    
    def __repr__(self):
        return f"CapaDensa({self.n_entradas} → {self.n_neuronas})"


# Ejemplo de uso
capa = CapaDensa(784, 128, seed=42)
capa.resumen()

# Procesar un batch
X = np.random.randn(32, 784)
salida = capa.forward(X)
print(f"\n🔄 Forward pass: {X.shape} → {salida.shape}")
```

**Actividad 2.1**: Crea una capa con 10 entradas y 5 neuronas. ¿Cuántos parámetros tiene? Verifica con `contar_parametros()`.

**Actividad 2.2**: Modifica la clase para que `forward()` también imprima las estadísticas de salida (media, std). ¿Cambia esto la funcionalidad principal?

### 2.3 Clase RedNeuronal

```python
class RedNeuronal:
    """
    Red neuronal multicapa con arquitectura flexible.
    
    Implementa una red fully-connected donde el usuario
    especifica el número de neuronas en cada capa.
    
    Args:
        arquitectura: Lista con número de neuronas por capa
                      ej: [784, 128, 64, 10]
    """
    
    def __init__(self, arquitectura, seed=None):
        assert len(arquitectura) >= 2, "Necesitas al menos entrada y salida"
        
        self.arquitectura = arquitectura
        self.capas = []
        
        # Crear capas densas entre cada par de dimensiones adyacentes
        for i in range(len(arquitectura) - 1):
            n_in = arquitectura[i]
            n_out = arquitectura[i + 1]
            capa = CapaDensa(n_in, n_out, seed=seed)
            self.capas.append(capa)
        
        print(f"\n🏗️  Red Neuronal creada:")
        print(f"   Arquitectura: {arquitectura}")
        print(f"   Capas: {len(self.capas)}")
        print(f"   Total parámetros: {self.contar_parametros():,}")
    
    def forward(self, X):
        """
        Forward propagation a través de todas las capas.
        
        Procesa los datos secuencialmente capa por capa.
        
        Args:
            X: Datos de entrada (batch_size, n_entrada)
        
        Returns:
            activacion: Salida final (batch_size, n_salida)
        """
        activacion = X
        for capa in self.capas:
            activacion = capa.forward(activacion)
        return activacion
    
    def contar_parametros(self):
        """Cuenta todos los parámetros de la red."""
        return sum(capa.contar_parametros() for capa in self.capas)
    
    def resumen(self):
        """Imprime la arquitectura completa de la red."""
        print("\n" + "=" * 60)
        print("RESUMEN DE LA RED NEURONAL")
        print("=" * 60)
        total = 0
        for i, capa in enumerate(self.capas):
            params = capa.contar_parametros()
            total += params
            print(f"  Capa {i+1}: {capa.n_entradas:5d} → {capa.n_neuronas:5d} "
                  f"| {params:10,} parámetros")
        print("-" * 60)
        print(f"  TOTAL:                        {total:10,} parámetros")
        print("=" * 60)
    
    def analizar_activaciones(self, X):
        """Analiza estadísticas de activaciones por capa."""
        print("\n📊 Análisis de Activaciones:")
        activacion = X
        for i, capa in enumerate(self.capas):
            activacion = capa.forward(activacion)
            print(f"  Capa {i+1}: mean={activacion.mean():.4f}, "
                  f"std={activacion.std():.4f}, "
                  f"min={activacion.min():.4f}, "
                  f"max={activacion.max():.4f}")


# Ejemplo de uso
red = RedNeuronal([784, 128, 64, 10], seed=42)
red.resumen()

X = np.random.randn(32, 784)
salida = red.forward(X)
print(f"\n🔢 Salida final: {salida.shape}")

red.analizar_activaciones(X)
```

**Actividad 2.3**: Crea redes con las siguientes arquitecturas y compara su número de parámetros:
   - `[100, 50, 10]`
   - `[100, 200, 50, 10]`
   - `[100, 500, 10]`

**Actividad 2.4**: Implementa un método `get_activaciones_intermedias(X)` que retorne las activaciones de cada capa (no solo la final).

**Actividad 2.5**: Agrega un método `guardar_pesos(filepath)` y `cargar_pesos(filepath)` usando `np.save` y `np.load`.

**Actividad 2.6**: Crea una función `test_red(arquitectura)` que verifique que la red produce las shapes correctas con un batch de 16 muestras.

### Preguntas de Reflexión

**Pregunta 2.1 (Concebir):** ¿Cuál es la ventaja de definir la arquitectura como una lista `[784, 128, 64, 10]` en lugar de crear cada capa manualmente?

**Pregunta 2.2 (Diseñar):** En el método `forward()`, guardamos el estado intermedio de cada capa. ¿Por qué esto es importante para el entrenamiento (aunque aún no lo implementemos)?

**Pregunta 2.3 (Implementar):** ¿Por qué el loop `for capa in self.capas` en `forward()` es correcto para conectar capas en secuencia? Traza mentalmente el flujo de datos.

**Pregunta 2.4 (Operar):** Si necesitas procesar 1 millón de imágenes, ¿por qué es más eficiente procesar en batches de 64 que de una en una?

---

## 🔬 Parte 3: Inicialización y Sus Efectos (35 min)

### 3.1 Introducción Conceptual: ¿Por qué importa la inicialización?

**¿Qué hacemos?** Estudiar diferentes estrategias para inicializar los pesos de la red.

**¿Por qué lo hacemos?** La inicialización de pesos determina el punto de partida del entrenamiento. Una mala inicialización puede:
- Hacer que todas las neuronas aprendan lo mismo (**problema de simetría**)
- Causar que las señales se desvanezcan o exploten al propagarse (**gradientes inestables**)
- Ralentizar enormemente el entrenamiento o impedir la convergencia

**Analogía:** Si quieres explorar un laberinto, ¿prefieres empezar en el centro (buena inicialización) o pegado a una pared (mala inicialización)? El punto de partida afecta cuánto tardarás en encontrar la salida.

**¿Qué resultados esperar?** Distribuciones de activaciones diferentes según la estrategia de inicialización. La buena inicialización mantiene la varianza estable entre capas.

### 3.2 El Problema de los Ceros

```python
def demostrar_problema_simetria():
    """Demuestra por qué inicializar en cero es problemático."""
    
    print("=" * 60)
    print("PROBLEMA DE SIMETRÍA CON PESOS EN CERO")
    print("=" * 60)
    
    # Red con todos los pesos en cero
    X = np.random.randn(5, 3)
    W_cero = np.zeros((3, 4))
    b_cero = np.zeros(4)
    
    salida_cero = X @ W_cero + b_cero
    
    print("\n❌ Con pesos en cero:")
    print(f"   Todas las salidas son cero: {np.all(salida_cero == 0)}")
    print(f"   Salidas únicas: {np.unique(salida_cero)}")
    print(f"   → Ninguna neurona aprende características diferentes")
    
    # Red con pesos aleatorios pequeños
    W_rand = np.random.randn(3, 4) * 0.01
    salida_rand = X @ W_rand + b_cero
    
    print("\n✅ Con pesos aleatorios pequeños:")
    print(f"   Media: {salida_rand.mean():.6f}")
    print(f"   Std: {salida_rand.std():.6f}")
    print(f"   → Cada neurona produce valores distintos")

demostrar_problema_simetria()
```

### 3.3 Comparación de Estrategias de Inicialización

```python
def comparar_inicializaciones(n_entradas=100, n_neuronas=100, n_capas=5):
    """
    Compara cómo distintas inicializaciones afectan la varianza
    de activaciones en redes profundas.
    """
    import matplotlib.pyplot as plt
    
    X = np.random.randn(1000, n_entradas)
    
    estrategias = {
        'Muy pequeños (×0.001)': lambda n, m: np.random.randn(n, m) * 0.001,
        'Pequeños (×0.01)':      lambda n, m: np.random.randn(n, m) * 0.01,
        'Xavier/Glorot':         lambda n, m: np.random.randn(n, m) * np.sqrt(1.0/n),
        'He (para ReLU)':        lambda n, m: np.random.randn(n, m) * np.sqrt(2.0/n),
    }
    
    print("=" * 65)
    print("COMPARACIÓN DE ESTRATEGIAS DE INICIALIZACIÓN")
    print(f"Red: [{n_entradas}] × {n_capas} capas de {n_neuronas}")
    print("=" * 65)
    print(f"\n{'Estrategia':<25} | " + " | ".join([f"Capa{i+1:2d}" for i in range(n_capas)]))
    print("-" * 65)
    
    for nombre, init_fn in estrategias.items():
        activacion = X.copy()
        stds = []
        
        for _ in range(n_capas):
            W = init_fn(activacion.shape[1], n_neuronas)
            b = np.zeros(n_neuronas)
            activacion = activacion @ W + b
            stds.append(activacion.std())
        
        std_str = " | ".join([f"{s:6.4f}" for s in stds])
        print(f"{nombre:<25} | {std_str}")
    
    print("\n💡 Interpretación:")
    print("   - Muy pequeños: varianza se desvanece → neuronas inactivas")
    print("   - Muy grandes: varianza explota → gradientes inestables")
    print("   - Xavier: mantiene varianza estable para activaciones lineales/tanh")
    print("   - He: mantiene varianza estable para ReLU")

comparar_inicializaciones()
```

**Actividad 3.1**: Ejecuta `comparar_inicializaciones()` y anota cuál estrategia mantiene la varianza más estable entre capas.

**Actividad 3.2**: Modifica la función para probar con 10 capas en lugar de 5. ¿Qué le ocurre a la varianza con la inicialización muy pequeña?

**Actividad 3.3**: Implementa la inicialización **Glorot Uniforme**: `W = uniform(-√(6/(n+m)), √(6/(n+m)))`. Compara con Xavier Gaussiano.

**Actividad 3.4**: Verifica que dos redes con la misma `seed` producen exactamente las mismas salidas.

### Preguntas de Reflexión

**Pregunta 3.1 (Concebir):** ¿Por qué el "problema de simetría" impide que una red con pesos iguales aprenda características diversas?

**Pregunta 3.2 (Diseñar):** Si sabes que usarás ReLU como activación (próximo lab), ¿qué inicialización elegirías y por qué?

**Pregunta 3.3 (Implementar):** ¿Por qué multiplicamos los pesos por `sqrt(2/n)` en la inicialización He en lugar de simplemente usar `0.01`?

**Pregunta 3.4 (Operar):** En producción, ¿por qué es importante fijar una `seed` aleatoria antes de inicializar una red?

---

## 🔬 Parte 4: Diseño de Arquitecturas (35 min)

### 4.1 Introducción Conceptual: ¿Cómo diseñar una red?

**¿Qué hacemos?** Diseñar arquitecturas de redes neuronales apropiadas para diferentes tipos de problemas.

**¿Por qué lo hacemos?** No existe una arquitectura "perfecta" universal. El diseño depende de:
- Número y tipo de características de entrada
- Tipo de problema (clasificación binaria, multiclase, regresión)
- Cantidad de datos disponibles
- Restricciones de tiempo y memoria

**Reglas prácticas de diseño:**
1. El número de neuronas de entrada = número de características
2. El número de neuronas de salida depende del problema
3. Las capas ocultas generalmente se reducen gradualmente hacia la salida
4. Más capas = más capacidad, pero también más difícil de entrenar

**¿Qué resultados esperar?** Arquitecturas funcionales con conteo de parámetros verificado.

### 4.2 Arquitecturas para Diferentes Problemas

```python
# Problema 1: Clasificación Binaria (spam/no-spam)
# Entrada: 5000 features (bag of words)
# Salida: 1 probabilidad (spam o no)
red_spam = RedNeuronal([5000, 256, 64, 1])
red_spam.resumen()

# Problema 2: Clasificación Multiclase (MNIST: 10 dígitos)
# Entrada: 784 píxeles
# Salida: 10 scores de clase
red_mnist = RedNeuronal([784, 512, 256, 128, 10])
red_mnist.resumen()

# Problema 3: Regresión (predicción de precios)
# Entrada: 20 características de la casa
# Salida: 1 valor continuo (precio)
red_precios = RedNeuronal([20, 64, 32, 16, 1])
red_precios.resumen()

# Problema 4: Clasificación de emociones (5 clases)
# Entrada: 1000 features de audio
# Salida: 5 probabilidades de emoción
red_emociones = RedNeuronal([1000, 256, 128, 64, 5])
red_emociones.resumen()
```

**Actividad 4.1**: Diseña una arquitectura para clasificar 50 tipos de flores con 30 características cada una. Justifica tu elección.

### 4.3 Redes Profundas vs. Anchas

```python
def comparar_profunda_vs_ancha():
    """
    Compara el número de parámetros de redes profundas vs anchas
    con similar capacidad.
    """
    print("=" * 60)
    print("REDES PROFUNDAS vs ANCHAS")
    print("=" * 60)
    
    arquitecturas = {
        "Muy profunda":  [100, 80, 60, 40, 20, 10, 5],
        "Profunda":      [100, 64, 32, 16, 5],
        "Equilibrada":   [100, 200, 100, 5],
        "Ancha":         [100, 500, 5],
        "Muy ancha":     [100, 1000, 5],
    }
    
    print(f"\n{'Nombre':<15} {'Arquitectura':<35} {'Parámetros':>12}")
    print("-" * 65)
    
    for nombre, arq in arquitecturas.items():
        red = RedNeuronal(arq)
        params = red.contar_parametros()
        arq_str = " → ".join(map(str, arq))
        print(f"{nombre:<15} {arq_str:<35} {params:>12,}")

comparar_profunda_vs_ancha()
```

**Actividad 4.2**: Diseña dos redes con aproximadamente el mismo número de parámetros (~50,000) pero arquitecturas muy distintas (una profunda, una ancha). Compara sus tiempos de forward pass.

### 4.4 La Limitación Matemática Sin Activación

Esta es una de las demostraciones más importantes del laboratorio:

```python
def demostrar_colapso_lineal():
    """
    Demuestra matemáticamente que una red sin activaciones
    no lineales se reduce a una sola transformación lineal.
    """
    print("=" * 60)
    print("DEMOSTRACIÓN: RED PROFUNDA = RED DE 1 CAPA (sin activación)")
    print("=" * 60)
    
    np.random.seed(42)
    X = np.random.randn(5, 3)
    
    # Red de 2 capas sin activación
    W1 = np.random.randn(3, 4) * 0.1
    b1 = np.random.randn(4) * 0.1
    W2 = np.random.randn(4, 2) * 0.1
    b2 = np.random.randn(2) * 0.1
    
    # Forward pass con 2 capas
    h1 = X @ W1 + b1          # Capa 1
    salida_2capas = h1 @ W2 + b2  # Capa 2
    
    # Equivalente matemático (1 sola transformación lineal):
    # h1 @ W2 + b2
    # = (X @ W1 + b1) @ W2 + b2
    # = X @ W1 @ W2 + b1 @ W2 + b2
    W_equivalente = W1 @ W2
    b_equivalente = b1 @ W2 + b2
    salida_1capa = X @ W_equivalente + b_equivalente
    
    print(f"\n📊 Red de 2 capas (W1={W1.shape}, W2={W2.shape}):")
    print(f"   Parámetros: {W1.size + b1.size + W2.size + b2.size}")
    
    print(f"\n📊 Equivalente de 1 capa (W={W_equivalente.shape}):")
    print(f"   Parámetros: {W_equivalente.size + b_equivalente.size}")
    
    print(f"\n✅ ¿Son idénticas las salidas?")
    son_iguales = np.allclose(salida_2capas, salida_1capa)
    print(f"   np.allclose(salida_2capas, salida_1capa) = {son_iguales}")
    
    print("\n⚠️  CONCLUSIÓN FUNDAMENTAL:")
    print("   Sin activación no lineal, una red profunda es IDÉNTICA")
    print("   a una red de 1 sola capa. Las capas adicionales no")
    print("   aportan capacidad representacional adicional.")
    print("   → Por esto necesitamos funciones de activación (Lab 03)!")

demostrar_colapso_lineal()
```

**Actividad 4.3**: Extiende la demostración a 3 capas lineales. ¿Puedes reducirlas a 1 capa equivalente?

**Actividad 4.4**: Diseña una red para el problema XOR y verifica (con la demostración) que sin activación no puede resolverlo.

**Actividad 4.5**: Investiga el **Teorema de Aproximación Universal**: ¿cuántas neuronas ocultas necesita una red de 1 capa para aproximar cualquier función continua?

### Preguntas de Reflexión

**Pregunta 4.1 (Concebir):** Si las redes sin activación son equivalentes a transformaciones lineales, ¿cuál es la utilidad de estudiarlas en este laboratorio?

**Pregunta 4.2 (Diseñar):** ¿Cuándo preferirías una red ancha sobre una profunda? ¿Hay ventajas computacionales?

**Pregunta 4.3 (Implementar):** En la demostración del colapso lineal, ¿por qué la red de 2 capas tiene MÁS parámetros que la equivalente de 1 capa pero hace lo mismo?

**Pregunta 4.4 (Operar):** En producción, si entrenas una red sin activaciones y observas que no mejora, ¿cómo diagnosticarías si el problema es la falta de no-linealidad?

---

## 🔬 Parte 5: Aplicaciones Prácticas y Visualización (35 min)

### 5.1 Introducción Conceptual: Visualizando el Flujo de Datos

**¿Qué hacemos?** Analizar cómo los datos se transforman al pasar por cada capa de la red.

**¿Por qué lo hacemos?** Entender las transformaciones intermedias es fundamental para:
- Diagnosticar problemas en el entrenamiento
- Verificar que la red está aprendiendo representaciones útiles
- Detectar problemas de inicialización (activaciones en cero o saturadas)
- Interpretar qué "aprende" cada capa

**¿Qué resultados esperar?** Gráficas que muestran cómo cambia la distribución de activaciones capa por capa.

### 5.2 Visualización de Activaciones

```python
import matplotlib.pyplot as plt

def visualizar_transformaciones(red, X, titulo="Transformaciones por Capa"):
    """
    Visualiza cómo se transforman los datos en cada capa de la red.
    
    Args:
        red: Instancia de RedNeuronal
        X: Datos de entrada (batch_size, n_entradas)
        titulo: Título del gráfico
    """
    n_capas = len(red.capas)
    fig, axes = plt.subplots(1, n_capas + 1, figsize=(4 * (n_capas + 1), 4))
    
    # Graficar entrada
    axes[0].hist(X.ravel(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_title("Entrada\n" + f"shape={X.shape}", fontsize=10)
    axes[0].set_xlabel("Valor de activación")
    axes[0].set_ylabel("Frecuencia")
    axes[0].grid(True, alpha=0.3)
    
    # Graficar activaciones por capa
    activacion = X
    for i, capa in enumerate(red.capas):
        activacion = capa.forward(activacion)
        axes[i+1].hist(activacion.ravel(), bins=50, edgecolor='black', 
                       alpha=0.7, color='darkorange')
        axes[i+1].set_title(
            f"Capa {i+1}\nshape={activacion.shape}\n"
            f"μ={activacion.mean():.3f}, σ={activacion.std():.3f}",
            fontsize=9
        )
        axes[i+1].set_xlabel("Valor de activación")
        axes[i+1].grid(True, alpha=0.3)
    
    plt.suptitle(titulo, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('transformaciones_capas.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico guardado como 'transformaciones_capas.png'")


# Ejemplo
np.random.seed(42)
red = RedNeuronal([784, 256, 128, 64, 10])
X = np.random.randn(500, 784)
visualizar_transformaciones(red, X)
```

### 5.3 Dataset Sintético y Análisis

```python
def experimento_datos_sinteticos():
    """
    Genera datos sintéticos y analiza el comportamiento de la red.
    """
    def generar_datos(n=1000, features=20, clases=5, seed=42):
        np.random.seed(seed)
        X = np.random.randn(n, features)
        y = np.random.randint(0, clases, n)
        return X, y
    
    X, y = generar_datos()
    red = RedNeuronal([20, 64, 32, 5])
    
    predicciones = red.forward(X)
    
    print("=" * 55)
    print("EXPERIMENTO CON DATOS SINTÉTICOS")
    print("=" * 55)
    print(f"Datos: {X.shape} | Clases: {np.unique(y)}")
    print(f"Predicciones (sin entrenar): {predicciones.shape}")
    print(f"\nEstadísticas de predicciones:")
    for clase in range(5):
        print(f"  Clase {clase}: media={predicciones[:, clase].mean():.4f}")
    
    # Análisis de batch processing
    import time
    print("\n⏱️  Análisis de batch processing:")
    for batch_size in [1, 10, 50, 100, 500, 1000]:
        start = time.time()
        for _ in range(100):  # 100 pasadas
            _ = red.forward(X[:batch_size])
        elapsed = (time.time() - start) / 100
        throughput = batch_size / elapsed
        print(f"  Batch {batch_size:5d}: {elapsed*1000:7.3f} ms/pasada "
              f"| {throughput:,.0f} muestras/seg")

experimento_datos_sinteticos()
```

**Actividad 5.1**: Ejecuta `visualizar_transformaciones()` con diferentes arquitecturas. ¿Cambia la distribución de activaciones?

**Actividad 5.2**: Modifica la función de visualización para mostrar un heatmap de la matriz de pesos de cada capa.

**Actividad 5.3**: Crea un experimento que compare el tiempo de forward pass de una red ancha vs una profunda con el mismo número de parámetros.

**Actividad 5.4**: Implementa una función que detecte si alguna capa tiene activaciones con varianza cercana a cero (posible problema de inicialización).

### Preguntas de Reflexión

**Pregunta 5.1 (Concebir):** ¿Qué información te proporciona la distribución de activaciones de una capa?

**Pregunta 5.2 (Diseñar):** Si todas las activaciones de una capa son prácticamente cero, ¿qué problema podría estar ocurriendo y cómo lo corregirías?

**Pregunta 5.3 (Implementar):** ¿Por qué es importante analizar las activaciones ANTES de entrenar la red?

**Pregunta 5.4 (Operar):** En un sistema en producción, ¿qué métricas monitorearías durante el forward pass para detectar problemas?

---

## 📊 Análisis Final de Rendimiento

### Benchmark de Implementaciones

En esta sección medirás el rendimiento de diferentes enfoques de implementación para entender las ventajas de la vectorización con NumPy.

**Fundamento:** La multiplicación matricial vectorizada de NumPy aprovecha bibliotecas BLAS/LAPACK optimizadas en C/Fortran y puede usar instrucciones SIMD del procesador, siendo órdenes de magnitud más rápida que loops en Python puro.

```python
import time
import numpy as np

def benchmark_implementaciones():
    """
    Compara el rendimiento de diferentes implementaciones
    de forward pass.
    """
    print("\n" + "=" * 65)
    print("BENCHMARK: COMPARACIÓN DE IMPLEMENTACIONES")
    print("=" * 65)
    
    configuraciones = [
        (100, 50, 500),
        (784, 128, 1000),
        (1000, 256, 2000),
    ]
    
    for n_entrada, n_neuronas, batch_size in configuraciones:
        print(f"\n📐 Config: {n_entrada}→{n_neuronas}, batch={batch_size}")
        
        X = np.random.randn(batch_size, n_entrada)
        W = np.random.randn(n_entrada, n_neuronas) * 0.01
        b = np.zeros(n_neuronas)
        
        # Método 1: NumPy @ operator (vectorizado)
        start = time.perf_counter()
        for _ in range(100):
            Y1 = X @ W + b
        t_numpy = (time.perf_counter() - start) / 100
        
        # Método 2: np.dot (vectorizado)
        start = time.perf_counter()
        for _ in range(100):
            Y2 = np.dot(X, W) + b
        t_dot = (time.perf_counter() - start) / 100
        
        # Método 3: Loop por muestras (no vectorizado)
        if batch_size <= 500:
            start = time.perf_counter()
            for _ in range(10):
                Y3 = np.array([np.dot(X[i], W) + b for i in range(batch_size)])
            t_loop = (time.perf_counter() - start) / 10
        else:
            Y3, t_loop = Y1, None
        
        print(f"   @ operator:    {t_numpy*1000:.4f} ms")
        print(f"   np.dot:        {t_dot*1000:.4f} ms")
        if t_loop:
            print(f"   Loop Python:   {t_loop*1000:.4f} ms")
            print(f"   🚀 Aceleración: {t_loop/t_numpy:.1f}x más rápido con vectorización")
        
        assert np.allclose(Y1, Y2), "¡Los resultados no coinciden!"

benchmark_implementaciones()
```

### Análisis de Escalabilidad

```python
def analizar_escalabilidad_red():
    """
    Analiza cómo escala el costo computacional con el tamaño de la red.
    """
    import time
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 55)
    print("ANÁLISIS DE ESCALABILIDAD")
    print("=" * 55)
    
    batch_size = 128
    n_neuronas_lista = [16, 32, 64, 128, 256, 512, 1024]
    tiempos = []
    
    print(f"\n{'N. Neuronas':<15} {'Tiempo(ms)':<15} {'Parámetros'}")
    print("-" * 45)
    
    for n in n_neuronas_lista:
        red = RedNeuronal([n, n, n, n])  # 3 capas de n neuronas
        X = np.random.randn(batch_size, n)
        
        # Medir tiempo
        start = time.perf_counter()
        for _ in range(200):
            _ = red.forward(X)
        elapsed = (time.perf_counter() - start) / 200
        tiempos.append(elapsed)
        
        print(f"{n:<15} {elapsed*1000:<15.3f} {red.contar_parametros():,}")
    
    # Visualizar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(n_neuronas_lista, np.array(tiempos)*1000, 'o-', 
             linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Número de neuronas por capa')
    ax1.set_ylabel('Tiempo por forward pass (ms)')
    ax1.set_title('Tiempo de Forward Pass vs. Tamaño de Capa')
    ax1.grid(True, alpha=0.3)
    
    ax2.loglog(n_neuronas_lista, np.array(tiempos)*1000, 'o-',
               linewidth=2, markersize=8, color='darkorange')
    ax2.set_xlabel('Número de neuronas por capa (escala log)')
    ax2.set_ylabel('Tiempo (ms, escala log)')
    ax2.set_title('Escalabilidad (log-log)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('escalabilidad_red.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("\n💡 Complejidad: O(N²) por capa (N = número de neuronas)")
    print("   El costo crece cuadráticamente con el tamaño de la red")

analizar_escalabilidad_red()
```

---

## 🎯 EJERCICIOS PROPUESTOS

### Ejercicio 1: Seguimiento de Dimensiones (Básico)

**Objetivo:** Consolidar el entendimiento de cómo fluyen los datos.

**Tareas:**
1. Para la red `[5, 8, 6, 3]`, traza manualmente las shapes de cada tensor
2. Verifica con código que las shapes son correctas
3. Calcula el número total de parámetros a mano y verifica con `contar_parametros()`

```python
# Esqueleto de solución
def analizar_dimensiones_red(arquitectura, batch_size=4):
    """
    Traza las dimensiones de todos los tensores en la red.
    
    Args:
        arquitectura: Lista con neuronas por capa ej: [5, 8, 6, 3]
        batch_size: Número de muestras en el batch
    """
    print(f"Red: {arquitectura}, Batch size: {batch_size}")
    print("-" * 50)
    
    # Tu código aquí: crear la red y analizar shapes
    # Pista: usa red.capas para acceder a las capas
    pass

analizar_dimensiones_red([5, 8, 6, 3], batch_size=4)
```

### Ejercicio 2: Diagnóstico de Red (Intermedio)

**Objetivo:** Implementar herramientas de diagnóstico que analicen el estado de una red.

**Tareas:**
1. Implementa `diagnostico_red(red, X)` que detecte:
   - Capas con activaciones en cero o casi cero (std < 0.001)
   - Capas con activaciones explosivas (std > 100)
   - Porcentaje de activaciones negativas
2. Prueba con redes con diferentes inicializaciones

```python
def diagnostico_red(red, X):
    """
    Analiza la salud de la red detectando problemas comunes.
    
    Returns:
        dict: Diccionario con estadísticas por capa y alertas
    """
    # Tu código aquí
    pass

# Prueba con diferentes inicializaciones
# ¿Qué problemas detecta el diagnóstico?
```

### Ejercicio 3: Visualización de Arquitectura (Intermedio)

**Objetivo:** Crear una visualización gráfica de la arquitectura de la red.

**Tareas:**
1. Implementa `graficar_arquitectura(arquitectura)` que dibuje:
   - Cada capa como una columna de círculos
   - Las conexiones entre capas (representativas, no todas)
   - El número de parámetros por capa
2. Usa matplotlib para la visualización

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def graficar_arquitectura(arquitectura, max_neuronas_mostradas=8):
    """
    Visualiza la arquitectura de una red neuronal.
    
    Args:
        arquitectura: Lista con número de neuronas por capa
        max_neuronas_mostradas: Máximo de neuronas a dibujar por capa
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Tu código aquí
    # Pista: usa ax.add_patch(patches.Circle(...)) para las neuronas
    # y ax.plot(...) para las conexiones
    pass

graficar_arquitectura([784, 128, 64, 10])
```

### Ejercicio 4: Comparación de Inicializaciones en Profundidad (Avanzado)

**Objetivo:** Estudiar empíricamente el efecto de la inicialización en redes muy profundas.

**Tareas:**
1. Crea una red de 20 capas (red muy profunda)
2. Inicializa con 4 estrategias: zeros, tiny (×0.001), Xavier, He
3. Para cada inicialización, grafica la varianza de activaciones por capa
4. Identifica qué estrategias causan **vanishing gradients** y cuáles causan **exploding gradients**

```python
def estudio_profundidad_inicializacion(n_capas=20, n_neuronas=100):
    """
    Estudia el efecto de la inicialización en redes profundas.
    Grafica la varianza de activaciones por capa para cada estrategia.
    """
    import matplotlib.pyplot as plt
    
    estrategias = {
        'Zeros':   lambda n, m: np.zeros((n, m)),
        'Tiny':    lambda n, m: np.random.randn(n, m) * 0.001,
        'Xavier':  lambda n, m: np.random.randn(n, m) * np.sqrt(1.0/n),
        'He':      lambda n, m: np.random.randn(n, m) * np.sqrt(2.0/n),
    }
    
    # Tu código aquí
    pass

estudio_profundidad_inicializacion()
```

### Ejercicio 5: Mini Framework de Redes Neuronales (Proyecto)

**Objetivo:** Construir un mini-framework extensible para redes neuronales.

**Tareas:**
1. Crea una clase base `Capa` con métodos abstractos `forward()` y `contar_parametros()`
2. Implementa `CapaDensa` y `CapaActivacion` (con función identidad por ahora)
3. Implementa `Secuencial` que agrupe capas con `add()`, `forward()`, `resumen()`
4. Agrega soporte para guardar y cargar el modelo completo en un archivo JSON/numpy

```python
class Capa:
    """Clase base abstracta para todas las capas."""
    def forward(self, X):
        raise NotImplementedError
    
    def contar_parametros(self):
        raise NotImplementedError


class Secuencial:
    """
    Contenedor de capas que permite construir redes secuencialmente.
    Uso:
        modelo = Secuencial()
        modelo.add(CapaDensa(784, 128))
        modelo.add(CapaDensa(128, 10))
        salida = modelo.forward(X)
    """
    def __init__(self):
        self.capas = []
    
    def add(self, capa):
        # Tu código aquí
        pass
    
    def forward(self, X):
        # Tu código aquí
        pass
    
    def resumen(self):
        # Tu código aquí
        pass
    
    def guardar(self, filepath):
        # Tu código aquí (usar np.savez)
        pass
    
    def cargar(self, filepath):
        # Tu código aquí
        pass
```

---

## 📝 Entregables

### 1. Código Implementado (60%)

**Requisitos mínimos:**
- Clase `CapaDensa` completa con docstrings, validaciones y método `resumen()`
- Clase `RedNeuronal` con `forward()`, `resumen()`, `contar_parametros()`, y `analizar_activaciones()`
- Al menos 2 ejercicios propuestos implementados y documentados
- Tests que verifiquen shapes correctas y reproducibilidad con seed

**Criterios de calidad:**
- Código limpio, PEP8, con comentarios explicativos
- Manejo apropiado de errores (`assert`, mensajes descriptivos)
- Funciones con docstrings completos (Args, Returns)

### 2. Notebook de Experimentación (25%)

**Debe incluir:**
- Todas las actividades de las partes 1-5 completadas y ejecutadas
- Visualizaciones claras (histogramas de activaciones, comparación de arquitecturas)
- Análisis comentado de los resultados de cada actividad
- Respuestas escritas a todas las Preguntas de Reflexión
- Experimentos adicionales creativos (mínimo 2)

### 3. Reporte Técnico (15%)

**Secciones requeridas:**
1. Introducción: objetivo del laboratorio y contexto
2. Marco teórico: conceptos clave (forward propagation, inicialización, parámetros)
3. Metodología: qué experimentos realizaste y cómo
4. Resultados: tablas y gráficas de experimentos
5. Análisis y discusión: interpretación de resultados
6. Conclusiones: aprendizajes clave y limitaciones encontradas

**Extensión:** 3-5 páginas, formato PDF

### Formato de Entrega

```
Lab02_Entrega_NombreApellido/
├── codigo/
│   ├── red_neuronal.py     # Clases principales
│   ├── utils.py            # Funciones auxiliares
│   └── tests.py            # Tests unitarios
├── notebooks/
│   └── experimentos.ipynb
├── reporte/
│   └── reporte_lab02.pdf
└── README.md               # Instrucciones de ejecución
```

---

## 🎯 Criterios de Evaluación (CDIO)

### Concebir (25%)

**Comprensión conceptual:**
- ✅ Explica por qué se necesitan múltiples capas para problemas complejos
- ✅ Comprende el Teorema de Aproximación Universal
- ✅ Identifica cuándo una red sin activación es insuficiente
- ✅ Propone arquitecturas adecuadas para problemas dados

**Evidencia:** Respuestas a preguntas de reflexión, introducción del reporte

### Diseñar (25%)

**Planificación de soluciones:**
- ✅ Diseña arquitecturas apropiadas para diferentes problemas
- ✅ Justifica elecciones de número de capas y neuronas
- ✅ Planifica experimentos significativos con hipótesis claras
- ✅ Considera trade-offs profundidad vs. anchura

**Evidencia:** Ejercicios 1-4, sección de metodología del reporte

### Implementar (30%)

**Construcción:**
- ✅ Clases `CapaDensa` y `RedNeuronal` funcionales y correctas
- ✅ Forward propagation implementado eficientemente
- ✅ Código limpio, documentado, con manejo de errores
- ✅ Tests unitarios que verifican comportamiento correcto

**Evidencia:** Código fuente, notebook ejecutado sin errores

### Operar (20%)

**Validación y análisis:**
- ✅ Ejecuta experimentos de benchmarking y escalabilidad
- ✅ Analiza e interpreta distribuciones de activaciones
- ✅ Diagnostica problemas de inicialización
- ✅ Extrae conclusiones fundamentadas de los experimentos

**Evidencia:** Notebook de experimentos, sección de resultados del reporte

### Rúbrica Detallada

| Criterio | Excelente (90-100%) | Bueno (75-89%) | Satisfactorio (60-74%) | Insuficiente (<60%) |
|----------|-------------------|----------------|----------------------|-------------------|
| **Implementación** | Código impecable, eficiente, bien documentado, con tests | Código funcional con documentación básica | Código funcional con errores menores | Código con errores o incompleto |
| **Experimentación** | Análisis profundo y creativo, hipótesis y conclusiones | Experimentos completos requeridos | Experimentos básicos | Experimentos incompletos |
| **Comprensión teórica** | Dominio total, conexiones con otros conceptos | Buen entendimiento, aplica correctamente | Comprensión básica | Comprensión limitada o incorrecta |
| **Documentación** | Excelente: clara, profesional, completa | Buena: completa y entendible | Básica: presente pero incompleta | Pobre o ausente |

---

## 📚 Referencias Adicionales

### Libros

1. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*
   - Capítulo 6: Deep Feedforward Networks (arquitecturas multicapa)
   - Disponible gratuitamente en: http://www.deeplearningbook.org

2. **Nielsen, M.** (2015). *Neural Networks and Deep Learning*
   - Capítulo 1: Using neural nets to recognize handwritten digits
   - Disponible en: http://neuralnetworksanddeeplearning.com

3. **Chollet, F.** (2021). *Deep Learning with Python* (2nd ed.)
   - Capítulo 2-3: Fundamentos de redes neuronales
   - Manning Publications

### Artículos Académicos

1. **Cybenko, G.** (1989). "Approximation by superpositions of a sigmoidal function"
   - Prueba original del Teorema de Aproximación Universal
   - *Mathematics of Control, Signals and Systems*, 2(4), 303-314

2. **Glorot, X., & Bengio, Y.** (2010). "Understanding the difficulty of training deep feedforward neural networks"
   - Introduce la inicialización Xavier/Glorot
   - *Proceedings of AISTATS*, 249-256

3. **He, K., Zhang, X., Ren, S., & Sun, J.** (2015). "Delving deep into rectifiers"
   - Introduce la inicialización He para ReLU
   - *Proceedings of ICCV*

### Recursos Online

1. **3Blue1Brown — "Neural Networks" series**
   - Visualizaciones excepcionales de forward propagation
   - https://www.youtube.com/watch?v=aircAruvnKk

2. **Stanford CS231n — Neural Networks Part 1**
   - Notas completas sobre arquitecturas y forward pass
   - https://cs231n.github.io/neural-networks-1/

3. **Deep Learning Book — Chapter 6**
   - Formulación matemática rigurosa
   - https://www.deeplearningbook.org/contents/mlp.html

### Tutoriales Interactivos

1. **TensorFlow Playground**
   - Experimenta con arquitecturas en el navegador
   - https://playground.tensorflow.org

2. **Neural Network Visualizer**
   - Visualización interactiva de forward pass
   - https://adamharley.com/nn_vis/

### Documentación Técnica

- **NumPy**: https://numpy.org/doc/ — Referencia completa de operaciones matriciales
- **Matplotlib**: https://matplotlib.org/ — Guía de visualizaciones
- **Python**: https://docs.python.org/3/ — Programación orientada a objetos

---

## 🎓 Notas Finales

### Conceptos Clave para Recordar

1. **Red Neuronal = Capas de Neuronas Conectadas en Secuencia**
   - Cada capa transforma la representación de los datos
   - Las capas profundas aprenden características más abstractas

2. **Forward Propagation: $a^{(l)} = f(a^{(l-1)} \cdot W^{(l)} + b^{(l)})$**
   - Cálculo secuencial desde entrada hasta salida
   - La salida de cada capa es la entrada de la siguiente

3. **Dimensiones: $(N, m) \cdot (m, k) = (N, k)$**
   - La dimensión compartida entre entrada y pesos DEBE coincidir
   - Siempre verifica shapes antes de entrenar

4. **Parámetros: $(n_{in} \times n_{out}) + n_{out}$ por capa**
   - Redes más grandes tienen mayor capacidad representacional
   - Pero también requieren más datos para entrenar

5. **Inicialización: NUNCA todo ceros**
   - Usar Xavier para activaciones tanh/lineal
   - Usar He para activaciones ReLU
   - Siempre usar `seed` para reproducibilidad

6. **Limitación fundamental: Sin activación, red profunda = red lineal**
   - La no-linealidad es imprescindible
   - Lab 03 introduce las funciones de activación

7. **Diseño: Balance profundidad vs. anchura**
   - Profunda: mejor generalización, más difícil de entrenar
   - Ancha: más parámetros, puede ser suficiente para algunos problemas

8. **Eficiencia: Batch processing y vectorización NumPy**
   - Procesar en batches es órdenes de magnitud más eficiente
   - Nunca usar loops de Python para operaciones matriciales

### Preparación para el Siguiente Lab

**Lab 03: Funciones de Activación** introducirá la no-linealidad que hace que las redes profundas sean verdaderamente poderosas.

Aprenderás:
- **ReLU**: `max(0, x)` — el estándar para capas ocultas
- **Sigmoid**: `1/(1+e^(-x))` — para clasificación binaria
- **Tanh**: `tanh(x)` — activación centrada en cero
- **Softmax**: normalización para clasificación multiclase
- Derivadas de cada función (necesarias para backpropagation)

**Para prepararte:**
1. Repasa cálculo diferencial: derivadas de funciones compuestas
2. Practica graficando funciones matemáticas con Matplotlib
3. Investiga qué es el "problema del gradiente que desaparece"
4. Reflexiona: ¿qué pasaría si todo `max(0,x)` hace al gradiente ser 0 o 1?

### Consejos de Estudio

1. **Implementa desde cero**: No uses TensorFlow/PyTorch en este lab
2. **Verifica siempre shapes**: `print(tensor.shape)` antes de cada operación
3. **Visualiza constantemente**: Histogramas y heatmaps revelan mucho
4. **Experimenta**: Cambia arquitecturas, seeds, batch sizes
5. **Documenta hallazgos**: Toma notas de qué funcionó y qué no
6. **Debug paso a paso**: Verifica intermedios antes de continuar
7. **Compara implementaciones**: Asegúrate que vectorizada y loop dan lo mismo

### Solución de Problemas Comunes

**Problema: `ValueError: matmul: Input operand 1 has a mismatch in its core dimension`**
- **Causa**: Shapes incompatibles en multiplicación matricial
- **Diagnóstico**: `print(X.shape, W.shape)` antes de la operación
- **Solución**: Verificar que el número de columnas de X = filas de W

**Problema: Todas las activaciones son cero**
- **Causa**: Inicialización con ceros o valores muy pequeños
- **Diagnóstico**: Verificar std de pesos con `capa.pesos.std()`
- **Solución**: Usar inicialización aleatoria (Xavier o He)

**Problema: Activaciones crecen exponencialmente por capa**
- **Causa**: Pesos inicializados con valores muy grandes
- **Diagnóstico**: `red.analizar_activaciones(X)` — ver std por capa
- **Solución**: Reducir escala de inicialización o usar Xavier/He

**Problema: Código muy lento al procesar datos grandes**
- **Causa**: Loops de Python en lugar de vectorización NumPy
- **Diagnóstico**: Usar `time.perf_counter()` para medir operaciones
- **Solución**: Reemplazar loops con operaciones matriciales `@` o `np.dot`

**Problema: Resultados no reproducibles**
- **Causa**: Falta de semilla aleatoria fija
- **Solución**: `np.random.seed(42)` al inicio del script o `seed` en constructores

### Comunidad y Soporte

- **Foro del curso**: Para dudas conceptuales y técnicas
- **Horas de oficina**: Para revisión personalizada de código
- **Grupo de estudio**: Trabaja los ejercicios propuestos con compañeros
- **Stack Overflow**: Para errores específicos de Python/NumPy

### Certificación de Completitud

Has completado exitosamente el Lab 02 cuando puedas:

- [ ] Explicar qué es forward propagation y cómo fluyen los datos
- [ ] Implementar `CapaDensa` y `RedNeuronal` desde cero sin consultar el material
- [ ] Calcular el número de parámetros de cualquier arquitectura
- [ ] Rastrear correctamente las shapes de tensores en una red
- [ ] Demostrar matemáticamente el colapso lineal sin activaciones
- [ ] Diseñar arquitecturas apropiadas para clasificación y regresión
- [ ] Comparar estrategias de inicialización y justificar cuál usar
- [ ] Interpretar histogramas de activaciones para diagnosticar problemas
- [ ] Medir y comparar el rendimiento de diferentes implementaciones

---

**¡Felicitaciones por completar el Lab 02!** Ahora tienes los fundamentos para construir cualquier arquitectura de red neuronal feedforward.

**Siguiente parada**: Lab 03 — Funciones de Activación 🚀

---

*Versión: 2.0 | Actualizado: 2024 | Licencia: MIT — Uso educativo*
