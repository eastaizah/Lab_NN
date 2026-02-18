# Guía de Laboratorio: Introducción a las Neuronas Artificiales
## 📋 Información del Laboratorio
**Título:** Fundamentos de Deep Learning - Neuronas Artificiales  
**Código:** Lab 01  
**Duración:** 2-3 horas  
**Nivel:** Básico  
## 🎯 Objetivos Específicos
Al completar este laboratorio, serás capaz de:
1. Comprender la estructura y funcionamiento de una neurona artificial
2. Identificar los componentes básicos: entradas, pesos, bias y salida
3. Implementar neuronas simples desde cero usando Python puro
4. Utilizar NumPy para cálculos vectorizados eficientes
5. Aplicar el producto punto (dot product) como operación fundamental
6. Crear capas de múltiples neuronas trabajando en conjunto
7. Procesar datos en batch (múltiples muestras simultáneamente)
8. Reconocer las limitaciones de neuronas individuales
9. Establecer las bases para construir redes neuronales completas
## 📚 Prerrequisitos
### Conocimientos
- Python básico-intermedio (variables, funciones, listas)
- Álgebra lineal básica (vectores, matrices, multiplicación)
- Conceptos matemáticos elementales (suma, producto)
- Manejo básico de arrays y listas
### Software
- Python 3.8+
- NumPy 1.19+
- Matplotlib (para visualizaciones opcionales)
- Jupyter Notebook (recomendado)
### Material de Lectura
Antes de comenzar, lee:
- `teoria.md` - Marco teórico completo sobre neuronas artificiales
- `README.md` - Estructura del laboratorio y recursos disponibles
## 📖 Introducción
Las **neuronas artificiales** son los bloques fundamentales que constituyen las redes neuronales modernas. Inspiradas en las neuronas biológicas del cerebro humano, estas unidades computacionales simples pueden, cuando se combinan en gran cantidad, resolver problemas complejos de reconocimiento de patrones, clasificación y predicción.
### Contexto del Problema
En el aprendizaje automático tradicional, los ingenieros deben diseñar manualmente las características (features) que el modelo utilizará. Este proceso, conocido como "feature engineering", requiere:
- Profundo conocimiento del dominio del problema
- Experimentación extensa para encontrar las características correctas
- Ajuste manual para diferentes tipos de datos
- Mucho tiempo y esfuerzo humano
Las redes neuronales ofrecen una alternativa revolucionaria: **aprendizaje automático de representaciones**. En lugar de diseñar características manualmente, la red aprende automáticamente qué patrones son importantes a partir de los datos.
### Enfoque con Neuronas Artificiales
Una neurona artificial es una unidad computacional que:
```
ENTRADAS (x₁, x₂, ..., xₙ)
    ↓
[PESOS (w₁, w₂, ..., wₙ) + BIAS (b)]
    ↓
SUMA PONDERADA: z = Σ(xᵢ × wᵢ) + b
    ↓
FUNCIÓN DE ACTIVACIÓN (en labs posteriores)
    ↓
SALIDA (y)
```
**Componentes clave:**
- **Entradas (x)**: Datos que la neurona recibe (características)
- **Pesos (w)**: Parámetros aprendibles que indican importancia de cada entrada
- **Bias (b)**: Parámetro aprendible que ajusta el umbral de activación
- **Suma ponderada**: Combinación lineal de entradas y pesos
- **Salida (y)**: Resultado del procesamiento de la neurona
### Conceptos Fundamentales
**1. Producto Punto (Dot Product):**
La operación fundamental en neuronas es el producto punto entre vectores:
$$\text{salida} = \mathbf{x} \cdot \mathbf{w} + b = \sum_{i=1}^{n} x_i \times w_i + b$$
**2. Procesamiento en Batch:**
En lugar de procesar una muestra a la vez, procesamos múltiples muestras simultáneamente usando álgebra matricial:
$$\mathbf{Y} = \mathbf{X} \cdot \mathbf{W} + \mathbf{b}$$
Donde:
- **X**: matriz de entradas (batch_size, n_entradas)
- **W**: matriz de pesos (n_entradas, n_neuronas)
- **b**: vector de bias (n_neuronas)
- **Y**: matriz de salidas (batch_size, n_neuronas)
**3. Parámetros Aprendibles:**
Los pesos y bias son los parámetros que la red "aprende" durante el entrenamiento:
- Se inicializan aleatoriamente
- Se ajustan iterativamente para minimizar el error
- Determinan el comportamiento de la neurona
### Aplicaciones Prácticas
Las neuronas artificiales y las redes que forman son la base de:
- **Visión Computacional**: Reconocimiento de objetos, rostros, escenas
- **Procesamiento de Lenguaje Natural**: Traducción, resumen, chatbots
- **Sistemas de Recomendación**: Netflix, YouTube, Spotify
- **Medicina**: Diagnóstico de enfermedades, análisis de imágenes médicas
- **Finanzas**: Predicción de mercados, detección de fraude
- **Robótica**: Control de movimientos, navegación autónoma
### Motivación Histórica
La neurona artificial fue propuesta por McCulloch y Pitts en 1943, inspirada en el funcionamiento del cerebro:
- **Neurona biológica**: Recibe señales eléctricas a través de dendritas, las procesa en el soma, y transmite el resultado por el axón
- **Neurona artificial**: Recibe entradas numéricas, calcula una suma ponderada, y produce una salida
Aunque mucho más simple que una neurona real, este modelo ha demostrado ser extraordinariamente poderoso cuando se combina con otras neuronas en redes.
## 🔬 Parte 1: Fundamentos - Tu Primera Neurona (30 min)
### 1.1 Introducción Conceptual
Una neurona artificial es una función matemática simple que transforma entradas en una salida mediante una suma ponderada.
**Analogía**: Imagina que estás decidiendo si ir al gimnasio. Consideras tres factores:
- **Clima** (x₁): ¿Está soleado? (0 = no, 1 = sí)
- **Energía** (x₂): ¿Te sientes con energía? (0-10)
- **Tiempo disponible** (x₃): ¿Tienes tiempo? (0 = no, 1 = sí)
Cada factor tiene diferente importancia para ti (pesos):
- w₁ = 0.3 (el clima importa un poco)
- w₂ = 0.6 (la energía importa mucho)
- w₃ = 0.8 (el tiempo es muy importante)
Y tienes una predisposición general (bias):
- b = -2.0 (en general, prefieres quedarte en casa)
**Cálculo de decisión**:
```
decisión = (1 × 0.3) + (8 × 0.6) + (1 × 0.8) + (-2.0)
         = 0.3 + 4.8 + 0.8 - 2.0
         = 3.9
```
Un valor positivo alto sugiere ir al gimnasio. ¡Eso es exactamente lo que hace una neurona!
### 1.2 Implementación Básica en Python Puro
Comencemos implementando una neurona usando solo Python, sin bibliotecas:
```python
# Neurona simple - Versión Python puro
def neurona_simple(entradas, pesos, bias):
    """
    Implementa una neurona artificial básica.
    
    Args:
        entradas: Lista de valores de entrada [x1, x2, x3, ...]
        pesos: Lista de pesos [w1, w2, w3, ...]
        bias: Valor del sesgo (número)
    
    Returns:
        salida: Resultado de la neurona (número)
    """
    # Inicializar la salida con el bias
    salida = bias
    
    # Sumar cada entrada multiplicada por su peso
    for entrada, peso in zip(entradas, pesos):
        salida += entrada * peso
    
    return salida
# Ejemplo de uso
entradas = [1.0, 2.0, 3.0]  # Tres características
pesos = [0.2, 0.8, -0.5]    # Pesos de cada característica
bias = 2.0                   # Sesgo
resultado = neurona_simple(entradas, pesos, bias)
print(f"Salida de la neurona: {resultado}")
# Salida esperada: (1.0×0.2) + (2.0×0.8) + (3.0×-0.5) + 2.0 = 2.3
```
**Actividad 1.1**: Ejecuta el código anterior y verifica manualmente el cálculo. ¿Coincide con tu cálculo a mano?
**Pregunta de Reflexión 1.1**: Si todos los pesos son cero, ¿qué valor tendrá la salida sin importar las entradas? ¿Por qué?
### 1.3 Producto Punto - La Operación Fundamental
El producto punto (dot product) es una forma más eficiente de calcular la suma ponderada:
```python
def producto_punto(vector_a, vector_b):
    """
    Calcula el producto punto entre dos vectores.
    
    Args:
        vector_a: Primera lista de números
        vector_b: Segunda lista de números (misma longitud)
    
    Returns:
        resultado: Suma de productos elemento a elemento
    """
    return sum(a * b for a, b in zip(vector_a, vector_b))
# Ejemplo de uso
x = [1.0, 2.0, 3.0]
w = [0.2, 0.8, -0.5]
dot_product = producto_punto(x, w)
print(f"Producto punto: {dot_product}")
# Resultado: 0.3
```
Ahora podemos reescribir nuestra neurona de forma más elegante:
```python
def neurona_dot(entradas, pesos, bias):
    """Neurona usando producto punto."""
    return producto_punto(entradas, pesos) + bias
```
**Actividad 1.2**: Implementa tu propia función `producto_punto_manual()` usando un loop explícito en lugar de la comprensión de listas.
**Pregunta de Reflexión 1.2**: ¿Por qué es importante usar producto punto en vez de loops cuando trabajamos con muchos datos?
### 1.4 Introducción a NumPy - Vectorización
NumPy es una biblioteca fundamental para computación científica en Python. Permite operaciones vectorizadas que son mucho más rápidas que loops de Python puro.
```python
import numpy as np
# Preparación del entorno
print("🔧 Configuración de NumPy")
print(f"Versión de NumPy: {np.__version__}")
# Crear arrays NumPy
entradas = np.array([1.0, 2.0, 3.0])
pesos = np.array([0.2, 0.8, -0.5])
bias = 2.0
print(f"\nEntradas: {entradas}")
print(f"Pesos: {pesos}")
print(f"Bias: {bias}")
# Calcular salida usando producto punto de NumPy
salida = np.dot(entradas, pesos) + bias
print(f"\nSalida de la neurona: {salida}")
# Alternativamente, usando el operador @
salida_alt = entradas @ pesos + bias
print(f"Salida (usando @): {salida_alt}")
```
**Ventajas de NumPy**:
1. **Velocidad**: ~100x más rápido que Python puro
2. **Sintaxis clara**: Código más legible y conciso
3. **Broadcasting**: Operaciones automáticas con diferentes dimensiones
4. **Estándar**: Usado en toda la comunidad de ML/DL
**Actividad 1.3**: Compara el tiempo de ejecución entre la implementación Python pura y NumPy para 1,000,000 de operaciones.
```python
import time
# Generar datos de prueba
n = 1000000
entradas = [1.0] * n
pesos = [0.5] * n
# Python puro
inicio = time.time()
resultado_puro = sum(e * p for e, p in zip(entradas, pesos))
tiempo_puro = time.time() - inicio
# NumPy
entradas_np = np.array(entradas)
pesos_np = np.array(pesos)
inicio = time.time()
resultado_numpy = np.dot(entradas_np, pesos_np)
tiempo_numpy = time.time() - inicio
print(f"Python puro: {tiempo_puro:.6f} segundos")
print(f"NumPy: {tiempo_numpy:.6f} segundos")
print(f"Aceleración: {tiempo_puro/tiempo_numpy:.2f}x")
```
### Actividades
**Actividad 1.4**: Crea una neurona que tome 5 entradas. Prueba con diferentes valores de entrada y observa cómo cambia la salida.
**Actividad 1.5**: Experimenta con el bias. Fija todos los pesos a cero y cambia solo el bias. ¿Qué hace el bias?
**Actividad 1.6**: Crea una neurona donde solo un peso sea 1.0 y el resto ceros. ¿Qué función realiza esta neurona?
### Preguntas de Reflexión
**Pregunta 1.3 (Concebir)**: ¿Cómo podrías usar una neurona para tomar una decisión automática basada en múltiples factores? Piensa en un ejemplo del mundo real.
**Pregunta 1.4 (Diseñar)**: Si quisieras que una neurona ignore completamente una entrada específica, ¿qué valor deberías darle a su peso correspondiente?
**Pregunta 1.5 (Implementar)**: ¿Qué diferencias observaste en la implementación entre Python puro y NumPy? ¿Cuál prefieres y por qué?
**Pregunta 1.6 (Operar)**: En aplicaciones del mundo real con millones de datos, ¿por qué es crítico usar NumPy en lugar de Python puro?
## 🔬 Parte 2: Capas de Neuronas (45 min)
### 2.1 De Una Neurona a Múltiples Neuronas
Una sola neurona es limitada. Para resolver problemas complejos, necesitamos combinar múltiples neuronas en una **capa**.
**Conceptualmente**:
```
       Entrada x₁ ─┬─> [Neurona 1] ─> salida₁
                   ├─> [Neurona 2] ─> salida₂
       Entrada x₂ ─┼─> [Neurona 3] ─> salida₃
                   ├─> [Neurona 4] ─> salida₄
       Entrada x₃ ─┘
```
Cada neurona en la capa:
- Recibe las **mismas** entradas
- Tiene sus **propios** pesos y bias únicos
- Produce su **propia** salida independiente
### 2.2 Implementación con Loops
```python
import numpy as np
def capa_neuronal_loop(entradas, pesos_matriz, biases):
    """
    Capa de neuronas implementada con loops.
    
    Args:
        entradas: Array (n_entradas,)
        pesos_matriz: Array (n_entradas, n_neuronas)
        biases: Array (n_neuronas,)
    
    Returns:
        salidas: Array (n_neuronas,)
    """
    salidas = []
    
    # Para cada neurona en la capa
    for i in range(len(biases)):
        # Obtener los pesos de esta neurona específica
        pesos_neurona = pesos_matriz[:, i]
        
        # Calcular salida de esta neurona
        salida_neurona = np.dot(entradas, pesos_neurona) + biases[i]
        salidas.append(salida_neurona)
    
    return np.array(salidas)
# Ejemplo: 3 entradas, 4 neuronas
entradas = np.array([1.0, 2.0, 3.0])
# Cada columna representa los pesos de una neurona
pesos = np.array([
    [0.2, 0.8, -0.5, 0.1],  # Pesos para entrada 1
    [0.5, -0.9, 0.3, 0.7],  # Pesos para entrada 2
    [-0.26, 0.4, 0.6, -0.2] # Pesos para entrada 3
])
biases = np.array([2.0, 3.0, 0.5, 1.0])
salidas = capa_neuronal_loop(entradas, pesos, biases)
print(f"Salidas de la capa: {salidas}")
print(f"Forma de salidas: {salidas.shape}")
```
**Actividad 2.1**: Verifica manualmente el cálculo de la primera neurona (índice 0). ¿Coincide con el valor en salidas[0]?
### 2.3 Implementación Vectorizada (Óptima)
La forma correcta y eficiente de implementar una capa es usando multiplicación matricial:
```python
def capa_neuronal_vectorizada(entradas, pesos, biases):
    """
    Capa de neuronas - Implementación vectorizada.
    
    Args:
        entradas: Array (n_entradas,)
        pesos: Array (n_entradas, n_neuronas)
        biases: Array (n_neuronas,)
    
    Returns:
        salidas: Array (n_neuronas,)
    """
    return np.dot(entradas, pesos) + biases
# Usando los mismos datos anteriores
salidas_vec = capa_neuronal_vectorizada(entradas, pesos, biases)
print(f"\nSalidas vectorizadas: {salidas_vec}")
print(f"¿Son iguales? {np.allclose(salidas, salidas_vec)}")
```
**¿Por qué es mejor?**
- Una sola línea de código
- Mucho más rápido (operaciones optimizadas en C)
- Aprovecha hardware especializado (CPU SIMD, GPU)
- Estándar en la industria
**Actividad 2.2**: Mide el tiempo de ejecución de ambas implementaciones con una capa de 1000 neuronas y 10000 entradas.
### 2.4 Procesamiento en Batch
En la práctica, nunca procesamos una muestra a la vez. Procesamos **batches** (lotes) de múltiples muestras simultáneamente.
```python
def capa_neuronal_batch(entradas_batch, pesos, biases):
    """
    Capa de neuronas que procesa un batch de muestras.
    
    Args:
        entradas_batch: Array (batch_size, n_entradas)
        pesos: Array (n_entradas, n_neuronas)
        biases: Array (n_neuronas,)
    
    Returns:
        salidas: Array (batch_size, n_neuronas)
    """
    return np.dot(entradas_batch, pesos) + biases
# Ejemplo: 3 muestras, cada una con 3 características
batch = np.array([
    [1.0, 2.0, 3.0],   # Muestra 1
    [0.5, 1.5, 2.5],   # Muestra 2
    [2.0, 1.0, 0.5]    # Muestra 3
])
print(f"\nBatch de entrada shape: {batch.shape}")
print(f"Pesos shape: {pesos.shape}")
print(f"Biases shape: {biases.shape}")
salidas_batch = capa_neuronal_batch(batch, pesos, biases)
print(f"\nSalidas del batch shape: {salidas_batch.shape}")
print(f"Salidas del batch:\n{salidas_batch}")
```
**Interpretación**:
- Cada fila en `salidas_batch` corresponde a una muestra
- Cada columna corresponde a una neurona
- `salidas_batch[i, j]` = salida de la neurona j para la muestra i
**Actividad 2.3**: Verifica que `salidas_batch[0]` sea igual a la salida cuando procesamos `batch[0]` individualmente.
### 2.5 Visualización de Dimensiones
Entender las dimensiones es crucial. Usemos una notación clara:
```python
def visualizar_dimensiones():
    """Imprime información clara sobre dimensiones."""
    
    # Configuración
    n_muestras = 5      # batch size
    n_entradas = 3      # features por muestra
    n_neuronas = 4      # neuronas en la capa
    
    # Crear datos de ejemplo
    X = np.random.randn(n_muestras, n_entradas)
    W = np.random.randn(n_entradas, n_neuronas) * 0.01
    b = np.zeros(n_neuronas)
    
    # Forward pass
    Y = np.dot(X, W) + b
    
    # Visualizar
    print("="*60)
    print("ANÁLISIS DE DIMENSIONES EN UNA CAPA NEURONAL")
    print("="*60)
    print(f"\n📊 Configuración:")
    print(f"   - Muestras en el batch: {n_muestras}")
    print(f"   - Características por muestra: {n_entradas}")
    print(f"   - Neuronas en la capa: {n_neuronas}")
    
    print(f"\n📐 Dimensiones de Tensores:")
    print(f"   - Entradas (X): {X.shape} = (batch_size, n_entradas)")
    print(f"   - Pesos (W): {W.shape} = (n_entradas, n_neuronas)")
    print(f"   - Biases (b): {b.shape} = (n_neuronas,)")
    print(f"   - Salidas (Y): {Y.shape} = (batch_size, n_neuronas)")
    
    print(f"\n🔢 Operación Matricial:")
    print(f"   Y = X @ W + b")
    print(f"   {Y.shape} = {X.shape} @ {W.shape} + {b.shape}")
    
    print(f"\n💡 Interpretación:")
    print(f"   - Cada fila de Y = salidas para una muestra")
    print(f"   - Cada columna de Y = salida de una neurona específica")
    print(f"   - Y[i,j] = salida de neurona j para muestra i")
    
    print(f"\n📊 Número de Parámetros:")
    n_params = n_entradas * n_neuronas + n_neuronas
    print(f"   - Pesos: {n_entradas} × {n_neuronas} = {n_entradas * n_neuronas}")
    print(f"   - Biases: {n_neuronas}")
    print(f"   - TOTAL: {n_params} parámetros aprendibles")
    print("="*60)
visualizar_dimensiones()
```
**Actividad 2.4**: Modifica la función para una capa con 10 entradas y 20 neuronas. ¿Cuántos parámetros tiene?
### 2.6 Implementación Orientada a Objetos
Para mayor claridad y reutilización, encapsulemos en una clase:
```python
class CapaDensa:
    """
    Capa densa (fully connected) de neuronas.
    """
    
    def __init__(self, n_entradas, n_neuronas):
        """
        Inicializa la capa.
        
        Args:
            n_entradas: Número de características de entrada
            n_neuronas: Número de neuronas en esta capa
        """
        # Inicializar pesos con valores pequeños aleatorios
        self.pesos = 0.01 * np.random.randn(n_entradas, n_neuronas)
        
        # Inicializar biases en cero
        self.biases = np.zeros(n_neuronas)
        
        print(f"✅ Capa creada: {n_entradas} entradas → {n_neuronas} neuronas")
        print(f"   Parámetros: {self.contar_parametros()}")
    
    def forward(self, entradas):
        """
        Propagación hacia adelante.
        
        Args:
            entradas: Array (batch_size, n_entradas)
        
        Returns:
            salidas: Array (batch_size, n_neuronas)
        """
        self.entradas = entradas
        self.salidas = np.dot(entradas, self.pesos) + self.biases
        return self.salidas
    
    def contar_parametros(self):
        """Cuenta el número total de parámetros."""
        return self.pesos.size + self.biases.size
    
    def info(self):
        """Imprime información de la capa."""
        print(f"\n📊 Información de la Capa:")
        print(f"   Shape de pesos: {self.pesos.shape}")
        print(f"   Shape de biases: {self.biases.shape}")
        print(f"   Total parámetros: {self.contar_parametros()}")
# Ejemplo de uso
capa = CapaDensa(n_entradas=3, n_neuronas=4)
capa.info()
# Procesar un batch
batch = np.random.randn(5, 3)  # 5 muestras, 3 características
salidas = capa.forward(batch)
print(f"\n🔄 Forward Pass:")
print(f"   Entrada: {batch.shape}")
print(f"   Salida: {salidas.shape}")
```
**Actividad 2.5**: Crea una capa con 784 entradas (imagen 28×28) y 128 neuronas. ¿Cuántos parámetros tiene?
**Actividad 2.6**: Procesa un batch de 32 imágenes. ¿Qué shape tendrá la salida?
### Actividades Integradoras
**Actividad 2.7**: Implementa una función `test_capa()` que verifique que tu implementación funciona correctamente con diferentes tamaños.
```python
def test_capa():
    """Prueba la capa con diferentes configuraciones."""
    configuraciones = [
        (10, 5, 3),    # (n_entradas, n_neuronas, batch_size)
        (784, 128, 32),
        (100, 50, 64)
    ]
    
    for n_in, n_out, batch_size in configuraciones:
        print(f"\n🧪 Probando: {n_in} → {n_out}, batch={batch_size}")
        
        capa = CapaDensa(n_in, n_out)
        X = np.random.randn(batch_size, n_in)
        Y = capa.forward(X)
        
        # Verificaciones
        assert Y.shape == (batch_size, n_out), "Shape incorrecto!"
        assert not np.isnan(Y).any(), "NaN detectado!"
        
        print(f"   ✅ Test pasado")
test_capa()
```
**Actividad 2.8**: Crea una visualización que muestre cómo diferentes valores de pesos afectan la salida.
### Preguntas de Reflexión
**Pregunta 2.1 (Concebir)**: ¿Por qué es útil procesar múltiples muestras simultáneamente (batches) en lugar de una por una?
**Pregunta 2.2 (Diseñar)**: Si tienes un batch de 64 muestras y una capa de 256 neuronas, ¿cuál es el shape de la matriz de salida?
**Pregunta 2.3 (Implementar)**: ¿Qué ventajas tiene la implementación orientada a objetos sobre funciones simples?
**Pregunta 2.4 (Operar)**: En un sistema de producción procesando millones de datos, ¿por qué es crítico usar operaciones vectorizadas?
## 🔬 Parte 3: Inicialización y Buenas Prácticas (30 min)
### 3.1 Importancia de la Inicialización
La forma en que inicializamos los pesos afecta dramáticamente el aprendizaje de la red.
#### Problema: Todos los Pesos en Cero
```python
# ❌ MAL: Todos los pesos en cero
class CapaMala:
    def __init__(self, n_entradas, n_neuronas):
        self.pesos = np.zeros((n_entradas, n_neuronas))  # ¡Problema!
        self.biases = np.zeros(n_neuronas)
# Demostración del problema
capa_mala = CapaMala(3, 4)
entradas = np.array([[1.0, 2.0, 3.0]])
salidas = np.dot(entradas, capa_mala.pesos) + capa_mala.biases
print("Con pesos en cero:")
print(f"Salidas: {salidas}")
print("❌ Todas las salidas son cero - no hay información!")
```
**Problema de Simetría**: Si todos los pesos son iguales:
- Todas las neuronas calcularán lo mismo
- Recibirán los mismos gradientes (en backpropagation)
- Aprenderán exactamente las mismas características
- ¡La capa se comporta como si tuviera una sola neurona!
#### Solución 1: Inicialización Aleatoria Pequeña
```python
class CapaBuena:
    def __init__(self, n_entradas, n_neuronas):
        # ✅ BIEN: Valores aleatorios pequeños
        self.pesos = 0.01 * np.random.randn(n_entradas, n_neuronas)
        self.biases = np.zeros(n_neuronas)  # Biases pueden ser cero
# Demostración
capa_buena = CapaBuena(3, 4)
salidas = np.dot(entradas, capa_buena.pesos) + capa_buena.biases
print("\nCon pesos aleatorios pequeños:")
print(f"Salidas: {salidas}")
print("✅ Cada neurona produce valores diferentes")
```
**Actividad 3.1**: Crea dos capas idénticas pero con semillas aleatorias diferentes. Verifica que producen salidas diferentes para las mismas entradas.
#### Solución 2: Inicialización Xavier/Glorot
Para redes más profundas, queremos que la varianza de las salidas sea similar a la de las entradas:
```python
class CapaXavier:
    def __init__(self, n_entradas, n_neuronas):
        # Inicialización Xavier/Glorot
        limite = np.sqrt(6.0 / (n_entradas + n_neuronas))
        self.pesos = np.random.uniform(-limite, limite, 
                                        (n_entradas, n_neuronas))
        self.biases = np.zeros(n_neuronas)
# Comparación de varianzas
capa_simple = CapaBuena(100, 100)
capa_xavier = CapaXavier(100, 100)
X = np.random.randn(1000, 100)
Y_simple = np.dot(X, capa_simple.pesos) + capa_simple.biases
Y_xavier = np.dot(X, capa_xavier.pesos) + capa_xavier.biases
print(f"\nVarianza de entradas: {np.var(X):.4f}")
print(f"Varianza con init simple: {np.var(Y_simple):.4f}")
print(f"Varianza con init Xavier: {np.var(Y_xavier):.4f}")
```
**Actividad 3.2**: Experimenta con diferentes escalas de inicialización (0.001, 0.01, 0.1, 1.0). ¿Cómo afecta a las salidas?
### 3.2 Análisis de Estadísticas
Es importante monitorear las estadísticas de las salidas para detectar problemas:
```python
def analizar_capa(capa, entradas):
    """
    Analiza las estadísticas de una capa.
    
    Args:
        capa: Instancia de CapaDensa
        entradas: Datos de entrada
    """
    salidas = capa.forward(entradas)
    
    print("\n" + "="*60)
    print("ANÁLISIS ESTADÍSTICO DE LA CAPA")
    print("="*60)
    
    print(f"\n📊 Estadísticas de Pesos:")
    print(f"   Media: {np.mean(capa.pesos):.6f}")
    print(f"   Desviación estándar: {np.std(capa.pesos):.6f}")
    print(f"   Mínimo: {np.min(capa.pesos):.6f}")
    print(f"   Máximo: {np.max(capa.pesos):.6f}")
    
    print(f"\n📊 Estadísticas de Salidas:")
    print(f"   Media: {np.mean(salidas):.6f}")
    print(f"   Desviación estándar: {np.std(salidas):.6f}")
    print(f"   Mínimo: {np.min(salidas):.6f}")
    print(f"   Máximo: {np.max(salidas):.6f}")
    
    # Detectar problemas
    print(f"\n⚠️  Diagnóstico:")
    if np.std(salidas) < 0.01:
        print("   ❌ Varianza muy baja - posible problema de inicialización")
    elif np.std(salidas) > 10:
        print("   ❌ Varianza muy alta - posible explosión de gradientes")
    else:
        print("   ✅ Varianza en rango saludable")
    
    if np.abs(np.mean(salidas)) > 1.0:
        print("   ⚠️  Media alejada de cero")
    else:
        print("   ✅ Media cercana a cero")
# Probar
capa = CapaDensa(100, 50)
X = np.random.randn(1000, 100)
analizar_capa(capa, X)
```
**Actividad 3.3**: Analiza capas con diferentes tamaños. ¿Cómo cambian las estadísticas?
### 3.3 Mejores Prácticas de Implementación
```python
class CapaDensaCompleta:
    """
    Implementación completa con buenas prácticas.
    """
    
    def __init__(self, n_entradas, n_neuronas, seed=None):
        """
        Args:
            n_entradas: Número de características de entrada
            n_neuronas: Número de neuronas en la capa
            seed: Semilla para reproducibilidad (opcional)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Validación de parámetros
        assert n_entradas > 0, "n_entradas debe ser positivo"
        assert n_neuronas > 0, "n_neuronas debe ser positivo"
        
        self.n_entradas = n_entradas
        self.n_neuronas = n_neuronas
        
        # Inicialización de pesos (He initialization)
        self.pesos = np.random.randn(n_entradas, n_neuronas) * np.sqrt(2.0 / n_entradas)
        self.biases = np.zeros(n_neuronas)
        
        # Para guardar valores intermedios (útil para debugging)
        self.entradas = None
        self.salidas = None
    
    def forward(self, entradas):
        """
        Propagación hacia adelante.
        
        Args:
            entradas: Array (batch_size, n_entradas)
        
        Returns:
            salidas: Array (batch_size, n_neuronas)
        """
        # Validación de dimensiones
        assert entradas.ndim == 2, "Entradas deben ser 2D (batch_size, n_entradas)"
        assert entradas.shape[1] == self.n_entradas, \
            f"Esperaba {self.n_entradas} entradas, recibió {entradas.shape[1]}"
        
        # Guardar entradas para backpropagation (futuro)
        self.entradas = entradas
        
        # Calcular salidas
        self.salidas = np.dot(entradas, self.pesos) + self.biases
        
        return self.salidas
    
    def get_params(self):
        """Retorna diccionario con parámetros."""
        return {
            'pesos': self.pesos.copy(),
            'biases': self.biases.copy()
        }
    
    def set_params(self, params):
        """Establece parámetros desde un diccionario."""
        self.pesos = params['pesos'].copy()
        self.biases = params['biases'].copy()
    
    def contar_parametros(self):
        """Cuenta parámetros totales."""
        return self.pesos.size + self.biases.size
    
    def __repr__(self):
        """Representación en string."""
        return f"CapaDensa({self.n_entradas} → {self.n_neuronas}, params={self.contar_parametros()})"
# Uso
capa = CapaDensaCompleta(784, 128, seed=42)
print(capa)
X = np.random.randn(32, 784)
Y = capa.forward(X)
print(f"Entrada: {X.shape} → Salida: {Y.shape}")
```
**Actividad 3.4**: Añade un método `summary()` que imprima información detallada de la capa.
### 3.4 Verificación de Implementación
```python
def test_capa_completa():
    """Suite de tests para verificar la capa."""
    
    print("\n🧪 SUITE DE TESTS PARA CAPA DENSA")
    print("="*60)
    
    # Test 1: Dimensiones
    print("\n1️⃣  Test de Dimensiones")
    capa = CapaDensaCompleta(10, 5)
    X = np.random.randn(3, 10)
    Y = capa.forward(X)
    assert Y.shape == (3, 5), "Dimensiones incorrectas"
    print("   ✅ Dimensiones correctas")
    
    # Test 2: Reproducibilidad
    print("\n2️⃣  Test de Reproducibilidad")
    capa1 = CapaDensaCompleta(10, 5, seed=42)
    capa2 = CapaDensaCompleta(10, 5, seed=42)
    X = np.random.randn(3, 10)
    Y1 = capa1.forward(X)
    Y2 = capa2.forward(X)
    assert np.allclose(Y1, Y2), "No es reproducible"
    print("   ✅ Reproducible con seed")
    
    # Test 3: Biases funcionan
    print("\n3️⃣  Test de Biases")
    capa = CapaDensaCompleta(10, 5, seed=42)
    capa.biases = np.ones(5) * 10.0  # Bias grande
    X = np.zeros((3, 10))  # Entrada cero
    Y = capa.forward(X)
    assert np.allclose(Y, 10.0), "Biases no funcionan"
    print("   ✅ Biases funcionan correctamente")
    
    # Test 4: Pesos afectan salida
    print("\n4️⃣  Test de Pesos")
    capa = CapaDensaCompleta(10, 5, seed=42)
    X = np.random.randn(3, 10)
    Y1 = capa.forward(X)
    capa.pesos *= 2  # Duplicar pesos
    Y2 = capa.forward(X)
    assert not np.allclose(Y1, Y2), "Pesos no afectan salida"
    print("   ✅ Pesos afectan salida")
    
    # Test 5: Batch processing
    print("\n5️⃣  Test de Batch Processing")
    capa = CapaDensaCompleta(10, 5, seed=42)
    X_single = np.random.randn(1, 10)
    X_batch = np.repeat(X_single, 10, axis=0)
    Y_single = capa.forward(X_single)
    Y_batch = capa.forward(X_batch)
    assert np.allclose(Y_single, Y_batch[0:1]), "Batch processing incorrecto"
    print("   ✅ Batch processing funciona")
    
    print("\n" + "="*60)
    print("✅ TODOS LOS TESTS PASARON")
    print("="*60)
test_capa_completa()
```
**Actividad 3.5**: Añade tests adicionales para casos extremos (valores muy grandes, muy pequeños, NaN, etc.)
### Preguntas de Reflexión
**Pregunta 3.1 (Concebir)**: ¿Por qué la inicialización aleatoria es crucial para romper la simetría entre neuronas?
**Pregunta 3.2 (Diseñar)**: ¿Qué podría suceder si inicializas los pesos con valores muy grandes (ej: 100)?
**Pregunta 3.3 (Implementar)**: ¿Por qué es útil guardar las entradas durante el forward pass?
**Pregunta 3.4 (Operar)**: En un entorno de producción, ¿por qué querrías hacer reproducible la inicialización usando seeds?
## 🔬 Parte 4: Aplicaciones Prácticas y Limitaciones (45 min)
### 4.1 Ejemplo Práctico: Clasificador Lineal Simple
Vamos a usar una neurona para crear un clasificador lineal que separe dos clases.
```python
import matplotlib.pyplot as plt
def generar_datos_clasificacion(n_muestras=100, seed=42):
    """
    Genera datos sintéticos para clasificación binaria.
    
    Returns:
        X: Features (n_muestras, 2)
        y: Labels (n_muestras,) - 0 o 1
    """
    np.random.seed(seed)
    
    # Clase 0: centrada en (2, 2)
    X_clase0 = np.random.randn(n_muestras//2, 2) + np.array([2, 2])
    y_clase0 = np.zeros(n_muestras//2)
    
    # Clase 1: centrada en (-2, -2)
    X_clase1 = np.random.randn(n_muestras//2, 2) + np.array([-2, -2])
    y_clase1 = np.ones(n_muestras//2)
    
    # Combinar
    X = np.vstack([X_clase0, X_clase1])
    y = np.hstack([y_clase0, y_clase1])
    
    # Mezclar
    indices = np.random.permutation(n_muestras)
    X = X[indices]
    y = y[indices]
    
    return X, y
# Generar y visualizar datos
X, y = generar_datos_clasificacion(200)
plt.figure(figsize=(10, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Clase 0', alpha=0.6, s=50)
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Clase 1', alpha=0.6, s=50)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Datos de Clasificación Binaria', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(f"Datos generados:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  Clase 0: {np.sum(y==0)} muestras")
print(f"  Clase 1: {np.sum(y==1)} muestras")
```
**Actividad 4.1**: Modifica los centros de las clases y observa cómo cambian los datos.
### 4.2 Neurona como Frontera de Decisión
Una sola neurona con 2 entradas define una línea recta en el espacio 2D:
```python
def visualizar_frontera_decision(pesos, bias):
    """
    Visualiza la frontera de decisión de una neurona.
    
    Args:
        pesos: Array (2,) - pesos de la neurona
        bias: float - bias de la neurona
    """
    # Generar datos
    X, y = generar_datos_clasificacion(200)
    
    # Calcular salidas de la neurona
    salidas = np.dot(X, pesos) + bias
    predicciones = (salidas > 0).astype(int)  # Umbral en 0
    
    # Calcular precisión
    precision = np.mean(predicciones == y)
    
    # Visualizar
    plt.figure(figsize=(10, 6))
    
    # Puntos de datos
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Clase 0 (real)', 
                alpha=0.6, s=50)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Clase 1 (real)', 
                alpha=0.6, s=50)
    
    # Frontera de decisión: w1*x1 + w2*x2 + b = 0
    # Despejamos: x2 = -(w1*x1 + b) / w2
    x1_range = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100)
    if pesos[1] != 0:
        x2_frontera = -(pesos[0] * x1_range + bias) / pesos[1]
        plt.plot(x1_range, x2_frontera, 'g--', linewidth=3, 
                label=f'Frontera de decisión (acc={precision:.2%})')
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(f'Neurona como Clasificador Lineal\nPesos={pesos}, Bias={bias:.2f}', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return precision
# Probar con diferentes configuraciones
print("Configuración 1: Pesos [1, 1], Bias 0")
acc1 = visualizar_frontera_decision(np.array([1.0, 1.0]), 0.0)
print(f"\nConfiguración 2: Pesos [1, -1], Bias 0")
acc2 = visualizar_frontera_decision(np.array([1.0, -1.0]), 0.0)
print(f"\nConfiguración 3: Pesos óptimos (ajustados)")
# Calcular pesos que separan las clases
X, y = generar_datos_clasificacion(200)
media_clase0 = X[y==0].mean(axis=0)
media_clase1 = X[y==1].mean(axis=0)
pesos_optimos = media_clase1 - media_clase0
pesos_optimos = pesos_optimos / np.linalg.norm(pesos_optimos)
acc3 = visualizar_frontera_decision(pesos_optimos, 0.0)
```
**Actividad 4.2**: Experimenta con diferentes valores de pesos y bias para maximizar la precisión.
### 4.3 Limitación: Problema XOR
El problema XOR es el ejemplo clásico que demuestra la limitación de una sola neurona:
```python
def demostrar_problema_xor():
    """
    Demuestra que una sola neurona no puede resolver XOR.
    """
    # Datos XOR
    X_xor = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_xor = np.array([0, 1, 1, 0])  # XOR output
    
    print("="*60)
    print("PROBLEMA XOR")
    print("="*60)
    print("\nTabla de verdad XOR:")
    print("  x1  x2  |  y")
    print("  -----------")
    for i in range(4):
        print(f"  {X_xor[i,0]}   {X_xor[i,1]}   |  {y_xor[i]}")
    
    # Intentar múltiples configuraciones de pesos
    mejores_pesos = None
    mejor_precision = 0
    
    print("\n🔍 Buscando pesos óptimos...")
    for _ in range(1000):
        # Probar pesos aleatorios
        pesos = np.random.randn(2)
        bias = np.random.randn()
        
        # Calcular salidas
        salidas = np.dot(X_xor, pesos) + bias
        predicciones = (salidas > 0).astype(int)
        
        # Calcular precisión
        precision = np.mean(predicciones == y_xor)
        
        if precision > mejor_precision:
            mejor_precision = precision
            mejores_pesos = pesos
            mejor_bias = bias
    
    print(f"\n📊 Resultados después de 1000 intentos:")
    print(f"   Mejor precisión: {mejor_precision:.2%}")
    print(f"   Mejores pesos: {mejores_pesos}")
    print(f"   Mejor bias: {mejor_bias:.4f}")
    
    # Visualizar
    plt.figure(figsize=(10, 6))
    colors = ['blue' if y == 0 else 'red' for y in y_xor]
    plt.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200, alpha=0.6, 
                edgecolors='black', linewidth=2)
    
    # Etiquetas
    for i, (x, y) in enumerate(X_xor):
        plt.annotate(f'({x},{y})→{y_xor[i]}', (x, y), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Intentar mostrar frontera de decisión
    if mejores_pesos is not None and mejores_pesos[1] != 0:
        x_range = np.linspace(-0.5, 1.5, 100)
        y_frontera = -(mejores_pesos[0] * x_range + mejor_bias) / mejores_pesos[1]
        plt.plot(x_range, y_frontera, 'g--', linewidth=2, 
                label=f'Mejor frontera (acc={mejor_precision:.0%})')
    
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title('Problema XOR - ¡No linealmente separable!', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.tight_layout()
    plt.show()
    
    print("\n⚠️  CONCLUSIÓN:")
    print("   Una sola neurona NO puede resolver XOR perfectamente.")
    print("   Máxima precisión posible: 75% (3 de 4 puntos correctos)")
    print("   Necesitamos MÚLTIPLES neuronas en capas (redes neuronales)")
    print("="*60)
demostrar_problema_xor()
```
**Pregunta de Reflexión 4.1**: ¿Por qué una línea recta no puede separar los puntos XOR?
### 4.4 Experimento: Capacidad Representacional
Exploremos cuántas funciones diferentes puede representar una capa de neuronas:
```python
def explorar_capacidad_representacional():
    """
    Explora diferentes patrones que puede aprender una capa.
    """
    # Crear grid de puntos
    x1 = np.linspace(-3, 3, 50)
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Diferentes configuraciones de capas
    configuraciones = [
        {
            'titulo': 'Una neurona - Frontera lineal',
            'n_neuronas': 1,
            'pesos': np.array([[1.0], [1.0]]),
            'bias': np.array([0.0])
        },
        {
            'titulo': 'Dos neuronas - Dos fronteras',
            'n_neuronas': 2,
            'pesos': np.array([[1.0, -1.0], [1.0, 1.0]]),
            'bias': np.array([0.0, 2.0])
        },
        {
            'titulo': 'Cuatro neuronas - Patrones complejos',
            'n_neuronas': 4,
            'pesos': np.random.randn(2, 4),
            'bias': np.random.randn(4)
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, config in enumerate(configuraciones):
        # Calcular salidas
        salidas = np.dot(X_grid, config['pesos']) + config['bias']
        
        # Visualizar la primera neurona (o suma de todas)
        if config['n_neuronas'] == 1:
            Z = salidas[:, 0].reshape(X1.shape)
        else:
            Z = salidas.sum(axis=1).reshape(X1.shape)
        
        # Graficar
        im = axes[idx].contourf(X1, X2, Z, levels=20, cmap='RdBu_r', alpha=0.7)
        axes[idx].contour(X1, X2, Z, levels=[0], colors='black', linewidths=2)
        axes[idx].set_title(config['titulo'], fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('x1')
        axes[idx].set_ylabel('x2')
        axes[idx].grid(True, alpha=0.3)
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    print("💡 Observaciones:")
    print("   - 1 neurona: Solo puede crear UNA frontera lineal")
    print("   - Múltiples neuronas: Pueden crear múltiples fronteras")
    print("   - Más neuronas = Mayor capacidad representacional")
    print("   - Pero SIN activación no lineal, ¡siguen siendo lineales!")
explorar_capacidad_representacional()
```
**Actividad 4.3**: Crea tu propia configuración con 8 neuronas y observa los patrones.
### 4.5 Caso de Estudio: Predicción Simple
Usemos neuronas para predecir el precio de una casa basándose en características:
```python
def caso_estudio_precios_casas():
    """
    Ejemplo realista: predecir precios de casas.
    """
    # Datos sintéticos
    np.random.seed(42)
    n_casas = 200
    
    # Features: [tamaño_m2, num_habitaciones, antiguedad_años]
    tamaño = np.random.uniform(50, 300, n_casas)
    habitaciones = np.random.randint(1, 6, n_casas)
    antiguedad = np.random.uniform(0, 50, n_casas)
    
    X = np.column_stack([tamaño, habitaciones, antiguedad])
    
    # Precio real (con una fórmula conocida + ruido)
    precio_real = (
        tamaño * 2000 +           # $2000 por m²
        habitaciones * 50000 +    # $50k por habitación
        -antiguedad * 1000 +      # -$1k por año
        100000                     # Base
    ) + np.random.normal(0, 20000, n_casas)  # Ruido
    
    # Normalizar features
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    precio_norm = (precio_real - precio_real.mean()) / precio_real.std()
    
    # Crear capa de regresión (1 neurona para 1 salida)
    capa = CapaDensaCompleta(3, 1, seed=42)
    
    # "Entrenar" manualmente ajustando pesos (simulado)
    # En labs futuros aprenderemos a entrenar automáticamente
    print("="*60)
    print("CASO DE ESTUDIO: PREDICCIÓN DE PRECIOS DE CASAS")
    print("="*60)
    
    print(f"\n📊 Dataset:")
    print(f"   Número de casas: {n_casas}")
    print(f"   Features: tamaño (m²), habitaciones, antigüedad (años)")
    print(f"   Target: precio ($)")
    
    print(f"\n📐 Estadísticas:")
    print(f"   Tamaño promedio: {tamaño.mean():.1f} m²")
    print(f"   Habitaciones promedio: {habitaciones.mean():.1f}")
    print(f"   Antigüedad promedio: {antiguedad.mean():.1f} años")
    print(f"   Precio promedio: ${precio_real.mean():,.0f}")
    
    # Predicciones con pesos iniciales (aleatorios)
    predicciones_iniciales = capa.forward(X_norm).flatten()
    predicciones_iniciales = predicciones_iniciales * precio_real.std() + precio_real.mean()
    
    error_inicial = np.mean(np.abs(predicciones_iniciales - precio_real))
    
    print(f"\n🔮 Predicciones con pesos aleatorios:")
    print(f"   Error Absoluto Medio: ${error_inicial:,.0f}")
    print(f"   (Muy malo - necesitamos entrenar!)")
    
    # Visualizar
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(precio_real / 1000, predicciones_iniciales / 1000, alpha=0.5)
    plt.plot([precio_real.min()/1000, precio_real.max()/1000], 
             [precio_real.min()/1000, precio_real.max()/1000], 
             'r--', linewidth=2, label='Predicción perfecta')
    plt.xlabel('Precio Real ($1000s)', fontsize=12)
    plt.ylabel('Precio Predicho ($1000s)', fontsize=12)
    plt.title('Pesos Aleatorios (Sin entrenar)', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    errores = predicciones_iniciales - precio_real
    plt.hist(errores / 1000, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Error de Predicción ($1000s)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Errores', fontsize=13, fontweight='bold')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n💡 Conclusión:")
    print(f"   Con pesos aleatorios, la neurona no hace buenas predicciones.")
    print(f"   En Lab 05-06 aprenderemos a ENTRENAR la red para minimizar errores.")
caso_estudio_precios_casas()
```
**Actividad 4.4**: Modifica la fórmula del precio real y observa cómo cambian las predicciones.
### Preguntas de Reflexión
**Pregunta 4.2 (Concebir)**: ¿Qué tipos de problemas puede resolver una sola neurona y cuáles no?
**Pregunta 4.3 (Diseñar)**: Para el problema de precios de casas, ¿cómo determinarías qué features son más importantes?
**Pregunta 4.4 (Implementar)**: ¿Por qué normalizamos las features antes de usarlas en la neurona?
**Pregunta 4.5 (Operar)**: En un sistema de producción, ¿cómo manejarías nuevos datos con valores fuera del rango de entrenamiento?
## 📊 Análisis Final de Rendimiento
### Comparación de Implementaciones
```python
def benchmark_implementaciones():
    """
    Compara el rendimiento de diferentes implementaciones.
    """
    import time
    
    print("\n" + "="*60)
    print("BENCHMARK: COMPARACIÓN DE IMPLEMENTACIONES")
    print("="*60)
    
    # Configuración
    configuraciones = [
        (100, 50, 1000),
        (784, 128, 10000),
        (1000, 500, 5000)
    ]
    
    for n_entradas, n_neuronas, batch_size in configuraciones:
        print(f"\n📐 Config: {n_entradas} entradas → {n_neuronas} neuronas, batch={batch_size}")
        
        # Generar datos
        X = np.random.randn(batch_size, n_entradas)
        W = np.random.randn(n_entradas, n_neuronas) * 0.01
        b = np.zeros(n_neuronas)
        
        # Método 1: NumPy vectorizado
        start = time.time()
        Y_numpy = np.dot(X, W) + b
        tiempo_numpy = time.time() - start
        
        # Método 2: Loop (solo para tamaños pequeños)
        if batch_size <= 1000:
            start = time.time()
            Y_loop = []
            for i in range(batch_size):
                y_i = np.dot(X[i], W) + b
                Y_loop.append(y_i)
            Y_loop = np.array(Y_loop)
            tiempo_loop = time.time() - start
            aceleracion = tiempo_loop / tiempo_numpy
        else:
            tiempo_loop = None
            aceleracion = None
        
        # Resultados
        print(f"   NumPy vectorizado: {tiempo_numpy*1000:.3f} ms")
        if tiempo_loop:
            print(f"   Loop Python: {tiempo_loop*1000:.3f} ms")
            print(f"   🚀 Aceleración: {aceleracion:.1f}x")
        
        # Verificar corrección
        if tiempo_loop and not np.allclose(Y_numpy, Y_loop):
            print("   ⚠️  ADVERTENCIA: Resultados no coinciden!")
benchmark_implementaciones()
```
### Análisis de Escalabilidad
```python
def analizar_escalabilidad():
    """
    Analiza cómo escala el costo computacional.
    """
    import time
    
    print("\n" + "="*60)
    print("ANÁLISIS DE ESCALABILIDAD")
    print("="*60)
    
    tamaños = [10, 50, 100, 500, 1000, 5000]
    tiempos = []
    
    for n in tamaños:
        X = np.random.randn(1000, n)
        W = np.random.randn(n, n)
        b = np.zeros(n)
        
        start = time.time()
        Y = np.dot(X, W) + b
        tiempo = time.time() - start
        tiempos.append(tiempo)
        
        print(f"   n={n:5d}: {tiempo*1000:7.2f} ms")
    
    # Visualizar
    plt.figure(figsize=(10, 6))
    plt.plot(tamaños, np.array(tiempos) * 1000, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Tamaño de capa (n)', fontsize=12)
    plt.ylabel('Tiempo (ms)', fontsize=12)
    plt.title('Escalabilidad del Forward Pass', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Complejidad: O(n³) para multiplicación matricial")
    print("   (batch_size × n_entradas × n_neuronas)")
analizar_escalabilidad()
```
## 🎯 EJERCICIOS PROPUESTOS
### Ejercicio 1: Implementación Completa (Básico)
**Objetivo**: Consolidar comprensión de neuronas individuales.
**Tareas**:
1. Implementa una clase `NeuronaSimple` sin usar NumPy (solo Python puro)
2. Añade métodos para:
   - Calcular salida
   - Obtener/establecer pesos
   - Imprimir información
3. Crea tests unitarios que verifiquen:
   - Salida correcta
   - Manejo de diferentes tamaños de entrada
   - Casos extremos (entrada vacía, pesos cero, etc.)
```python
# Esqueleto
class NeuronaSimple:
    def __init__(self, n_entradas):
        # Tu código aquí
        pass
    
    def calcular(self, entradas):
        # Tu código aquí
        pass
    
    def set_pesos(self, pesos, bias):
        # Tu código aquí
        pass
```
### Ejercicio 2: Capa con Activación (Intermedio)
**Objetivo**: Preparar el camino para funciones de activación.
**Tareas**:
1. Extiende `CapaDensaCompleta` para incluir una función de activación
2. Implementa tres activaciones:
   - Lineal (identidad): f(x) = x
   - Step (escalón): f(x) = 1 if x > 0 else 0
   - Sign (signo): f(x) = 1 if x > 0 else -1
3. Visualiza cómo cada activación afecta las salidas
```python
class CapaConActivacion(CapaDensaCompleta):
    def __init__(self, n_entradas, n_neuronas, activacion='lineal'):
        super().__init__(n_entradas, n_neuronas)
        self.activacion = activacion
    
    def forward(self, entradas):
        # Tu código aquí
        pass
```
### Ejercicio 3: Red de Dos Capas (Avanzado)
**Objetivo**: Combinar múltiples capas.
**Tareas**:
1. Crea una clase `RedNeuronalSimple` que conecte dos capas
2. Implementa forward pass a través de ambas capas
3. Visualiza cómo se transforman los datos en cada capa
4. Prueba con el problema XOR (aunque no lo resolverá perfectamente sin entrenamiento)
```python
class RedNeuronalSimple:
    def __init__(self, arquitectura):
        """
        Args:
            arquitectura: lista [n_entradas, n_ocultas, n_salidas]
        """
        # Tu código aquí
        pass
    
    def forward(self, X):
        # Tu código aquí
        pass
```
### Ejercicio 4: Análisis de Sensibilidad (Desafío)
**Objetivo**: Entender cómo diferentes parámetros afectan el comportamiento.
**Tareas**:
1. Crea una función que analice cómo cambios en pesos afectan salidas
2. Visualiza:
   - Gradiente de salida respecto a cada peso
   - Sensibilidad a cambios en el bias
   - Rango efectivo de operación
3. Identifica pesos "muertos" (que no afectan la salida)
### Ejercicio 5: Mini Proyecto Integrador (Proyecto)
**Objetivo**: Aplicar todo lo aprendido.
**Proyecto**: Sistema de recomendación simple
- **Entrada**: Features de un usuario [edad, género, intereses]
- **Salida**: Probabilidad de que le guste un producto
- **Tareas**:
  1. Generar datos sintéticos
  2. Crear y configurar una capa de neuronas
  3. Experimentar con diferentes pesos (manual)
  4. Visualizar resultados
  5. Documentar hallazgos
## 📝 Entregables
Para completar este laboratorio, debes entregar:
### 1. Código Implementado (60%)
- Archivo `neurona.py` con todas las implementaciones
- Todas las funciones deben tener docstrings
- Código limpio y comentado
- Tests incluidos
### 2. Notebook de Experimentación (25%)
- `experimentos.ipynb` con:
  - Todas las actividades completadas
  - Visualizaciones claras y etiquetadas
  - Análisis de resultados
  - Respuestas a preguntas de reflexión
### 3. Reporte Técnico (15%)
- Documento PDF (2-3 páginas) que incluya:
  - Resumen de conceptos aprendidos
  - Resultados de experimentos clave
  - Análisis de limitaciones encontradas
  - Reflexiones personales
  - Ideas para mejoras futuras
### Formato de Entrega
```
Lab01_NombreApellido/
├── codigo/
│   ├── neurona.py
│   ├── capa.py
│   └── tests.py
├── notebooks/
│   ├── experimentos.ipynb
│   └── visualizaciones.ipynb
├── reporte/
│   └── reporte_lab01.pdf
└── README.md
```
## 🎯 Criterios de Evaluación (CDIO)
### Concebir (25%)
- ✅ Comprende el concepto de neurona artificial
- ✅ Identifica componentes y su función
- ✅ Reconoce limitaciones y capacidades
- ✅ Propone aplicaciones apropiadas
**Evidencia**: Respuestas a preguntas de reflexión, diseño de experimentos
### Diseñar (25%)
- ✅ Diseña arquitecturas de capas apropiadas
- ✅ Elige inicializaciones correctas
- ✅ Planifica experimentos significativos
- ✅ Considera casos extremos
**Evidencia**: Código de ejercicios, notebook de experimentos
### Implementar (30%)
- ✅ Implementa neuronas correctamente
- ✅ Usa NumPy eficientemente
- ✅ Código limpio y documentado
- ✅ Maneja errores apropiadamente
**Evidencia**: Código fuente, tests pasados
### Operar (20%)
- ✅ Ejecuta experimentos sistemáticamente
- ✅ Analiza resultados críticamente
- ✅ Documenta hallazgos claramente
- ✅ Propone mejoras basadas en observaciones
**Evidencia**: Reporte técnico, visualizaciones, análisis
### Rúbrica Detallada
| Criterio | Excelente (100%) | Bueno (80%) | Aceptable (60%) | Insuficiente (<60%) |
|----------|------------------|-------------|-----------------|---------------------|
| **Implementación** | Código perfecto, eficiente, documentado | Funciona bien, minor issues | Funciona parcialmente | No funciona |
| **Experimentación** | Experimentos completos y creativos | Todos requeridos completados | Algunos experimentos faltantes | Mínimo o nada |
| **Análisis** | Profundo, crítico, insightful | Completo y correcto | Superficial | Ausente o incorrecto |
| **Documentación** | Excelente, clara, profesional | Buena, entendible | Básica | Pobre o ausente |
## 📚 Referencias Adicionales
### Libros
1. **Nielsen, M.** (2015). "Neural Networks and Deep Learning"
   - Capítulo 1: Usando redes neuronales para reconocer dígitos escritos
   - Disponible gratuitamente online
2. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). "Deep Learning"
   - Capítulo 6: Perceptrones multicapa
   - http://www.deeplearningbook.org
3. **Harrison, M.** (2019). "Neural Networks from Scratch in Python"
   - Capítulos 1-2: Neuronas y capas
   - Énfasis en implementación desde cero
### Artículos Académicos
1. **McCulloch, W.S., & Pitts, W.** (1943). "A logical calculus of the ideas immanent in nervous activity"
   - Paper original sobre neuronas artificiales
2. **Rosenblatt, F.** (1958). "The perceptron: A probabilistic model for information storage"
   - Introducción del perceptrón
### Recursos Online
1. **3Blue1Brown** - "Neural Networks" series
   - https://www.youtube.com/watch?v=aircAruvnKk
   - Excelentes visualizaciones
2. **Stanford CS231n** - Convolutional Neural Networks
   - http://cs231n.stanford.edu
   - Notas de clase sobre neuronas y capas
3. **Distill.pub** - Artículos interactivos sobre ML
   - https://distill.pub
   - Visualizaciones de alta calidad
### Tutoriales Interactivos
1. **TensorFlow Playground**
   - https://playground.tensorflow.org
   - Experimenta con redes neuronales en el navegador
2. **Neural Network Playground**
   - https://experiments.withgoogle.com/ai/neural-network-playground
   - Visualización interactiva
### Documentación Técnica
1. **NumPy Documentation**
   - https://numpy.org/doc/
   - Referencia completa de NumPy
2. **Python Scientific Lecture Notes**
   - https://scipy-lectures.org
   - Guía de stack científico de Python
## 🎓 Notas Finales
### Conceptos Clave para Recordar
1. **Neurona Artificial = Suma Ponderada + Bias**
   - Operación fundamental: producto punto
   - Parámetros aprendibles: pesos y bias
2. **NumPy es Esencial**
   - Vectorización = Eficiencia
   - ~100x más rápido que Python puro
   - Estándar en la industria
3. **Procesamiento en Batch**
   - Nunca procesar muestras individuales
   - Aprovechar paralelismo
   - Más eficiente en GPU
4. **Inicialización Importa**
   - Romper simetría con valores aleatorios
   - Valores pequeños previenen explosión
   - Xavier/He para redes profundas
5. **Limitaciones de Neuronas Individuales**
   - Solo problemas linealmente separables
   - XOR es imposible con una neurona
   - Necesitamos redes para complejidad
### Preparación para el Siguiente Lab
En **Lab 02: Primera Red Neuronal**, aprenderemos:
- Conectar múltiples capas de neuronas
- Forward propagation a través de redes profundas
- Arquitecturas comunes
- El problema de la linealidad sin activaciones
**Para prepararte**:
1. Asegúrate de entender completamente el producto punto matricial
2. Practica rastrear dimensiones de tensores
3. Revisa multiplicación de matrices
4. Piensa en cómo combinarías múltiples neuronas
### Consejos de Estudio
1. **Implementa desde cero**: No uses librerías de alto nivel todavía
2. **Visualiza todo**: Gráficas ayudan a entender
3. **Experimenta libremente**: Cambia parámetros y observa efectos
4. **Documenta hallazgos**: Toma notas de observaciones
5. **Haz preguntas**: Si algo no es claro, investiga más
### Solución de Problemas Comunes
**Problema**: Salidas muy grandes o muy pequeñas
- **Causa**: Inicialización inadecuada
- **Solución**: Usar inicialización Xavier/He
**Problema**: Todas las salidas iguales
- **Causa**: Pesos todos iguales (simetría)
- **Solución**: Inicialización aleatoria
**Problema**: Código muy lento
- **Causa**: Usando loops en lugar de NumPy
- **Solución**: Vectorizar operaciones
**Problema**: Errores de dimensiones
- **Causa**: Shapes incompatibles en multiplicación
- **Solución**: Verificar dimensiones con `.shape`
### Comunidad y Soporte
- **Foro del curso**: Para preguntas técnicas
- **Horas de oficina**: Consultas personalizadas
- **Grupo de estudio**: Aprender con compañeros
- **Stack Overflow**: Para errores de código
### Certificación de Completitud
Has completado exitosamente el Lab 01 cuando puedes:
- [ ] Explicar qué es una neurona artificial a un principiante
- [ ] Implementar una neurona desde cero usando solo NumPy
- [ ] Crear capas de múltiples neuronas
- [ ] Procesar batches de datos eficientemente
- [ ] Entender las limitaciones de neuronas individuales
- [ ] Visualizar y analizar comportamiento de neuronas
- [ ] Aplicar neuronas a problemas sencillos de clasificación
---
**¡Felicitaciones por completar el Lab 01! Has dado el primer paso fundamental en tu viaje por el Deep Learning. 🎉**
**Siguiente parada**: Lab 02 - Primera Red Neuronal 🚀
---
*Última actualización: 2024*  
*Versión: 1.0*  
*Licencia: MIT - Uso educativo*
