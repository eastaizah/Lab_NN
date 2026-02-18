# Guía de Laboratorio: Primera Red Neuronal Multicapa

## 📋 Información del Laboratorio

**Título:** Primera Red Neuronal Multicapa  
**Código:** Lab 02  
**Duración:** 2-3 horas  
**Nivel:** Básico-Intermedio  

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender arquitectura de redes neuronales multicapa
2. Implementar forward propagation desde cero
3. Diseñar arquitecturas para diferentes problemas
4. Calcular número de parámetros en una red
5. Entender flujo de datos a través de capas
6. Implementar redes usando programación orientada a objetos
7. Visualizar activaciones y transformaciones
8. Reconocer limitaciones sin funciones de activación
9. Aplicar buenas prácticas de inicialización

## 📚 Prerrequisitos

### Conocimientos

- Completar Lab 01
- Python intermedio (clases, funciones, NumPy)
- Álgebra lineal básica
- Comprensión de conceptos de labs anteriores

### Software

- Python 3.8+
- NumPy 1.19+
- Matplotlib 3.0+
- Jupyter Notebook (recomendado)

### Material de Lectura

Antes de comenzar, lee:
- `teoria.md` - Marco teórico completo
- `README.md` - Visión general del laboratorio

## 📖 Introducción

En Lab 01 aprendimos sobre **neuronas individuales**. Ahora conectaremos múltiples neuronas en **capas** y múltiples capas en **redes neuronales** completas.

### Contexto del Problema

Las neuronas individuales tienen limitaciones:
- Solo patrones linealmente separables
- No pueden resolver XOR
- Capacidad representacional limitada

Para problemas reales necesitamos **redes neuronales profundas** con múltiples capas.

### Enfoque con Redes Neuronales

```
ENTRADA → [CAPA OCULTA 1] → [CAPA OCULTA 2] → [SALIDA]
```

Arquitectura típica para MNIST:
```
[784 entradas] → [128 neuronas] → [64 neuronas] → [10 salidas]
```

### Conceptos Fundamentales

**Forward Propagation:** Cálculo secuencial de salidas capa por capa.

**Dimensiones:** Crucial entender (batch, features) @ (features, neurons) = (batch, neurons)

**Parámetros:** Para cada capa = (n_entradas × n_neuronas) + n_neuronas

### Aplicaciones Prácticas

- Visión por computadora (ResNet, VGG)
- PLN (BERT, GPT)
- Sistemas de recomendación
- Diagnóstico médico
- Trading algorítmico

### Motivación Histórica

De perceptrones simples (1958) a redes profundas modernas con billones de parámetros.

## 🔬 Parte 1: Construyendo Tu Primera Red (45 min)

### 1.1 Dos Capas Conectadas

```python
import numpy as np

# Arquitectura: 3 → 4 → 2
X = np.array([[1, 2, 3]])
W1 = np.random.randn(3, 4) * 0.01
b1 = np.zeros(4)
a1 = X @ W1 + b1

W2 = np.random.randn(4, 2) * 0.01
b2 = np.zeros(2)
salida = a1 @ W2 + b2

print(f"Salida: {salida.shape}")  # (1, 2)
```

### 1.2 Red Completa para MNIST

```python
# Red [784, 128, 64, 10]
X = np.random.randn(32, 784)  # batch de 32 imágenes

W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros(128)
a1 = X @ W1 + b1  # (32, 128)

W2 = np.random.randn(128, 64) * 0.01
b2 = np.zeros(64)
a2 = a1 @ W2 + b2  # (32, 64)

W3 = np.random.randn(64, 10) * 0.01
b3 = np.zeros(10)
salida = a2 @ W3 + b3  # (32, 10)
```

### Actividades

1. Crear red [10, 20, 15, 5] y verificar dimensiones
2. Calcular parámetros de [784, 256, 128, 10]
3. Experimentar con diferentes batch sizes

## 🔬 Parte 2: Programación Orientada a Objetos (45 min)

### 2.1 Clase CapaDensa

```python
class CapaDensa:
    def __init__(self, n_entradas, n_neuronas):
        self.pesos = np.random.randn(n_entradas, n_neuronas) * 0.01
        self.biases = np.zeros(n_neuronas)
    
    def forward(self, entradas):
        self.salida = entradas @ self.pesos + self.biases
        return self.salida
```

### 2.2 Clase RedNeuronal

```python
class RedNeuronal:
    def __init__(self, arquitectura):
        self.capas = []
        for i in range(len(arquitectura) - 1):
            capa = CapaDensa(arquitectura[i], arquitectura[i+1])
            self.capas.append(capa)
    
    def forward(self, X):
        activacion = X
        for capa in self.capas:
            activacion = capa.forward(activacion)
        return activacion
```

### Actividades

1. Implementar método contar_parametros()
2. Agregar método resumen()
3. Visualizar activaciones por capa

## 🔬 Parte 3: Diseño de Arquitecturas (40 min)

### 3.1 Arquitecturas para Problemas Diferentes

**Clasificación Binaria:**
```python
red_spam = RedNeuronal([5000, 256, 64, 1])
```

**Clasificación Multiclase:**
```python
red_mnist = RedNeuronal([784, 512, 256, 128, 10])
```

**Regresión:**
```python
red_precios = RedNeuronal([20, 64, 32, 16, 1])
```

### 3.2 Profundas vs Anchas

```python
profunda = RedNeuronal([100, 80, 60, 40, 20, 10])
ancha = RedNeuronal([100, 500, 10])
```

### 3.3 Limitación Sin Activación

```python
# Demostración: Red de 2 capas = Red de 1 capa
h1 = X @ W1 + b1
salida = h1 @ W2 + b2

# Equivalente:
W_combinado = W1 @ W2
b_combinado = b1 @ W2 + b2
salida_equivalente = X @ W_combinado + b_combinado
```

**Conclusión:** Sin activación no lineal, red profunda = red de 1 capa

### Actividades

1. Diseñar red para 50 clases con 10 características
2. Comparar número de parámetros de diferentes arquitecturas
3. Demostrar equivalencia matemática

## 🔬 Parte 4: Aplicaciones Prácticas (40 min)

### 4.1 Dataset Sintético

```python
def generar_datos(n=1000, features=20, clases=5):
    X = np.random.randn(n, features)
    y = np.random.randint(0, clases, n)
    return X, y

X, y = generar_datos()
red = RedNeuronal([20, 64, 32, 5])
predicciones = red.forward(X)
```

### 4.2 Comparación Batch Sizes

```python
import time
for batch_size in [1, 10, 50, 100]:
    start = time.time()
    _ = red.forward(X[:batch_size])
    print(f"Batch {batch_size}: {time.time()-start:.4f}s")
```

### 4.3 Estadísticas de Activaciones

```python
def analizar_activaciones(red, X):
    _ = red.forward(X)
    for i, capa in enumerate(red.capas):
        print(f"Capa {i}: mean={capa.salida.mean():.4f}, "
              f"std={capa.salida.std():.4f}")
```

### Actividades

1. Generar datos sintéticos y hacer predicciones
2. Medir throughput con diferentes batch sizes
3. Analizar distribución de activaciones

## 📊 Análisis Final de Rendimiento

### Comparación de Implementaciones

En esta sección compararás diferentes enfoques de implementación para entender las ventajas de cada uno.

**Criterios de comparación:**
- Velocidad de ejecución
- Uso de memoria
- Claridad del código
- Mantenibilidad

### Métricas de Desempeño

Mide y compara:
- Tiempo de forward pass
- Escalabilidad con tamaño de datos
- Eficiencia computacional

## 🎯 EJERCICIOS PROPUESTOS

### Ejercicio 1: Diseño de Arquitectura (Básico)

Diseña red para diagnóstico médico:
- 50 características de entrada
- 3 diagnósticos posibles
- Calcula parámetros totales

### Ejercicio 2: Comparación de Arquitecturas (Intermedio)

Para MNIST, diseña:
- Red poco profunda (2 capas ocultas)
- Red profunda (5+ capas ocultas)
- Ambas con ~100,000 parámetros

Compara tiempo de forward pass y distribución de parámetros.

### Ejercicio 3: Visualización Avanzada (Intermedio)

Implementa:
- Visualización de arquitectura como gráfico
- Heatmap de pesos por capa
- Distribución de activaciones

### Ejercicio 4: Optimización de Inicialización (Avanzado)

Compara:
- Todos ceros
- Random pequeño
- Xavier
- He

Analiza varianza de activaciones en red profunda.

### Ejercicio 5: Mini Framework (Proyecto)

Crea framework con:
- Clase Layer base
- DenseLayer, ActivationLayer
- Sequential para encadenar
- Métodos add(), forward(), summary()
- Guardado/carga de parámetros

## 📝 Entregables

### 1. Código Implementado (60%)

**Requisitos mínimos:**
- Implementaciones completas y funcionales
- Código limpio y bien documentado
- Pruebas y validación
- Manejo apropiado de errores

### 2. Notebook de Experimentación (25%)

**Debe incluir:**
- Experimentos con diferentes configuraciones
- Visualizaciones claras
- Análisis de resultados
- Comparaciones y conclusiones

### 3. Reporte Técnico (15%)

**Secciones:**
1. Introducción y objetivos
2. Metodología
3. Resultados experimentales
4. Análisis y discusión
5. Conclusiones

**Extensión:** 3-5 páginas

### Formato de Entrega

```
Lab02_Entrega/
├── codigo/
│   └── [archivos .py]
├── notebooks/
│   └── experimentos.ipynb
├── reporte/
│   └── reporte_lab02.pdf
└── README.md
```

## 🎯 Criterios de Evaluación (CDIO)

### Concebir (25%)

**Comprender el problema:**
- Identificar requisitos y restricciones
- Analizar alternativas de solución
- Reconocer implicaciones de decisiones de diseño

### Diseñar (25%)

**Planificar soluciones:**
- Diseñar arquitecturas apropiadas
- Estructurar código eficientemente
- Considerar escalabilidad y mantenibilidad

### Implementar (30%)

**Construcción:**
- Código funcional y correcto
- Implementación eficiente
- Documentación adecuada
- Pruebas comprehensivas

### Operar (20%)

**Validación y análisis:**
- Experimentación sistemática
- Análisis crítico de resultados
- Visualizaciones informativas
- Conclusiones fundamentadas

### Rúbrica Detallada

| Criterio | Excelente (90-100%) | Bueno (75-89%) | Satisfactorio (60-74%) | Insuficiente (<60%) |
|----------|-------------------|---------------|---------------------|-------------------|
| **Implementación** | Impecable, eficiente, documentado | Funcional con docs | Básico funcional | Con errores |
| **Experimentación** | Análisis profundo | Completo | Básico | Incompleto |
| **Documentación** | Excelente | Buena | Básica | Pobre |
| **Comprensión** | Dominio total | Buen entendimiento | Comprensión básica | Comprensión limitada |

## 📚 Referencias Adicionales

### Libros

1. **"Deep Learning" - Goodfellow, Bengio, Courville**
   - Capítulos relevantes para este lab
   - www.deeplearningbook.org

2. **"Neural Networks and Deep Learning" - Michael Nielsen**
   - neuralnetworksanddeeplearning.com

### Recursos Online

1. **CS231n: Stanford**
   - http://cs231n.stanford.edu/

2. **3Blue1Brown: Neural Networks**
   - Videos educativos excelentes

3. **TensorFlow Playground**
   - https://playground.tensorflow.org/

### Documentación

- NumPy: https://numpy.org/doc/
- Matplotlib: https://matplotlib.org/
- Python: https://docs.python.org/3/

## 🎓 Notas Finales

### Conceptos Clave para Recordar

1. **Arquitectura:** Capas de entrada, ocultas y salida
2. **Forward Propagation:** Cálculo secuencial capa por capa
3. **Dimensiones:** (batch, features) @ (features, neurons) = (batch, neurons)
4. **Parámetros:** Pesos y biases aprendibles
5. **Inicialización:** Nunca ceros, usar random/Xavier/He
6. **Limitación:** Sin activación, red profunda = 1 capa
7. **Diseño:** Balance profundidad vs anchura
8. **Eficiencia:** Batch processing y vectorización

### Preparación para el Siguiente Lab

**Lab 03: Funciones de Activación**

Aprenderás:
- ReLU, Sigmoid, Tanh, Softmax
- Por qué son necesarias
- Cómo implementarlas
- Derivadas para backpropagation

Prepárate repasando cálculo diferencial básico.

### Consejos de Estudio

1. **Implementa desde cero** - No uses frameworks todavía
2. **Visualiza** - Dibuja y grafica para entender
3. **Experimenta** - Prueba diferentes configuraciones
4. **Debug sistemáticamente** - Verifica paso a paso
5. **Documenta** - Anota hallazgos y experimentos

### Solución de Problemas Comunes

**Errores de dimensiones:**
- Verifica shape de todas las matrices
- Usa print(variable.shape) liberalmente

**Resultados inesperados:**
- Verifica inicialización
- Asegura reproducibilidad con seed
- Revisa cada paso del cálculo

**Código lento:**
- Usa vectorización de NumPy
- Evita loops innecesarios
- Procesa en batches

### Certificación de Completitud

Has completado exitosamente Lab 02 cuando puedas:

- [ ] Comprender arquitectura de redes neuronales multicapa
- [ ] Implementar forward propagation desde cero
- [ ] Diseñar arquitecturas para diferentes problemas
- [ ] Calcular número de parámetros en una red
- [ ] Entender flujo de datos a través de capas
- [ ] Implementar redes usando programación orientada a objetos
- [ ] Visualizar activaciones y transformaciones
- [ ] Reconocer limitaciones sin funciones de activación
- [ ] Aplicar buenas prácticas de inicialización

**¡Felicitaciones!** Continúa con el siguiente laboratorio.

---

**¿Preguntas?** Revisa teoría, experimenta, y consulta referencias.

**¡Éxito en tu aprendizaje! 🚀**
