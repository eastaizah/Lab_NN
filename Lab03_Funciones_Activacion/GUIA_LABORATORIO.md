# Guía de Laboratorio: Funciones de Activación

## 📋 Información del Laboratorio

**Título:** Funciones de Activación  
**Código:** Lab 03  
**Duración:** 2-3 horas  
**Nivel:** Básico-Intermedio  

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender el rol de funciones de activación
2. Implementar ReLU, Sigmoid, Tanh, Softmax desde cero
3. Calcular derivadas para backpropagation
4. Visualizar y comparar diferentes activaciones
5. Elegir activación apropiada para cada problema
6. Reconocer problema del gradiente que desaparece
7. Evitar neuronas muertas en ReLU
8. Implementar activaciones eficientemente
9. Entender no-linealidad en redes profundas

## 📚 Prerrequisitos

### Conocimientos

- Completar Lab 01-02
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

Las **funciones de activación** introducen **no-linealidad** en redes neuronales. Sin ellas, cualquier red profunda es equivalente a regresión lineal.

### Contexto del Problema

En Lab 02 vimos que sin activación, una red profunda colapsa a una sola capa lineal. Para aprender patrones complejos necesitamos no-linealidad.

### Funciones de Activación

Transforman la salida de cada neurona agregando capacidad de modelar relaciones complejas:

```
Entrada → Suma Ponderada → ACTIVACIÓN → Salida
```

### Conceptos Fundamentales

**1. No-linealidad:** Permite aprender XOR, círculos, patrones complejos

**2. Principales funciones:**
- **ReLU:** max(0, x) - Estándar para capas ocultas
- **Sigmoid:** 1/(1+e^-x) - Clasificación binaria
- **Tanh:** tanh(x) - Centrada en cero
- **Softmax:** Clasificación multiclase

**3. Derivadas:** Necesarias para backpropagation

### Aplicaciones

Cada activación tiene su uso ideal:
- Capas ocultas → ReLU
- Salida binaria → Sigmoid
- Salida multiclase → Softmax

## 🔬 Parte 1: Implementación de Activaciones (45 min)

### 1.1 ReLU (Rectified Linear Unit)

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivada(x):
    return (x > 0).astype(float)

# Prueba
x = np.array([-2, -1, 0, 1, 2])
print(f"ReLU: {relu(x)}")  # [0, 0, 0, 1, 2]
print(f"Derivada: {relu_derivada(x)}")  # [0, 0, 0, 1, 1]
```

### 1.2 Sigmoid

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    s = sigmoid(x)
    return s * (1 - s)

# Prueba
x = np.array([-2, 0, 2])
print(f"Sigmoid: {sigmoid(x)}")  # [0.12, 0.5, 0.88]
```

### 1.3 Tanh

```python
def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1 - np.tanh(x)**2

# Prueba
x = np.array([-1, 0, 1])
print(f"Tanh: {tanh(x)}")  # [-0.76, 0, 0.76]
```

### 1.4 Softmax

```python
def softmax(x):
    # Estabilización numérica
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Prueba
x = np.array([[1, 2, 3]])
probs = softmax(x)
print(f"Softmax: {probs}")  # [[0.09, 0.24, 0.67]]
print(f"Suma: {probs.sum()}")  # 1.0
```

### Actividades

1. Implementar Leaky ReLU
2. Graficar todas las funciones
3. Verificar derivadas numéricamente

## 🔬 Parte 2: Integración con Redes (45 min)

### 2.1 Clase Activación

```python
class Activacion:
    def __init__(self, funcion, derivada):
        self.funcion = funcion
        self.derivada = derivada
    
    def forward(self, entradas):
        self.entradas = entradas
        self.salida = self.funcion(entradas)
        return self.salida
    
    def backward(self, grad_salida):
        return grad_salida * self.derivada(self.entradas)
```

### 2.2 Red con Activaciones

```python
class RedConActivaciones:
    def __init__(self, arquitectura, activaciones):
        self.capas = []
        for i in range(len(arquitectura)-1):
            self.capas.append(CapaDensa(arquitectura[i], arquitectura[i+1]))
            if i < len(activaciones):
                self.capas.append(Activacion(activaciones[i], None))
    
    def forward(self, X):
        activacion = X
        for capa in self.capas:
            activacion = capa.forward(activacion)
        return activacion
```

### 2.3 Ejemplo de Uso

```python
# Red para clasificación binaria
red = RedConActivaciones(
    arquitectura=[10, 20, 15, 1],
    activaciones=[relu, relu, sigmoid]
)

X = np.random.randn(32, 10)
output = red.forward(X)
print(f"Output shape: {output.shape}")  # (32, 1)
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")  # [0, 1]
```

### Actividades

1. Crear red para MNIST con ReLU en ocultas y Softmax en salida
2. Comparar salidas con/sin activación
3. Medir impacto en tiempo de ejecución

## 🔬 Parte 3: Visualización y Análisis (40 min)

### 3.1 Graficar Funciones

```python
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.legend()
plt.title('Funciones de Activación')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, relu_derivada(x), label='ReLU'')
plt.plot(x, sigmoid_derivada(x), label='Sigmoid'')
plt.plot(x, tanh_derivada(x), label='Tanh'')
plt.legend()
plt.title('Derivadas')
plt.grid(True)

plt.subplot(1, 3, 3)
# Comparar saturación
plt.plot(x, sigmoid_derivada(x), label='Sigmoid' (satura)')
plt.plot(x, relu_derivada(x), label='ReLU' (no satura)')
plt.legend()
plt.title('Problema de Saturación')
plt.grid(True)

plt.tight_layout()
plt.savefig('activaciones.png')
```

### 3.2 Problema del Gradiente que Desaparece

```python
def demostrar_gradiente_desaparece():
    x = np.array([10.0])  # Valor grande
    
    # Sigmoid satura
    for i in range(10):
        grad = sigmoid_derivada(x)
        print(f"Capa {i}: gradiente = {grad[0]:.10f}")
        x = sigmoid(x)  # Propagar
    
    # ReLU no satura
    x = np.array([10.0])
    for i in range(10):
        grad = relu_derivada(x)
        print(f"Capa {i}: gradiente = {grad[0]}")
        x = relu(x)
```

### 3.3 Neuronas Muertas en ReLU

```python
def detectar_neuronas_muertas(red, X):
    _ = red.forward(X)
    for i, capa in enumerate(red.capas):
        if hasattr(capa, 'salida'):
            muertas = (capa.salida <= 0).all(axis=0).sum()
            total = capa.salida.shape[1]
            print(f"Capa {i}: {muertas}/{total} neuronas muertas")
```

### Actividades

1. Visualizar todas las funciones y derivadas
2. Demostrar saturación de gradiente
3. Detectar neuronas muertas en red con ReLU

## 🔬 Parte 4: Casos de Uso (40 min)

### 4.1 Clasificación Binaria

```python
# Spam detection
red_spam = RedConActivaciones(
    [100, 64, 32, 1],
    [relu, relu, sigmoid]
)
```

### 4.2 Clasificación Multiclase

```python
# MNIST
red_mnist = RedConActivaciones(
    [784, 256, 128, 10],
    [relu, relu, softmax]
)
```

### 4.3 Regresión

```python
# Predicción de precios (sin activación en salida)
red_regresion = RedConActivaciones(
    [20, 64, 32, 1],
    [relu, relu, lambda x: x]  # Identidad en salida
)
```

### 4.4 Comparación Experimental

```python
def comparar_activaciones():
    X = np.random.randn(100, 10)
    
    configs = [
        ('Solo Sigmoid', [sigmoid] * 3),
        ('Solo ReLU', [relu] * 3),
        ('Mixto', [relu, relu, sigmoid])
    ]
    
    for nombre, acts in configs:
        red = RedConActivaciones([10, 20, 15, 5], acts)
        salida = red.forward(X)
        print(f"{nombre}: mean={salida.mean():.3f}, std={salida.std():.3f}")
```

### Actividades

1. Implementar red para cada tipo de problema
2. Comparar diferentes combinaciones de activaciones
3. Medir impacto en distribución de salidas

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

## �� EJERCICIOS PROPUESTOS

### Ejercicio 1: Implementar ELU (Básico)

```python
ELU(x) = x si x > 0
       = α(e^x - 1) si x <= 0
```

Implementa forward y backward.

### Ejercicio 2: Análisis de Saturación (Intermedio)

Grafica derivadas de Sigmoid y Tanh para x en [-10, 10].
¿En qué rangos se saturan?

### Ejercicio 3: Red con Diferentes Activaciones (Intermedio)

Entrena red simple con:
- Solo Sigmoid
- Solo ReLU  
- Mezcla

Compara velocidad de convergencia.

### Ejercicio 4: Softmax con Temperatura (Avanzado)

```python
Softmax(x/T)  donde T = temperatura
```

Observa cómo T afecta distribución de probabilidades.

### Ejercicio 5: Detección de Problemas (Avanzado)

Implementa:
- Detector de gradientes que desaparecen
- Detector de neuronas muertas
- Recomendador de activación

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
Lab03_Entrega/
├── codigo/
│   └── [archivos .py]
├── notebooks/
│   └── experimentos.ipynb
├── reporte/
│   └── reporte_lab03.pdf
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

1. **No-linealidad es esencial:** Sin activación, red = regresión lineal
2. **ReLU es el estándar:** Simple, eficiente, efectiva
3. **Softmax para multiclase:** Convierte scores a probabilidades
4. **Sigmoid para binaria:** Salida entre 0 y 1
5. **Gradientes importan:** Evitar saturación
6. **Neuronas muertas:** Problema de ReLU con inicialización mala
7. **Derivadas simples:** ReLU' = 1 si x>0, 0 si no
8. **Combinaciones ideales:** ReLU+ReLU+Softmax para clasificación

### Preparación para el Siguiente Lab

**Lab 04: Funciones de Pérdida**

Aprenderás:
- MSE, MAE para regresión
- Cross-Entropy para clasificación
- Gradient descent
- Learning rate
- Overfitting

Prepárate para entender optimización.

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

Has completado exitosamente Lab 03 cuando puedas:

- [ ] Comprender el rol de funciones de activación
- [ ] Implementar ReLU, Sigmoid, Tanh, Softmax desde cero
- [ ] Calcular derivadas para backpropagation
- [ ] Visualizar y comparar diferentes activaciones
- [ ] Elegir activación apropiada para cada problema
- [ ] Reconocer problema del gradiente que desaparece
- [ ] Evitar neuronas muertas en ReLU
- [ ] Implementar activaciones eficientemente
- [ ] Entender no-linealidad en redes profundas

**¡Felicitaciones!** Continúa con el siguiente laboratorio.

---

**¿Preguntas?** Revisa teoría, experimenta, y consulta referencias.

**¡Éxito en tu aprendizaje! 🚀**
