# Lab 03: Funciones de Activación

## Objetivos de Aprendizaje

Al completar este laboratorio, serás capaz de:

1. Comprender el rol de las funciones de activación en redes neuronales
2. Implementar desde cero las funciones de activación más importantes
3. Calcular derivadas para backpropagation
4. Visualizar y comparar diferentes funciones de activación
5. Elegir la función de activación apropiada para diferentes problemas

## Estructura del Laboratorio

```
Lab03_Funciones_Activacion/
├── README.md                 # Esta guía
├── teoria.md                 # Fundamentos teóricos
├── practica.ipynb           # Notebook interactivo
└── codigo/
    └── activaciones.py      # Implementaciones completas
```

## Requisitos Previos

- Completar Lab 01 y Lab 02
- Comprensión básica de derivadas
- Familiaridad con NumPy

## Contenido Teórico

El archivo `teoria.md` cubre:

- **Introducción a funciones de activación**: Por qué son necesarias
- **Sigmoid**: Ecuación, derivada, ventajas y desventajas
- **Tanh**: Función hiperbólica tangente
- **ReLU**: La función más popular en deep learning
- **Leaky ReLU**: Solución al problema de neuronas muertas
- **Softmax**: Para clasificación multiclase
- **Comparaciones**: Cuándo usar cada una
- **Problema del gradiente que desaparece**: Cómo evitarlo

## Práctica

### Parte 1: Implementación Básica (30 min)

Implementa las funciones de activación desde cero:

```python
# Ejecutar el código principal
cd codigo/
python activaciones.py
```

Esto generará:
- Visualizaciones de funciones y derivadas
- Comparación de saturación de gradientes
- Ejemplos de Softmax
- Verificación de gradientes numéricos

### Parte 2: Notebook Interactivo (45 min)

Abre `practica.ipynb` y completa los ejercicios:

1. **Visualización**: Graficar funciones de activación
2. **Derivadas**: Calcular y verificar derivadas
3. **Experimentos**: Comparar comportamiento en redes
4. **Casos de uso**: Ejercicios prácticos

```bash
jupyter notebook practica.ipynb
```

### Parte 3: Experimentos (30 min)

1. **Experimento 1**: Comparar ReLU vs Sigmoid en una red profunda
2. **Experimento 2**: Observar el problema del gradiente que desaparece
3. **Experimento 3**: Evaluar el problema de neuronas muertas

## Conceptos Clave

### 1. No Linealidad

Sin funciones de activación, una red neuronal profunda es equivalente a una regresión lineal:

```
Red sin activación:  y = W3 * W2 * W1 * x = W_combinado * x
Red con activación:  y = σ(W3 * σ(W2 * σ(W1 * x)))
```

### 2. Elección de Activación

| Capa | Problema | Función Recomendada |
|------|----------|---------------------|
| Oculta | General | ReLU |
| Oculta | Neuronas muertas | Leaky ReLU |
| Salida | Clasificación binaria | Sigmoid |
| Salida | Clasificación multiclase | Softmax |
| Salida | Regresión | Lineal |

### 3. Gradientes

Las derivadas son cruciales para backpropagation:

```python
# ReLU es simple:
df/dx = 1 si x > 0, 0 si x <= 0

# Sigmoid es más compleja:
df/dx = f(x) * (1 - f(x))
```

## Ejercicios

### Ejercicio 1: Implementar ELU

Implementa la función ELU (Exponential Linear Unit):

```python
ELU(x) = x si x > 0
       = α(e^x - 1) si x <= 0
```

### Ejercicio 2: Análisis de Saturación

Grafica las derivadas de Sigmoid y Tanh para x en [-10, 10]. ¿En qué rangos se saturan?

### Ejercicio 3: Red con Diferentes Activaciones

Crea una red simple y entrénala con:
- Solo Sigmoid
- Solo ReLU
- Mezcla de ambas

Compara la velocidad de convergencia.

### Ejercicio 4: Softmax Temperature

Implementa Softmax con temperatura:

```python
Softmax(x/T) donde T es la temperatura
```

Observa cómo T afecta la distribución de probabilidades.

## Preguntas de Reflexión

1. **¿Por qué ReLU es tan efectiva a pesar de su simpleza?**
   
   Pista: Piensa en eficiencia computacional y gradientes.

2. **¿Cuándo preferirías Sigmoid sobre ReLU?**
   
   Pista: Considera el tipo de problema y la capa.

3. **¿Qué significa que una neurona "muera"?**
   
   Pista: Piensa en términos de gradientes.

4. **¿Por qué Softmax suma 1?**
   
   Pista: Interpretación probabilística.

## Verificación de Comprensión

Después de completar el laboratorio, deberías poder:

- [ ] Explicar por qué necesitamos funciones de activación
- [ ] Implementar ReLU, Sigmoid, Tanh y Softmax desde cero
- [ ] Calcular las derivadas de estas funciones
- [ ] Identificar cuándo usar cada función
- [ ] Reconocer el problema del gradiente que desaparece
- [ ] Visualizar y comparar diferentes activaciones

## Recursos Adicionales

### Lecturas Recomendadas

1. **Paper original de ReLU**: "Rectified Linear Units Improve Restricted Boltzmann Machines" (Nair & Hinton, 2010)
2. **Understanding activations**: Deep Learning Book, Chapter 6
3. **Visualización interactiva**: [playground.tensorflow.org](https://playground.tensorflow.org)

### Videos

- 3Blue1Brown: "But what is a neural network?" (visualización excelente)
- Stanford CS231n: Lecture 6 (Training Neural Networks I)

### Herramientas

- [Neural Network Playground](https://playground.tensorflow.org)
- [Distill.pub - Activation Functions](https://distill.pub)

## Solución de Problemas

### Error: "RuntimeWarning: overflow encountered in exp"

**Causa**: Valores muy grandes en la exponencial de Sigmoid/Softmax

**Solución**: Usar estabilización numérica:
```python
# En lugar de: exp(x)
# Usar: exp(x - max(x))
```

### Neuronas muertas en ReLU

**Síntoma**: Muchas salidas son cero

**Soluciones**:
1. Reducir learning rate
2. Usar Leaky ReLU
3. Verificar inicialización de pesos

### Gradientes que desaparecen

**Síntoma**: Red profunda no aprende en capas iniciales

**Soluciones**:
1. Cambiar de Sigmoid a ReLU
2. Usar batch normalization (en labs posteriores)
3. Reducir profundidad de la red

## Próximo Laboratorio

En **Lab 04: Funciones de Pérdida**, aprenderemos:
- Mean Squared Error (MSE)
- Cross-Entropy Loss
- Cómo combinar pérdida con activación
- Optimización básica

## Notas Finales

Las funciones de activación son fundamentales en deep learning. ReLU ha revolucionado el campo por su simplicidad y efectividad. Sin embargo, entender todas las opciones te permitirá tomar mejores decisiones arquitectónicas.

**Recuerda**: No hay una función "perfecta". La elección depende del problema, la arquitectura y la experimentación.

---

**¿Preguntas o problemas?** Revisa la teoría, experimenta con el código, y recuerda: la mejor forma de aprender es implementando desde cero.

**¡Buena suerte! 🚀**
