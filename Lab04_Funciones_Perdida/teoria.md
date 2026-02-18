# Teoría: Funciones de Pérdida y Optimización

## Introducción

Las **funciones de pérdida** (loss functions) son el corazón del aprendizaje en redes neuronales. Cuantifican qué tan bien (o mal) las predicciones de nuestro modelo se ajustan a los datos reales. La optimización es el proceso de ajustar los parámetros del modelo para **minimizar** esta pérdida.

## ¿Qué es una Función de Pérdida?

Una función de pérdida mide la discrepancia entre:
- **Predicciones del modelo** (ŷ)
- **Valores reales** (y)

```
L = f(y, ŷ)
```

Donde:
- L = Pérdida (loss)
- y = Valores verdaderos
- ŷ = Predicciones del modelo
- Queremos: L → 0 (pérdida mínima)

## Principales Funciones de Pérdida

### 1. Mean Squared Error (MSE) - Error Cuadrático Medio

**Uso**: Problemas de **regresión**

**Ecuación**:
```
MSE = (1/n) * Σ(y_i - ŷ_i)²
```

**Derivada**:
```
∂MSE/∂ŷ = (2/n) * (ŷ - y)
```

**Características**:
- Penaliza fuertemente errores grandes (debido al cuadrado)
- Siempre positiva
- Sensible a outliers
- Unidades: cuadrado de las unidades originales

**Ventajas**:
- Matemáticamente conveniente (diferenciable)
- Penaliza errores grandes
- Bien entendida y estudiada

**Desventajas**:
- Muy sensible a outliers
- No probabilística
- Asume errores normalmente distribuidos

**Ejemplo**:
```python
# Si predecimos [3, 5, 2] y real es [2, 4, 3]
# MSE = ((3-2)² + (5-4)² + (2-3)²) / 3
# MSE = (1 + 1 + 1) / 3 = 1.0
```

**Cuándo usar**:
- Regresión con valores continuos
- Cuando los errores grandes son especialmente indeseables
- Predicción de precios, temperaturas, etc.

### 2. Mean Absolute Error (MAE) - Error Absoluto Medio

**Uso**: Problemas de **regresión** (alternativa a MSE)

**Ecuación**:
```
MAE = (1/n) * Σ|y_i - ŷ_i|
```

**Derivada**:
```
∂MAE/∂ŷ = sign(ŷ - y)
```

**Características**:
- Menos sensible a outliers que MSE
- Penalización lineal de errores
- Unidades: mismas que los datos originales

**Ventajas**:
- Robusta a outliers
- Interpretable (error promedio)
- Mismas unidades que los datos

**Desventajas**:
- Derivada no continua en cero
- Penaliza igual errores grandes y pequeños

**Cuándo usar**:
- Cuando hay muchos outliers
- Cuando todos los errores son igualmente importantes

### 3. Binary Cross-Entropy (Log Loss)

**Uso**: Clasificación **binaria**

**Ecuación**:
```
BCE = -(1/n) * Σ[y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i)]
```

Donde:
- y ∈ {0, 1} (clase verdadera)
- ŷ ∈ (0, 1) (probabilidad predicha)

**Derivada** (con sigmoid):
```
∂BCE/∂z = ŷ - y  (muy simple!)
```

**Características**:
- Interpretación probabilística
- Penaliza predicciones confiadas pero incorrectas
- Rango: [0, ∞)

**Ventajas**:
- Interpretación probabilística clara
- Derivada simple con sigmoid
- Bien calibrada para probabilidades

**Desventajas**:
- Requiere salidas en rango (0, 1)
- Sensible a predicciones muy confiadas pero incorrectas
- Puede dar valores muy altos para predicciones malas

**Ejemplo**:
```python
# Clase real: 1 (positivo)
# Predicción: 0.9 (90% confianza)
# BCE = -(1 * log(0.9) + 0 * log(0.1)) ≈ 0.105 (baja pérdida)

# Clase real: 1 (positivo)
# Predicción: 0.1 (10% confianza)
# BCE = -(1 * log(0.1) + 0 * log(0.9)) ≈ 2.303 (alta pérdida)
```

**Cuándo usar**:
- Clasificación binaria (sí/no, spam/no spam)
- Cuando necesitas probabilidades calibradas
- Con activación sigmoid en la salida

### 4. Categorical Cross-Entropy

**Uso**: Clasificación **multiclase** (una etiqueta por muestra)

**Ecuación**:
```
CCE = -(1/n) * Σ Σ y_{i,c} * log(ŷ_{i,c})
```

Donde:
- y es one-hot encoded: [0, 1, 0, 0] para clase 2 de 4
- ŷ son probabilidades de Softmax: suma = 1

**Derivada** (con softmax):
```
∂CCE/∂z = ŷ - y  (idéntica a binary cross-entropy!)
```

**Características**:
- Para K clases mutuamente excluyentes
- Interpreta salida como distribución de probabilidad
- Penaliza predicciones incorrectas con alta confianza

**Ventajas**:
- Natural para clasificación multiclase
- Derivada simple con softmax
- Interpretación probabilística

**Desventajas**:
- Requiere one-hot encoding
- Puede dar pérdidas muy altas para predicciones muy incorrectas

**Ejemplo**:
```python
# 3 clases: [perro, gato, pájaro]
# Clase real: gato → [0, 1, 0]
# Predicción: [0.1, 0.7, 0.2]
# CCE = -(0*log(0.1) + 1*log(0.7) + 0*log(0.2))
# CCE ≈ 0.357
```

**Cuándo usar**:
- Clasificación con 3+ clases excluyentes
- Reconocimiento de dígitos (0-9)
- Clasificación de imágenes (perro/gato/pájaro)
- Con activación softmax en la salida

### 5. Sparse Categorical Cross-Entropy

**Uso**: Igual que Categorical Cross-Entropy pero con etiquetas enteras

**Diferencia**:
```python
# Categorical Cross-Entropy
y = [0, 1, 0, 0]  # One-hot encoding

# Sparse Categorical Cross-Entropy
y = 1  # Índice de la clase
```

**Ventaja**: Más eficiente en memoria (no requiere one-hot)

## Comparación Visual

```
Problema              | Función de Pérdida
--------------------- | ---------------------------
Regresión             | MSE, MAE
Clasificación Binaria | Binary Cross-Entropy
Clasificación Multiclase | Categorical Cross-Entropy
```

## Optimización: Gradient Descent

Una vez que tenemos una función de pérdida, necesitamos **minimizarla**.

### Gradient Descent (Descenso del Gradiente)

**Idea**: Movernos en la dirección opuesta al gradiente.

**Algoritmo**:
```
1. Inicializar parámetros θ aleatoriamente
2. Repetir hasta convergencia:
   a. Calcular gradiente: ∇L(θ)
   b. Actualizar parámetros: θ = θ - α * ∇L(θ)
```

Donde:
- θ = parámetros (pesos y sesgos)
- α = learning rate (tasa de aprendizaje)
- ∇L(θ) = gradiente de la pérdida respecto a θ

### Learning Rate (α)

**Concepto clave**: Controla el tamaño del paso en cada actualización.

```
α muy pequeño:  Aprendizaje lento, pero estable
α muy grande:   Aprendizaje rápido, pero inestable (puede diverger)
α óptimo:       Balance entre velocidad y estabilidad
```

**Visualización**:
```
         Pérdida
            ^
            |     α muy grande
            |    /\    /\
            |   /  \  /  \
            |  /    \/    \
            | /
            |/___α óptimo____
            |        \
            |         \_____ α muy pequeño
            +-------------------> Iteraciones
```

**Valores típicos**: 0.001, 0.01, 0.1

### Variantes de Gradient Descent

#### 1. Batch Gradient Descent
```
- Usa TODOS los datos en cada actualización
- Lento pero preciso
- Requiere mucha memoria
```

#### 2. Stochastic Gradient Descent (SGD)
```
- Usa UNA muestra en cada actualización
- Rápido pero ruidoso
- Puede escapar de mínimos locales
```

#### 3. Mini-batch Gradient Descent
```
- Usa un LOTE pequeño de datos
- Balance entre batch y stochastic
- MÁS COMÚN en la práctica
- Típicamente: 32, 64, 128, 256 muestras
```

### Optimizadores Avanzados

En labs posteriores veremos:
- **Momentum**: Acelera el descenso
- **RMSprop**: Adapta learning rate
- **Adam**: Combina momentum y RMSprop (muy popular)

## Relación entre Pérdida y Activación

Algunas combinaciones son especialmente eficientes:

| Problema | Activación Final | Función de Pérdida |
|----------|------------------|-------------------|
| Regresión | Lineal | MSE, MAE |
| Clasificación Binaria | Sigmoid | Binary Cross-Entropy |
| Clasificación Multiclase | Softmax | Categorical Cross-Entropy |

**Por qué**: Las derivadas se simplifican matemáticamente.

Por ejemplo:
```python
# Sigmoid + Binary Cross-Entropy
∂Loss/∂z = ŷ - y  # ¡Muy simple!

# Softmax + Categorical Cross-Entropy
∂Loss/∂z = ŷ - y  # ¡También muy simple!
```

## Overfitting y Regularización (Problema y Solución)

### Overfitting (Sobreajuste)

**Problema**: El modelo aprende los datos de entrenamiento demasiado bien, incluyendo el ruido.

**Síntomas**:
- Pérdida de entrenamiento ↓ (baja)
- Pérdida de validación ↑ (alta)
- Mal desempeño en datos nuevos

**Visualización**:
```
Pérdida  |
         |          Validación
         |         /
         |        /
         |   Train \___
         |_______________
              Épocas
```

### Regularización (Básica)

**L2 Regularization (Weight Decay)**:
```
Loss_total = Loss_original + λ * Σ(w²)
```
- Penaliza pesos grandes
- λ controla la fuerza de regularización

**L1 Regularization**:
```
Loss_total = Loss_original + λ * Σ|w|
```
- Genera sparsity (algunos pesos = 0)

**Dropout**:
El dropout es una técnica de regularización en redes neuronales que desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento (típicamente entre 20% y 50%). Esto evita que la red dependa excesivamente de neuronas específicas, reduciendo el sobreajuste (overfitting) y forzando al modelo a aprender patrones más generales y robustos.

¿Cómo funciona?

- Entrenamiento: En cada iteración (batch), se selecciona una probabilidad 
 (tasa de dropout) y se apagan aleatoriamente esa fracción de neuronas, impidiéndoles contribuir a la propagación hacia adelante y atrás.
- Inferencia (Prueba): Se activan todas las neuronas, pero sus pesos se escalan (multiplican) por para equilibrar la menor actividad durante el entrenamiento.

Ejemplo Práctico:
Imagina una red neuronal donde una capa oculta tiene 100 neuronas y aplicas un Dropout del 50% (p=0.5):
- Iteración 1: El algoritmo desactiva aleatoriamente 50 neuronas. El modelo aprende con las otras 50.
- Iteración 2: Se seleccionan otras 50 neuronas al azar para apagar, y el modelo entrena con un subconjunto distinto.
- Resultado: Ninguna neurona se vuelve indispensable, lo que mejora la generalización. 

Beneficios:
- Reduce el sobreajuste: Impide que las neuronas co-adapten sus pesos de forma compleja.
- Mejora la robustez: Crea redes más fiables al no depender de una "neurona estrella".
- Efecto de ensamble: Funciona como si entrenaras múltiples sub-redes diferentes simultáneamente.

## Métricas vs Pérdidas

**Pérdida**: Lo que optimizamos
**Métrica**: Lo que medimos (para humanos)

Ejemplos:
- **Pérdida**: Cross-Entropy
- **Métrica**: Accuracy (precisión)

No siempre coinciden. Por ejemplo, podemos optimizar cross-entropy pero reportar accuracy.

## Conceptos Clave para Recordar

1. **Pérdida cuantifica error**: Qué tan lejos están las predicciones
2. **Elección depende del problema**: Regresión vs Clasificación
3. **Gradient descent minimiza**: Iterativamente reduce la pérdida
4. **Learning rate es crucial**: Ni muy grande ni muy pequeño
5. **Mini-batches son estándar**: Balance entre velocidad y precisión
6. **Activación + Pérdida**: Algunas combinaciones son ideales

## Ejercicios de Reflexión

1. ¿Por qué MSE penaliza más los errores grandes?
2. ¿Qué pasa si el learning rate es demasiado grande?
3. ¿Por qué cross-entropy es mejor que MSE para clasificación?
4. ¿Cuándo usarías MAE en lugar de MSE?

## Próximos Pasos

En el siguiente laboratorio implementaremos **Backpropagation**, el algoritmo que calcula los gradientes necesarios para gradient descent.
