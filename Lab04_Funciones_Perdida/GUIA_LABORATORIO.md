# Guía de Laboratorio: Funciones de Pérdida y Optimización

## 📋 Información del Laboratorio

**Título:** Funciones de Pérdida y Optimización  
**Código:** Lab 04  
**Duración:** 2-3 horas  
**Nivel:** Básico-Intermedio  

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender qué son funciones de pérdida y su propósito
2. Implementar MSE, MAE, Cross-Entropy desde cero
3. Elegir función de pérdida apropiada para cada problema
4. Implementar gradient descent básico
5. Comprender efecto del learning rate
6. Reconocer y detectar overfitting
7. Calcular derivadas de funciones de pérdida
8. Combinar pérdida con activación eficientemente
9. Aplicar regularización básica

## 📚 Prerrequisitos

### Conocimientos

- Completar Lab 01-03
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

Las **funciones de pérdida** cuantifican qué tan bien las predicciones del modelo se ajustan a los datos reales. Son el corazón del aprendizaje en redes neuronales.

### Contexto del Problema

Hasta ahora construimos redes y generamos predicciones, pero ¿cómo sabemos si son buenas? Necesitamos una métrica que:
- Cuantifique el error
- Guíe la optimización
- Permita comparar modelos

### Funciones de Pérdida

Miden discrepancia entre predicciones (ŷ) y valores reales (y):

```
L = f(y, ŷ)
Objetivo: L → 0 (minimizar pérdida)
```

### Conceptos Fundamentales

**1. Tipos principales:**
- **MSE:** Regresión (penaliza errores grandes)
- **MAE:** Regresión (robusta a outliers)
- **Binary Cross-Entropy:** Clasificación binaria
- **Categorical Cross-Entropy:** Clasificación multiclase

**2. Optimización:** Gradient descent ajusta parámetros para minimizar pérdida

**3. Learning Rate:** Controla tamaño del paso de actualización

### Aplicaciones

- Regresión: Predecir precios, temperaturas
- Clasificación: Spam detection, reconocimiento de imágenes
- Optimización: Entrenar cualquier modelo de ML

---

## 🤔 Preguntas de Reflexión

> Antes de comenzar a programar, dedica unos minutos a reflexionar sobre las siguientes preguntas. No necesitas tener las respuestas correctas ahora; el objetivo es activar tu pensamiento crítico y motivar el aprendizaje.

1. **Sobre la elección de la función de pérdida:** Si tienes un problema de predicción de precios de casas donde algunos valores son extremadamente altos (mansiones), ¿qué función de pérdida crees que sería más adecuada, MSE o MAE? ¿Por qué los errores grandes deberían o no deberían penalizarse más?

2. **Sobre la interpretación probabilística:** En clasificación binaria, la cross-entropy utiliza logaritmos. ¿Qué crees que sucede con la pérdida cuando el modelo predice una probabilidad de 0.99 para la clase correcta? ¿Y cuando predice 0.01? ¿Por qué el logaritmo captura mejor esta asimetría que el error cuadrático?

3. **Sobre el learning rate:** Imagina que estás bajando una montaña en la oscuridad. El learning rate sería el tamaño de cada paso. ¿Qué pasaría si tus pasos fueran demasiado grandes? ¿Y demasiado pequeños? ¿Existe un tamaño de paso "perfecto" universal?

4. **Sobre overfitting:** Un modelo entrena durante 1000 épocas y logra un error de entrenamiento casi cero, pero su error en datos nuevos es 10 veces mayor. ¿Qué crees que está ocurriendo? ¿Cómo distinguirías este fenómeno durante el entrenamiento?

5. **Sobre regularización:** Si la regularización penaliza los pesos grandes, ¿estamos realmente "empeorando" el entrenamiento a propósito? ¿Por qué sacrificar rendimiento en entrenamiento podría mejorar el rendimiento en datos nuevos?

6. **Sobre la relación pérdida-derivada:** El gradiente de la función de pérdida indica la dirección de mayor crecimiento. Si queremos minimizar la pérdida, ¿en qué dirección deberíamos mover los parámetros? ¿Por qué substraemos el gradiente en lugar de sumarlo?

---

## 🔬 Parte 1: Funciones para Regresión (45 min)

### 1.1 Mean Squared Error (MSE)

**¿Qué hacemos?** Implementamos el Error Cuadrático Medio (MSE) y su derivada, la función de pérdida más utilizada para problemas de regresión.

**¿Por qué lo hacemos?** MSE mide el promedio de los cuadrados de las diferencias entre los valores predichos y los reales. Al elevar al cuadrado, se consiguen dos efectos deseables: los errores siempre son positivos (no se cancelan entre sí) y los errores grandes reciben una penalización desproporcionadamente mayor que los errores pequeños. Matemáticamente:

$$\text{MSE}(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

La derivada respecto a las predicciones $\hat{y}$ es:

$$\frac{\partial \text{MSE}}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)$$

Esta derivada es la que se utiliza en backpropagation para ajustar los parámetros de la red.

**¿Cómo lo hacemos?** Usamos operaciones vectorizadas de NumPy: calculamos la diferencia elemento a elemento, la elevamos al cuadrado y tomamos la media. La derivada es simplemente el doble de la diferencia normalizada por el tamaño del conjunto.

**¿Qué resultados debemos esperar?** Para predicciones perfectas `y_pred == y_true`, MSE debe dar exactamente 0. A medida que las predicciones se alejan de los valores reales, MSE crece cuadráticamente. Un error promedio de 1 unidad da MSE = 1, pero un error promedio de 2 unidades da MSE = 4 (no 2).

```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_derivada(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

# Ejemplo
y_true = np.array([2, 4, 3])
y_pred = np.array([3, 5, 2])
print(f"MSE: {mse(y_true, y_pred)}")  # 1.0
```

### 1.2 Mean Absolute Error (MAE)

**¿Qué hacemos?** Implementamos el Error Absoluto Medio (MAE) y su derivada (subgradiente), una alternativa a MSE más robusta ante valores atípicos.

**¿Por qué lo hacemos?** A diferencia de MSE, MAE trata todos los errores de forma lineal, sin importar su magnitud. Esto lo hace menos sensible a outliers. La fórmula es:

$$\text{MAE}(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

Su derivada (técnicamente un subgradiente, ya que el valor absoluto no es diferenciable en 0) es:

$$\frac{\partial \text{MAE}}{\partial \hat{y}} = \frac{1}{n} \cdot \text{sign}(\hat{y} - y)$$

donde $\text{sign}(x) = +1$ si $x > 0$ y $-1$ si $x < 0$.

**¿Cómo lo hacemos?** NumPy provee `np.abs()` para el valor absoluto y `np.sign()` para la función signo. Nótese que el subgradiente siempre tiene magnitud constante ±1/n, lo que puede hacer la optimización menos eficiente cerca del mínimo.

**¿Qué resultados debemos esperar?** Para el mismo conjunto de predicciones, MAE generalmente da un valor menor o igual que la raíz cuadrada de MSE. La diferencia se amplía cuando hay errores grandes (outliers).

```python
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_derivada(y_true, y_pred):
    return np.sign(y_pred - y_true) / len(y_true)

# Ejemplo
print(f"MAE: {mae(y_true, y_pred)}")  # 1.0
```

### 1.3 Comparación con Outliers

**¿Qué hacemos?** Comparamos el comportamiento de MSE y MAE cuando el conjunto de datos contiene un valor atípico (outlier) extremo.

**¿Por qué lo hacemos?** Comprender la sensibilidad diferencial a outliers es fundamental para elegir la función de pérdida correcta. En datos reales, los outliers son frecuentes (errores de medición, casos excepcionales) y pueden distorsionar el entrenamiento. MSE penaliza los outliers cuadráticamente: un error 10 veces mayor produce una pérdida 100 veces mayor. MAE los trata linealmente, siendo mucho más robusto.

**¿Cómo lo hacemos?** Creamos un conjunto de datos donde todos los errores son pequeños excepto uno que es extremadamente grande (100 en lugar del valor real 5). Calculamos MSE y MAE para comparar el impacto.

**¿Qué resultados debemos esperar?** MSE reportará un valor muy alto (dominado por el outlier elevado al cuadrado), mientras que MAE reportará un valor más moderado. Esto ilustra por qué en problemas donde los outliers son inevitables o representativos (como detección de fraude), MAE o funciones híbridas como Huber Loss son preferibles.

```python
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.1, 3.1, 4.1, 100])  # último es outlier

print(f"MSE: {mse(y_true, y_pred):.2f}")  # Alto (penaliza mucho outlier)
print(f"MAE: {mae(y_true, y_pred):.2f}")  # Menor (más robusto)
```

### Actividades

1. Implementar Huber Loss
2. Comparar MSE vs MAE con diferentes datos
3. Visualizar curvas de pérdida

---

## 🔬 Parte 2: Funciones para Clasificación (45 min)

### 2.1 Binary Cross-Entropy

**¿Qué hacemos?** Implementamos la Entropía Cruzada Binaria (BCE), la función de pérdida estándar para problemas de clasificación binaria (dos clases), junto con su derivada.

**¿Por qué lo hacemos?** Para clasificación binaria, las predicciones son probabilidades $\hat{y} \in (0, 1)$ obtenidas con la función Sigmoid. MSE no es adecuado aquí porque el espacio de probabilidades no es lineal. La BCE tiene una interpretación probabilística directa: mide la log-verosimilitud negativa bajo un modelo de Bernoulli:

$$\text{BCE}(y, \hat{y}) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

Cuando la predicción $\hat{y}$ es correcta y confiada (cercana a 1 para $y=1$, o a 0 para $y=0$), la pérdida es mínima. Cuando el modelo predice con alta confianza la clase equivocada, el logaritmo genera una penalización muy grande (el logaritmo de un número cercano a 0 tiende a $-\infty$). La derivada combinada con Sigmoid simplifica elegantemente a $\hat{y} - y$.

**¿Cómo lo hacemos?** Usamos `np.clip()` para evitar el cálculo de $\log(0)$, que es indefinido. Esto añade un epsilon numérico ($\epsilon = 10^{-15}$) como límite inferior y superior de las predicciones.

**¿Qué resultados debemos esperar?** Predicciones perfectas producen una pérdida cercana a 0. Para el ejemplo dado (predicciones de alta confianza correctas), esperamos una BCE muy pequeña. Si intercambiamos `y_true` y `y_pred`, la pérdida aumentará dramáticamente.

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # Evitar log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivada(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

# Ejemplo
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])
print(f"BCE: {binary_cross_entropy(y_true, y_pred):.4f}")
```

### 2.2 Categorical Cross-Entropy

**¿Qué hacemos?** Implementamos la Entropía Cruzada Categórica (CCE), la extensión de BCE para problemas de clasificación multiclase (más de dos clases), donde las etiquetas están en formato *one-hot*.

**¿Por qué lo hacemos?** Cuando hay $C$ clases posibles, las predicciones son vectores de probabilidad $\hat{y} \in \mathbb{R}^C$ producidos por Softmax, y las etiquetas reales se representan en formato one-hot (vector con un 1 en la posición de la clase correcta y 0 en el resto). La CCE mide qué tan lejos está la distribución predicha de la distribución real:

$$\text{CCE}(y, \hat{y}) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

Dado que $y$ es one-hot, en la práctica solo el término correspondiente a la clase correcta contribuye a la suma. La pérdida es simplemente $-\log(\hat{y}_{\text{clase correcta}})$: cuanto mayor sea la probabilidad asignada a la clase correcta, menor será la pérdida.

**¿Cómo lo hacemos?** Multiplicamos elemento a elemento `y_true * np.log(y_pred)` y sumamos a lo largo del eje de las clases (`axis=1`), luego tomamos el negativo de la media. El clipping previene errores numéricos.

**¿Qué resultados debemos esperar?** Para el ejemplo dado, con probabilidades de 0.7 y 0.8 para las clases correctas, esperamos una pérdida pequeña (alrededor de 0.2-0.3). Si el modelo asignara probabilidades bajas a las clases correctas, la pérdida aumentaría significativamente.

```python
def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # y_true: one-hot encoded
    # y_pred: probabilidades de softmax
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Ejemplo
y_true = np.array([[0, 1, 0], [1, 0, 0]])  # One-hot
y_pred = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])  # Softmax
print(f"CCE: {categorical_cross_entropy(y_true, y_pred):.4f}")
```

### 2.3 Sparse Categorical Cross-Entropy

**¿Qué hacemos?** Implementamos la versión "sparse" de CCE, que acepta directamente los índices de clase en lugar de vectores one-hot.

**¿Por qué lo hacemos?** Cuando el número de clases $C$ es muy grande (por ejemplo, 10,000 categorías en clasificación de palabras), almacenar las etiquetas en formato one-hot requiere una matriz de tamaño $n \times C$, lo cual puede ser prohibitivamente costoso en memoria. La Sparse CCE acepta simplemente el índice de la clase correcta (un entero), siendo matemáticamente equivalente a CCE pero mucho más eficiente en memoria:

$$\text{Sparse CCE}(y, \hat{y}) = -\frac{1}{n}\sum_{i=1}^{n}\log(\hat{y}_{i, y_i})$$

donde $y_i$ es el índice (entero) de la clase correcta para la muestra $i$. En comparación con CCE:
- **CCE**: etiquetas como `[[0, 1, 0], [1, 0, 0]]` (one-hot, más memoria)
- **Sparse CCE**: etiquetas como `[1, 0]` (índices, mucho menos memoria)

**¿Cómo lo hacemos?** Usamos indexación avanzada de NumPy (`y_pred[range(n), y_true]`) para seleccionar directamente la probabilidad predicha para cada clase correcta, evitando construir la representación one-hot.

**¿Qué resultados debemos esperar?** Para las mismas predicciones y etiquetas (expresadas de forma diferente), Sparse CCE debe producir exactamente el mismo resultado numérico que CCE. Esto sirve como verificación de consistencia entre las dos implementaciones.

```python
def sparse_categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    # y_true: índices de clase (no one-hot)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n_samples = y_pred.shape[0]
    log_probs = np.log(y_pred[range(n_samples), y_true])
    return -np.mean(log_probs)

# Ejemplo
y_true = np.array([1, 0])  # Índices
y_pred = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
print(f"Sparse CCE: {sparse_categorical_cross_entropy(y_true, y_pred):.4f}")
```

### Actividades

1. Implementar todas las pérdidas
2. Comparar BCE vs MSE para clasificación
3. Verificar derivadas numéricamente

---

## 🔬 Parte 3: Gradient Descent (45 min)

### 3.1 Implementación Básica

**¿Qué hacemos?** Implementamos el algoritmo de Gradient Descent (Descenso de Gradiente) desde cero para optimizar los parámetros de un modelo de regresión lineal.

**¿Por qué lo hacemos?** Gradient Descent es el algoritmo fundamental de optimización en deep learning. Su objetivo es encontrar los parámetros $w$ (pesos) y $b$ (sesgo) que minimizan la función de pérdida. El algoritmo sigue estos pasos en cada iteración (época):

1. **Forward pass:** Calcular predicciones $\hat{y} = X \cdot w + b$
2. **Cálculo de pérdida:** $L = \text{MSE}(y, \hat{y})$
3. **Cálculo de gradientes:** $\nabla_w L = \frac{\partial L}{\partial w}$ y $\nabla_b L = \frac{\partial L}{\partial b}$
4. **Actualización de parámetros:** $w \leftarrow w - \alpha \cdot \nabla_w L$ y $b \leftarrow b - \alpha \cdot \nabla_b L$

donde $\alpha$ es el **learning rate** (tasa de aprendizaje). La clave está en el signo negativo: nos movemos en la dirección **opuesta** al gradiente, que es la dirección de mayor descenso.

**¿Cómo lo hacemos?** Para regresión lineal con MSE, los gradientes tienen forma cerrada: $\frac{\partial L}{\partial w} = \frac{2}{n} X^T (\hat{y} - y)$ y $\frac{\partial L}{\partial b} = \frac{2}{n} \sum(\hat{y} - y)$. Usamos multiplicación matricial (`@`) para eficiencia.

**¿Qué resultados debemos esperar?** La pérdida debe disminuir monótonamente con cada época (para un learning rate apropiado). Al imprimir cada 10 épocas, veremos cómo el modelo converge gradualmente hacia la solución óptima.

```python
def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    # Inicializar parámetros
    w = np.zeros(X.shape[1])
    b = 0
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = X @ w + b
        
        # Calcular pérdida
        loss = mse(y, y_pred)
        losses.append(loss)
        
        # Calcular gradientes
        dw = 2 * X.T @ (y_pred - y) / len(y)
        db = 2 * np.mean(y_pred - y)
        
        # Actualizar parámetros
        w -= learning_rate * dw
        b -= learning_rate * db
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return w, b, losses
```

### 3.2 Efecto del Learning Rate

**¿Qué hacemos?** Comparamos visualmente el efecto de diferentes valores de learning rate sobre la curva de convergencia del entrenamiento.

**¿Por qué lo hacemos?** El learning rate $\alpha$ es el hiperparámetro más crítico en la optimización. Governa la dinámica de convergencia:

- **$\alpha$ muy pequeño** (ej. 0.0001): convergencia lenta, requiere muchas épocas, puede quedarse atascado en mínimos locales.
- **$\alpha$ óptimo** (ej. 0.01): convergencia estable y rápida hacia el mínimo global.
- **$\alpha$ grande** (ej. 0.5): oscilaciones alrededor del mínimo; el algoritmo "salta" de un lado al otro sin converger.
- **$\alpha$ muy grande** (ej. 1.0+): divergencia; la pérdida **aumenta** en lugar de disminuir, el entrenamiento falla completamente.

Esta propiedad se relaciona con el radio espectral de la matriz hessiana de la función de pérdida: existe una tasa de aprendizaje máxima teórica más allá de la cual el gradiente descent diverge.

**¿Cómo lo hacemos?** Entrenamos el mismo modelo con los mismos datos usando cuatro valores de learning rate diferentes, y graficamos todas las curvas de pérdida en la misma figura para comparación directa.

**¿Qué resultados debemos esperar?** Veremos cuatro comportamientos claramente distintos: convergencia lenta, convergencia óptima, oscilaciones y divergencia. La gráfica resultante es una de las visualizaciones más instructivas en el aprendizaje de deep learning.

```python
def comparar_learning_rates(X, y):
    lrs = [0.001, 0.01, 0.1, 1.0]
    
    plt.figure(figsize=(12, 4))
    
    for lr in lrs:
        _, _, losses = gradient_descent(X, y, learning_rate=lr, epochs=100)
        plt.plot(losses, label=f'LR={lr}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Efecto del Learning Rate')
    plt.grid(True)
    plt.savefig('learning_rates.png')
```

### 3.3 Variantes de Gradient Descent

**¿Qué hacemos?** Implementamos las tres variantes principales del algoritmo de Gradient Descent: Batch GD, Stochastic GD (SGD) y Mini-Batch GD.

**¿Por qué lo hacemos?** La diferencia fundamental entre las variantes radica en **cuántos datos** se usan para calcular el gradiente en cada actualización. Esto crea un trade-off entre precisión del gradiente y velocidad de actualización:

| Variante | Datos por update | Gradiente | Velocidad | Uso de memoria | Convergencia |
|----------|-----------------|-----------|-----------|----------------|--------------|
| **Batch GD** | Todo el dataset | Exacto | Lenta | Alta | Suave, estable |
| **SGD** | 1 muestra | Ruidoso | Muy rápida | Mínima | Ruidosa, puede escapar mínimos locales |
| **Mini-Batch GD** | $k$ muestras ($k$=32-256) | Aproximado | Balanceada | Moderada | **Estándar en práctica** |

Mini-Batch GD combina lo mejor de ambos mundos: es lo suficientemente rápido (múltiples actualizaciones por época) y lo suficientemente preciso (el gradiente promediado sobre un batch es una buena estimación del gradiente real).

**¿Cómo lo hacemos?** Para SGD procesamos cada muestra individualmente. Para Mini-Batch, mezclamos aleatoriamente los datos en cada época (`np.random.permutation`) y procesamos en chunks de tamaño `batch_size`. El shuffle es crucial para evitar que el modelo "memorice" el orden de los datos.

**¿Qué resultados debemos esperar?** Con los mismos datos y épocas, los tres métodos deberían llegar a soluciones similares. Sin embargo, SGD tendrá una curva de pérdida ruidosa (zigzagueante), mientras que Batch GD tendrá una curva perfectamente suave pero más lenta por época.

```python
# Batch Gradient Descent (ya implementado arriba)

# Stochastic Gradient Descent
def sgd(X, y, learning_rate=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    
    for epoch in range(epochs):
        for i in range(len(X)):
            xi = X[i:i+1]
            yi = y[i:i+1]
            
            y_pred = xi @ w + b
            dw = 2 * xi.T @ (y_pred - yi)
            db = 2 * (y_pred - yi)
            
            w -= learning_rate * dw.flatten()
            b -= learning_rate * db[0]
    
    return w, b

# Mini-batch Gradient Descent
def mini_batch_gd(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, len(X), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            y_pred = X_batch @ w + b
            dw = 2 * X_batch.T @ (y_pred - y_batch) / len(X_batch)
            db = 2 * np.mean(y_pred - y_batch)
            
            w -= learning_rate * dw
            b -= learning_rate * db
    
    return w, b
```

### Actividades

1. Implementar gradient descent para clasificación
2. Comparar batch, SGD, mini-batch
3. Encontrar learning rate óptimo

---

## 🔬 Parte 4: Overfitting y Regularización (40 min)

### 4.1 Demostración de Overfitting

**¿Qué hacemos?** Demostramos el fenómeno de overfitting ajustando modelos polinomiales de diferente complejidad a un conjunto de datos pequeño, observando cómo la brecha entre error de entrenamiento y validación crece con la complejidad del modelo.

**¿Por qué lo hacemos?** Overfitting es el problema más fundamental y frecuente en machine learning. Surge del **trade-off sesgo-varianza** (*bias-variance tradeoff*):

- **Underfitting (alto sesgo):** El modelo es demasiado simple para capturar los patrones reales. Alto error tanto en entrenamiento como en validación.
- **Overfitting (alta varianza):** El modelo es demasiado complejo y "memoriza" los datos de entrenamiento, incluyendo el ruido. Error muy bajo en entrenamiento pero muy alto en validación.
- **Balance óptimo:** El modelo captura los patrones reales sin memorizar el ruido. Ambos errores son bajos y similares.

Matemáticamente, el error esperado de un modelo puede descomponerse como:

$$\text{Error} = \text{Sesgo}^2 + \text{Varianza} + \text{Ruido irreducible}$$

Aumentar la complejidad del modelo reduce el sesgo pero aumenta la varianza. El objetivo es encontrar la complejidad que minimiza la suma total.

**¿Cómo lo hacemos?** Usamos regresión polinomial con grados 1 (lineal), 3 (cúbico) y 10 (alto grado). Para datos generados con una relación lineal más ruido gaussiano, el modelo de grado 10 tendrá suficiente capacidad para "memorizar" los 15 puntos de entrenamiento perfectamente, pero fallará en los 5 puntos de validación.

**¿Qué resultados debemos esperar?** Para grado 1: errores similares en train y val (underfitting moderado). Para grado 3: ambos errores bajos (modelo adecuado). Para grado 10: error de entrenamiento muy bajo pero error de validación muy alto (overfitting severo). Esta divergencia es la "señal de alarma" del overfitting.

```python
def simular_overfitting():
    # Generar datos
    np.random.seed(42)
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = 2 * X + 1 + np.random.randn(20, 1) * 2
    
    # Split train/val
    X_train, X_val = X[:15], X[15:]
    y_train, y_val = y[:15], y[15:]
    
    # Entrenar con diferentes complejidades
    for degree in [1, 3, 10]:
        # Crear features polinomiales
        X_poly_train = np.column_stack([X_train**i for i in range(1, degree+1)])
        X_poly_val = np.column_stack([X_val**i for i in range(1, degree+1)])
        
        w, b, _ = gradient_descent(X_poly_train, y_train, epochs=1000)
        
        train_loss = mse(y_train, X_poly_train @ w + b)
        val_loss = mse(y_val, X_poly_val @ w + b)
        
        print(f"Degree {degree}: Train={train_loss:.2f}, Val={val_loss:.2f}")
```

### 4.2 L2 Regularization

**¿Qué hacemos?** Implementamos la regularización L2 (también llamada *Ridge* o *weight decay*), añadiendo un término de penalización a la función de pérdida que desincentiva pesos grandes.

**¿Por qué lo hacemos?** La regularización L2 es la técnica más clásica para combatir el overfitting. La intuición es elegante: pesos grandes indican que el modelo depende excesivamente de características específicas del dataset de entrenamiento (incluyendo el ruido). Al penalizar los pesos grandes, forzamos al modelo a distribuir la "responsabilidad" entre más características, generalizando mejor.

La función de pérdida regularizada es:

$$L_{\text{reg}}(w) = L(w) + \lambda \|w\|_2^2 = L(w) + \lambda \sum_{j} w_j^2$$

donde $\lambda$ (lambda) es el **coeficiente de regularización** que controla el balance entre ajustar los datos y mantener pesos pequeños. El gradiente de la pérdida regularizada es:

$$\frac{\partial L_{\text{reg}}}{\partial w} = \frac{\partial L}{\partial w} + 2\lambda w$$

El término $2\lambda w$ actúa como una "fuerza restauradora" que empuja continuamente los pesos hacia cero en cada actualización: $w \leftarrow w(1 - 2\alpha\lambda) - \alpha \frac{\partial L}{\partial w}$. Por eso también se llama *weight decay* (decaimiento de pesos).

**¿Cómo lo hacemos?** Añadimos `lambda_reg * np.sum(w**2)` a la pérdida calculada y `2 * lambda_reg * w` al gradiente de los pesos. Nota importante: el sesgo $b$ generalmente **no** se regulariza, ya que no contribuye al overfitting de la misma manera.

**¿Qué resultados debemos esperar?** Con regularización, el modelo de alto grado polinomial debería producir pesos más pequeños y una brecha train/val reducida. Con $\lambda$ demasiado grande, el modelo underfit (ignora los datos). El valor óptimo de $\lambda$ se encuentra mediante validación cruzada.

```python
def gradient_descent_l2(X, y, lambda_reg=0.01, learning_rate=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    
    for epoch in range(epochs):
        y_pred = X @ w + b
        
        # Pérdida con regularización
        loss = mse(y, y_pred) + lambda_reg * np.sum(w**2)
        
        # Gradientes con regularización
        dw = 2 * X.T @ (y_pred - y) / len(y) + 2 * lambda_reg * w
        db = 2 * np.mean(y_pred - y)
        
        w -= learning_rate * dw
        b -= learning_rate * db
    
    return w, b
```

### 4.3 Early Stopping

**¿Qué hacemos?** Implementamos el algoritmo de Early Stopping (parada temprana), una técnica de regularización implícita que detiene el entrenamiento cuando el rendimiento en el conjunto de validación deja de mejorar.

**¿Por qué lo hacemos?** A diferencia de L2 que modifica la función de pérdida, Early Stopping es una forma de regularización "gratis": no requiere cambiar el modelo ni añadir hiperparámetros de regularización. La idea es simple pero poderosa: durante el entrenamiento, la pérdida de entrenamiento disminuye monotónicamente, pero la pérdida de validación típicamente tiene forma de "U" (primero baja, luego sube cuando empieza el overfitting). Early Stopping detiene el entrenamiento en el "valle" de esa U.

El mecanismo de **paciencia** (*patience*) es crucial: no detenemos el entrenamiento ante la primera época en que la validación no mejora (podría ser una fluctuación temporal), sino solo si no mejora durante $p$ épocas consecutivas. Esto hace el algoritmo más robusto a oscilaciones en la pérdida de validación. El proceso:

1. Monitorear la pérdida de validación en cada época
2. Si mejora → guardar los pesos actuales como "mejor modelo" y resetear contador
3. Si no mejora → incrementar contador de paciencia
4. Si contador ≥ paciencia → detener y restaurar los mejores pesos

**¿Cómo lo hacemos?** Guardamos copias de los mejores pesos (`best_w`, `best_b`) y un contador de paciencia. Al final del entrenamiento (ya sea por paciencia o por completar todas las épocas), devolvemos los mejores pesos encontrados, no los últimos.

**¿Qué resultados debemos esperar?** El entrenamiento se detendrá antes de las 1000 épocas configuradas. El mensaje "Early stopping en epoch X" indica cuándo se activó. Los pesos devueltos corresponden al mejor modelo en validación, no al modelo final sobreajustado.

```python
def train_with_early_stopping(X_train, y_train, X_val, y_val, patience=10):
    w = np.zeros(X_train.shape[1])
    b = 0
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_w, best_b = w.copy(), b
    
    for epoch in range(1000):
        # Entrenar
        y_pred = X_train @ w + b
        dw = 2 * X_train.T @ (y_pred - y_train) / len(y_train)
        db = 2 * np.mean(y_pred - y_train)
        w -= 0.01 * dw
        b -= 0.01 * db
        
        # Validar
        y_val_pred = X_val @ w + b
        val_loss = mse(y_val, y_val_pred)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_w, best_b = w.copy(), b
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping en epoch {epoch}")
            break
    
    return best_w, best_b
```

### Actividades

1. Demostrar overfitting con datos polinomiales
2. Aplicar L2 regularization
3. Implementar early stopping

---

## 📊 Análisis Final de Rendimiento

### Por qué el Análisis de Rendimiento es Fundamental

**¿Por qué analizamos el rendimiento de nuestras implementaciones?** En el contexto de deep learning, la eficiencia computacional no es un lujo sino una necesidad. Las redes neuronales reales se entrenan con millones de parámetros y millones de ejemplos; una implementación 10 veces más lenta puede significar días de entrenamiento adicionales. Comprender el rendimiento de nuestras implementaciones nos permite:

- **Identificar cuellos de botella:** Saber qué parte del código consume más tiempo permite optimizarla prioritariamente.
- **Escalar adecuadamente:** Entender cómo el tiempo de ejecución crece con el tamaño de los datos (complejidad algorítmica).
- **Tomar decisiones informadas:** Elegir entre claridad de código y eficiencia computacional según el contexto.
- **Prepararse para frameworks:** NumPy vectorizado se asemeja al comportamiento de TensorFlow/PyTorch en CPU; entender estos patrones de rendimiento facilita la transición.

El análisis de rendimiento también revela la importancia de la **vectorización**: reemplazar loops de Python con operaciones matriciales de NumPy puede producir aceleraciones de 100x o más, ya que NumPy delega las operaciones a rutinas optimizadas en C/Fortran (BLAS/LAPACK).

### Comparación de Implementaciones

**¿Qué métricas comparamos?**
- **Tiempo de forward pass:** Cuánto tarda calcular la pérdida dado un conjunto de predicciones
- **Escalabilidad:** Cómo varía el tiempo con el número de muestras ($n$) y características ($d$)
- **Eficiencia de gradient descent:** Tiempo por época para batch vs. mini-batch vs. SGD

```python
import time
import numpy as np

def benchmark_loss_functions(n_samples=10000):
    """
    Mide y compara el tiempo de ejecución de cada función de pérdida
    para un conjunto de datos de tamaño n_samples.
    """
    np.random.seed(42)
    y_true = np.random.rand(n_samples)
    y_pred = np.random.rand(n_samples)

    funciones = {
        'MSE': lambda: mse(y_true, y_pred),
        'MAE': lambda: mae(y_true, y_pred),
        'BCE': lambda: binary_cross_entropy(y_true, y_pred),
    }

    print(f"{'Función':<20} {'Tiempo (ms)':>15} {'Resultado':>15}")
    print("-" * 52)

    for nombre, fn in funciones.items():
        # Warm-up para evitar efectos de caché fría
        fn()
        # Medición con múltiples repeticiones para mayor precisión
        repeticiones = 100
        inicio = time.perf_counter()
        for _ in range(repeticiones):
            resultado = fn()
        fin = time.perf_counter()
        tiempo_ms = (fin - inicio) / repeticiones * 1000
        print(f"{nombre:<20} {tiempo_ms:>14.4f}ms {resultado:>15.6f}")


def benchmark_gradient_descent(n_samples=1000, n_features=10, epochs=50):
    """
    Compara el tiempo por época de las tres variantes de gradient descent.
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    variantes = {
        'Batch GD':      lambda: gradient_descent(X, y, epochs=epochs),
        'Mini-Batch GD': lambda: mini_batch_gd(X, y, batch_size=64, epochs=epochs),
        'SGD':           lambda: sgd(X, y, epochs=epochs),
    }

    print(f"\n{'Variante':<20} {'Tiempo total (s)':>18} {'Tiempo/época (ms)':>20}")
    print("-" * 60)

    for nombre, fn in variantes.items():
        inicio = time.perf_counter()
        fn()
        fin = time.perf_counter()
        tiempo_total = fin - inicio
        tiempo_por_epoca_ms = (tiempo_total / epochs) * 1000
        print(f"{nombre:<20} {tiempo_total:>17.4f}s {tiempo_por_epoca_ms:>19.2f}ms")


def analizar_escalabilidad():
    """
    Analiza cómo escala el tiempo de MSE con el tamaño del dataset.
    """
    tamanios = [100, 1_000, 10_000, 100_000, 1_000_000]
    tiempos = []

    print(f"\n{'N muestras':<15} {'Tiempo MSE (ms)':>18}")
    print("-" * 35)

    for n in tamanios:
        y_true = np.random.rand(n)
        y_pred = np.random.rand(n)
        inicio = time.perf_counter()
        for _ in range(10):
            mse(y_true, y_pred)
        fin = time.perf_counter()
        t_ms = (fin - inicio) / 10 * 1000
        tiempos.append(t_ms)
        print(f"{n:<15,} {t_ms:>17.4f}ms")

    return tamanios, tiempos


# Ejecutar benchmarks
print("=" * 52)
print("BENCHMARK: FUNCIONES DE PÉRDIDA")
print("=" * 52)
benchmark_loss_functions()

print("\n" + "=" * 60)
print("BENCHMARK: VARIANTES DE GRADIENT DESCENT")
print("=" * 60)
benchmark_gradient_descent()

print("\n" + "=" * 35)
print("ESCALABILIDAD DE MSE")
print("=" * 35)
analizar_escalabilidad()
```

**¿Qué resultados debemos esperar?**

- **Funciones de pérdida:** MSE y MAE deberían ser muy rápidas (~0.1-1 ms para 10k muestras). BCE será ligeramente más lenta por el cálculo de logaritmos.
- **Variantes de GD:** Batch GD tendrá el tiempo por época más predecible. SGD será el más lento en tiempo total por los loops de Python. Mini-Batch GD será el más eficiente.
- **Escalabilidad:** MSE debería escalar aproximadamente de forma lineal con el número de muestras (complejidad O(n)), lo que confirma la eficiencia de la vectorización de NumPy.

### Criterios de Comparación

Al evaluar implementaciones, considera estos cuatro ejes:

| Criterio | ¿Qué medir? | ¿Cuándo priorizar? |
|----------|-------------|-------------------|
| **Velocidad** | `time.perf_counter()`, repeticiones | Producción, datasets grandes |
| **Memoria** | Evitar copias innecesarias, in-place ops | Datasets que no caben en RAM |
| **Claridad** | ¿Se entiende qué hace el código? | Educación, prototipado |
| **Mantenibilidad** | ¿Es fácil modificar/extender? | Proyectos a largo plazo |

---

## 🎯 EJERCICIOS PROPUESTOS

### Ejercicio 1: Implementar Huber Loss (Básico)

```python
Huber(y, ŷ) = 0.5(y-ŷ)² si |y-ŷ| ≤ δ
            = δ|y-ŷ| - 0.5δ² en otro caso
```

### Ejercicio 2: Learning Rate Scheduler (Intermedio)

Implementa:
```python
lr_new = lr_initial * decay_rate^epoch
```

Compara con LR fijo.

### Ejercicio 3: Early Stopping Mejorado (Intermedio)

- Monitorea pérdida de validación
- Guarda mejor modelo
- Restaura al finalizar

### Ejercicio 4: Comparación de Optimizadores (Avanzado)

Compara:
- Batch GD
- SGD
- Mini-batch GD
- Momentum (bonus)

### Ejercicio 5: Detección Automática de Overfitting (Proyecto)

Sistema que:
- Detecta divergencia train/val
- Recomienda λ de regularización
- Aplica early stopping automáticamente

---

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
Lab04_Entrega/
├── codigo/
│   └── [archivos .py]
├── notebooks/
│   └── experimentos.ipynb
├── reporte/
│   └── reporte_lab04.pdf
└── README.md
```

---

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

---

## 📊 Rúbrica de Evaluación

| Criterio | Excelente (90-100%) | Bueno (75-89%) | Satisfactorio (60-74%) | Insuficiente (<60%) |
|----------|-------------------|---------------|---------------------|-------------------|
| **Implementación** | Impecable, eficiente, documentado | Funcional con docs | Básico funcional | Con errores |
| **Experimentación** | Análisis profundo | Completo | Básico | Incompleto |
| **Documentación** | Excelente | Buena | Básica | Pobre |
| **Comprensión** | Dominio total | Buen entendimiento | Comprensión básica | Comprensión limitada |

---

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

---

## 🎓 Notas Finales

### Conceptos Clave para Recordar

1. **Pérdida cuantifica error:** Entre predicciones y realidad
2. **MSE para regresión:** Penaliza errores grandes
3. **Cross-Entropy para clasificación:** Interpretación probabilística
4. **Gradient descent minimiza:** Ajusta parámetros iterativamente
5. **Learning rate crucial:** Balance velocidad/estabilidad
6. **Mini-batch es estándar:** Mejor que batch o SGD puro
7. **Overfitting es común:** Entrenar mucho en pocos datos
8. **Regularización ayuda:** L2, L1, dropout, early stopping

### Preparación para el Siguiente Lab

**Lab 05: Backpropagation**

Aprenderás:
- Chain rule para redes
- Grafos computacionales
- Cálculo eficiente de gradientes
- Implementación completa de backprop

Prepárate repasando cálculo y chain rule.

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

Has completado exitosamente Lab 04 cuando puedas:

- [ ] Comprender qué son funciones de pérdida y su propósito
- [ ] Implementar MSE, MAE, Cross-Entropy desde cero
- [ ] Elegir función de pérdida apropiada para cada problema
- [ ] Implementar gradient descent básico
- [ ] Comprender efecto del learning rate
- [ ] Reconocer y detectar overfitting
- [ ] Calcular derivadas de funciones de pérdida
- [ ] Combinar pérdida con activación eficientemente
- [ ] Aplicar regularización básica

**¡Felicitaciones!** Continúa con el siguiente laboratorio.

---

**¿Preguntas?** Revisa teoría, experimenta, y consulta referencias.

**¡Éxito en tu aprendizaje! 🚀**
