# Guía de Laboratorio: Métricas de Evaluación y Matriz de Confusión

## 📋 Información del Laboratorio

**Título:** Fundamentos de Deep Learning - Métricas de Evaluación  
**Código:** Lab 07  
**Duración:** 2-3 horas  
**Nivel:** Intermedio  

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender y construir matrices de confusión para clasificación
2. Calcular e interpretar métricas fundamentales (Accuracy, Precision, Recall, F1-Score)
3. Identificar cuándo usar cada métrica según el problema de negocio
4. Trabajar efectivamente con datasets balanceados y desbalanceados
5. Implementar validación cruzada (K-Fold) desde cero
6. Analizar errores del modelo sistemáticamente
7. Visualizar resultados de evaluación de manera efectiva
8. Tomar decisiones informadas sobre umbrales de clasificación
9. Calcular métricas para clasificación multiclase
10. Generar reportes de evaluación profesionales

## 📚 Prerrequisitos

### Conocimientos

- Python intermedio (NumPy, manipulación de datos)
- Redes neuronales básicas y entrenamiento (Labs 05-06)
- Conceptos de clasificación binaria y multiclase
- Estadística básica (promedios, distribuciones)

### Software

- Python 3.8+
- NumPy 1.19+
- Matplotlib y Seaborn (visualizaciones)
- Scikit-learn (métricas de referencia)
- Pandas (manipulación de datos)

### Material de Lectura

Antes de comenzar, lee:
- `teoria.md` - Marco teórico completo sobre métricas
- `README.md` - Estructura del laboratorio
- Labs anteriores (especialmente Lab 06 sobre Entrenamiento)

## 📖 Introducción

### El Problema de la Evaluación

Has entrenado un modelo y obtuviste 95% de accuracy. ¿Excelente, verdad?

**No necesariamente.** Imagina esto:

```
Dataset de fraude bancario:
- 9,500 transacciones legítimas (95%)
- 500 transacciones fraudulentas (5%)

Modelo "tonto" que siempre predice "NO FRAUDE":
- Accuracy: 95%
- Fraudes detectados: 0
- ¡Completamente inútil!
```

**Este laboratorio te enseña a evaluar modelos correctamente.**

### ¿Por Qué Necesitamos Múltiples Métricas?

Diferentes problemas requieren diferentes métricas:

**Detección de Spam:**
- Falso Positivo (FP): Email importante marcado como spam → **MUY MALO**
- Falso Negativo (FN): Spam en inbox → Tolerable
- **Métrica clave: Precision** (minimizar FP)

**Detección de Cáncer:**
- Falso Positivo (FP): Persona sana diagnosticada → Tolerable (más pruebas)
- Falso Negativo (FN): Cáncer no detectado → **MUY MALO**
- **Métrica clave: Recall** (minimizar FN)

**Clasificación General:**
- Ambos errores igualmente importantes
- **Métrica clave: F1-Score** (balance)

### La Matriz de Confusión: Tu Mejor Amiga

La matriz de confusión muestra **exactamente** dónde se equivoca tu modelo:

```
                    Predicción
                 Positivo  Negativo
              ┌──────────┬──────────┐
Real       P  │    TP    │    FN    │
           N  │    FP    │    TN    │
              └──────────┴──────────┘
```

- **TP (True Positives)**: ✓ Correctamente identificados como positivos
- **TN (True Negatives)**: ✓ Correctamente identificados como negativos
- **FP (False Positives)**: ✗ Negativos incorrectamente marcados como positivos (Error Tipo I)
- **FN (False Negatives)**: ✗ Positivos incorrectamente marcados como negativos (Error Tipo II)

### Métricas Fundamentales

```
Accuracy   = (TP + TN) / Total          → Proporción de aciertos
Precision  = TP / (TP + FP)             → De las predicciones +, ¿cuántas correctas?
Recall     = TP / (TP + FN)             → De los casos + reales, ¿cuántos detectamos?
F1-Score   = 2 * (P * R) / (P + R)      → Media armónica de Precision y Recall
```

### Aplicaciones en el Mundo Real

**Medicina:**
- Diagnóstico de enfermedades (Recall crítico)
- Análisis de imágenes médicas (Balance P/R)

**Finanzas:**
- Detección de fraude (Recall crítico)
- Aprobación de créditos (Precision importante)

**E-commerce:**
- Sistemas de recomendación (Precision para UX)
- Detección de reseñas falsas (Balance)

**Seguridad:**
- Detección de intrusiones (Recall crítico)
- Sistemas de autenticación (Balance)

## 🤔 Preguntas de Reflexión Iniciales

1. ¿Por qué accuracy no siempre es una buena métrica?
2. ¿Qué métrica usarías para un detector de bombas en aeropuertos?
3. ¿Cómo afecta el desbalance de clases a la evaluación?
4. ¿Qué significa "recall de 80%"?
5. ¿Cuándo preferirías precision sobre recall?

## 🔬 Parte 1: Matriz de Confusión (40 min)

### 1.1 Implementación Desde Cero

La **matriz de confusión** es una tabla cuadrada de dimensión K×K (donde K es el número de clases) que resume el rendimiento de un clasificador comparando las etiquetas predichas con las etiquetas reales. Cada fila representa la **clase verdadera** de las muestras, mientras que cada columna representa la **clase predicha** por el modelo; esta convención es fundamental para interpretar correctamente los valores. La **diagonal principal** contiene los aciertos del modelo —es decir, los casos en que la predicción coincide con la realidad—, mientras que los elementos **fuera de la diagonal** representan errores, indicando confusiones entre pares de clases específicas. Construir esta clase desde cero, en lugar de simplemente llamar a `sklearn.metrics.confusion_matrix`, obliga al estudiante a entender la estructura interna del cálculo: el conteo de co-ocurrencias entre cada par (clase_real, clase_predicha), lo cual desarrolla intuición sobre cómo interpretar cada celda. Al finalizar, se espera obtener una clase reutilizable con métodos de visualización que permitan identificar de un vistazo cuáles son los pares de clases más confundidos por el modelo.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrix:
    """Matriz de confusión con visualización"""
    
    def __init__(self, y_true, y_pred, labels=None):
        """
        y_true: etiquetas verdaderas
        y_pred: predicciones del modelo
        labels: nombres de las clases (opcional)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.labels = labels
        
        # Calcular matriz
        self.matrix = self._compute_matrix()
        
        # Para clasificación binaria, extraer TP, TN, FP, FN
        if self.matrix.shape == (2, 2):
            self.tn = self.matrix[0, 0]
            self.fp = self.matrix[0, 1]
            self.fn = self.matrix[1, 0]
            self.tp = self.matrix[1, 1]
    
    def _compute_matrix(self):
        """Computa la matriz de confusión"""
        classes = np.unique(np.concatenate([self.y_true, self.y_pred]))
        n_classes = len(classes)
        
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                matrix[i, j] = np.sum(
                    (self.y_true == true_class) & (self.y_pred == pred_class)
                )
        
        return matrix
    
    def plot(self, normalize=False, cmap='Blues', figsize=(8, 6)):
        """Visualiza la matriz de confusión"""
        matrix = self.matrix.astype(float)
        
        if normalize:
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
            fmt = '.2%'
            title = 'Matriz de Confusión (Normalizada)'
        else:
            fmt = 'd'
            title = 'Matriz de Confusión'
        
        plt.figure(figsize=figsize)
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, 
                   xticklabels=self.labels if self.labels else 'auto',
                   yticklabels=self.labels if self.labels else 'auto',
                   cbar_kws={'label': 'Frecuencia' if not normalize else 'Proporción'})
        
        plt.xlabel('Predicción', fontsize=12)
        plt.ylabel('Valor Real', fontsize=12)
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Imprime resumen de la matriz"""
        print("=" * 60)
        print("MATRIZ DE CONFUSIÓN")
        print("=" * 60)
        print(self.matrix)
        
        if self.matrix.shape == (2, 2):
            print(f"\nTrue Negatives  (TN): {self.tn}")
            print(f"False Positives (FP): {self.fp}")
            print(f"False Negatives (FN): {self.fn}")
            print(f"True Positives  (TP): {self.tp}")
            
            total = self.tn + self.fp + self.fn + self.tp
            print(f"\nTotal de muestras: {total}")
            print(f"  Negativos reales: {self.tn + self.fp}")
            print(f"  Positivos reales: {self.fn + self.tp}")
        
        print("=" * 60)

# Ejemplo de uso
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1])

cm = ConfusionMatrix(y_true, y_pred, labels=['Negativo', 'Positivo'])
cm.summary()
cm.plot()
```

### 1.2 Caso de Estudio: Detector de Spam

Un concepto fundamental en evaluación de clasificadores es la **asimetría en el costo de los errores**: no todos los tipos de error tienen la misma gravedad para el negocio o el usuario. En un detector de spam, un **Falso Positivo** (un correo legítimo marcado como spam) puede hacer que el usuario pierda un mensaje importante —como una confirmación de vuelo o una oferta de trabajo—, lo que constituye un error **muy grave**. Por el contrario, un **Falso Negativo** (un mensaje de spam que pasa al inbox) es simplemente molesto pero no causa daño real, resultando en un error **tolerable**. Esta asimetría debería influir directamente en el diseño del modelo: en lugar del umbral por defecto de 0.5, convendría usar un umbral más alto para predecir "spam", aceptando más FN a cambio de reducir los FP. Para un clasificador de ~85% de accuracy en un conjunto donde el 30% son spam, se espera que la mayoría de los errores sean FN (spam no detectado), ya que esa estrategia conservadora protege mejor los correos legítimos.

```python
# Simular predicciones de un detector de spam
np.random.seed(42)

# Generar datos
n_samples = 1000
true_spam_rate = 0.3

# Etiquetas verdaderas
y_true = np.random.binomial(1, true_spam_rate, n_samples)

# Simular predicciones (modelo con ~85% accuracy)
y_pred = y_true.copy()
# Introducir algunos errores
error_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
y_pred[error_indices] = 1 - y_pred[error_indices]

# Crear matriz de confusión
cm_spam = ConfusionMatrix(y_true, y_pred, labels=['Ham', 'Spam'])

print("DETECTOR DE SPAM")
cm_spam.summary()
cm_spam.plot(normalize=True)

# Interpretación
print("\nINTERPRETACIÓN:")
print(f"- {cm_spam.tp} spam detectados correctamente")
print(f"- {cm_spam.fn} spam que pasaron (NO detectados) ⚠️")
print(f"- {cm_spam.fp} emails legítimos marcados como spam ⚠️⚠️")
print(f"- {cm_spam.tn} emails legítimos clasificados correctamente")
```

**Actividad 1.1:** Crea una matriz de confusión para un problema médico (detección de enfermedad). Documenta cuántos FP y FN obtuviste y reflexiona sobre cuál es más grave en el contexto médico.

### 1.3 Matriz de Confusión Multiclase

Cuando el problema tiene K > 2 clases, la matriz de confusión se extiende a una tabla K×K donde cada celda (i, j) contiene el número de ejemplos de la clase real i que fueron predichos como clase j. El análisis de esta matriz sigue el enfoque **"uno contra el resto"** (One vs. Rest): para cada clase k, se evalúa cuántos de sus ejemplos fueron correctamente identificados (celda diagonal) y hacia qué otras clases tiende a confundirse (celdas fuera de la diagonal en la fila k). En la inspección visual, lo más importante es identificar los pares de clases con mayor confusión mutua, ya que esto revela si hay similitudes semánticas o de representación que el modelo no logra diferenciar. La **matriz de confusión normalizada** (dividiendo cada fila por el total de muestras de esa clase) es especialmente reveladora en problemas multiclase: permite comparar el rendimiento por clase independientemente de cuántas muestras tiene cada una, exponiendo clases donde el modelo rinde pobremente aunque representen pocos ejemplos. Se espera que las celdas diagonales tengan valores cercanos a 1.0 en un modelo bien entrenado, con errores concentrados entre clases visualmente similares.

```python
# Ejemplo con 3 clases
y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred_multi = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2, 2, 1, 2])

cm_multi = ConfusionMatrix(
    y_true_multi, 
    y_pred_multi, 
    labels=['Clase A', 'Clase B', 'Clase C']
)

print("CLASIFICACIÓN MULTICLASE")
cm_multi.summary()
cm_multi.plot(normalize=True)
```

## 🔬 Parte 2: Métricas de Clasificación (50 min)

### 2.1 Implementación de Métricas Básicas

Cada métrica de clasificación captura un aspecto diferente del comportamiento del modelo, y elegir la correcta es tan importante como diseñar la arquitectura. A continuación se desarrolla la intuición detrás de cada una:

- **Accuracy** `= (TP + TN) / Total`: Mide la fracción de predicciones correctas sobre el total. Es apropiada cuando las clases están balanceadas y los errores tienen el mismo costo, pero se vuelve **engañosa** en datasets desbalanceados —un clasificador que siempre predice la clase mayoritaria puede tener 99% de accuracy y ser completamente inútil.

- **Precision** `= TP / (TP + FP)`: Responde a la pregunta *"de todas las veces que el modelo dijo 'positivo', ¿cuántas veces tenía razón?"*. Alta precisión significa pocos falsos positivos; es la métrica clave cuando el costo de una alarma falsa es alto (p. ej., spam filters, sistemas de aprobación de crédito).

- **Recall (Sensibilidad)** `= TP / (TP + FN)`: Responde a *"de todos los casos positivos reales, ¿cuántos detectó el modelo?"*. Alto recall significa que el modelo "no se pierde" casos positivos; es crítico cuando el costo de no detectar un positivo es alto (p. ej., diagnóstico médico, detección de fraude).

- **F1-Score** `= 2·(P·R)/(P+R)`: La **media armónica** de Precision y Recall. A diferencia de la media aritmética, la armónica penaliza fuertemente cuando uno de los dos valores es bajo: un modelo con Precision=1.0 y Recall=0.0 obtiene F1=0, no 0.5. Esto lo hace más informativo cuando existe un balance entre ambos objetivos.

- **Specificity** `= TN / (TN + FP)`: También llamada "True Negative Rate", mide qué tan bien el modelo identifica los negativos reales. Es la contraparte del Recall para la clase negativa; en medicina se conoce como "especificidad de la prueba".

- **F-beta Score** `= (1+β²)·(P·R)/(β²·P+R)`: Generalización del F1 que permite controlar el balance entre Precision y Recall. Con **β < 1** se da más peso a Precision (útil cuando FP son más costosos); con **β > 1** se prioriza Recall (útil cuando FN son más costosos). El F2-Score (β=2) es común en detección médica.

- **MCC (Matthews Correlation Coefficient)**: Considerado por muchos investigadores como la métrica individual más informativa para clasificación binaria, ya que considera los cuatro valores de la matriz de confusión (TP, TN, FP, FN) de forma simétrica. Tiene rango [-1, 1], donde 1 es predicción perfecta, 0 equivale a una predicción aleatoria y -1 indica predicción completamente inversa. A diferencia del F1, no se ve distorsionado por el desbalance de clases.

La implementación desde cero de esta clase consolidará la comprensión de cada fórmula y permitirá ver cómo interactúan entre sí en el reporte final.

```python
class ClassificationMetrics:
    """Calculadora de métricas de clasificación"""
    
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        
        # Calcular matriz de confusión
        self.cm = ConfusionMatrix(y_true, y_pred)
        
        if hasattr(self.cm, 'tp'):  # Clasificación binaria
            self.tp = self.cm.tp
            self.tn = self.cm.tn
            self.fp = self.cm.fp
            self.fn = self.cm.fn
    
    def accuracy(self):
        """
        Accuracy = (TP + TN) / Total
        Proporción de predicciones correctas
        """
        correct = np.sum(self.y_true == self.y_pred)
        total = len(self.y_true)
        return correct / total
    
    def precision(self, zero_division=0):
        """
        Precision = TP / (TP + FP)
        De las predicciones positivas, ¿cuántas fueron correctas?
        """
        denominator = self.tp + self.fp
        if denominator == 0:
            return zero_division
        return self.tp / denominator
    
    def recall(self, zero_division=0):
        """
        Recall = TP / (TP + FN)
        De los positivos reales, ¿cuántos detectamos?
        """
        denominator = self.tp + self.fn
        if denominator == 0:
            return zero_division
        return self.tp / denominator
    
    def f1_score(self):
        """
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        Media armónica de Precision y Recall
        """
        p = self.precision()
        r = self.recall()
        
        if p + r == 0:
            return 0.0
        
        return 2 * (p * r) / (p + r)
    
    def specificity(self):
        """
        Specificity = TN / (TN + FP)
        De los negativos reales, ¿cuántos identificamos?
        """
        denominator = self.tn + self.fp
        if denominator == 0:
            return 0.0
        return self.tn / denominator
    
    def f_beta_score(self, beta=1.0):
        """
        F-beta score: permite dar más peso a Precision o Recall
        
        beta < 1: Más peso a Precision
        beta > 1: Más peso a Recall
        beta = 1: F1-Score
        """
        p = self.precision()
        r = self.recall()
        
        if p + r == 0:
            return 0.0
        
        beta_squared = beta ** 2
        return (1 + beta_squared) * (p * r) / (beta_squared * p + r)
    
    def matthews_correlation_coefficient(self):
        """
        MCC: Correlación entre predicciones y realidad
        Rango: [-1, 1]
        1: Perfecto, 0: Aleatorio, -1: Totalmente incorrecto
        """
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = np.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * 
            (self.tn + self.fp) * (self.tn + self.fn)
        )
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def report(self):
        """Genera reporte completo de métricas"""
        print("=" * 70)
        print("REPORTE DE EVALUACIÓN")
        print("=" * 70)
        
        print("\nMATRIZ DE CONFUSIÓN:")
        print(f"  TN: {self.tn:5d}  |  FP: {self.fp:5d}")
        print(f"  FN: {self.fn:5d}  |  TP: {self.tp:5d}")
        
        print("\nMÉTRICAS PRINCIPALES:")
        print(f"  Accuracy:    {self.accuracy():.4f}  ({self.accuracy()*100:.2f}%)")
        print(f"  Precision:   {self.precision():.4f}  ({self.precision()*100:.2f}%)")
        print(f"  Recall:      {self.recall():.4f}  ({self.recall()*100:.2f}%)")
        print(f"  F1-Score:    {self.f1_score():.4f}  ({self.f1_score()*100:.2f}%)")
        
        print("\nMÉTRICAS ADICIONALES:")
        print(f"  Specificity: {self.specificity():.4f}  ({self.specificity()*100:.2f}%)")
        print(f"  F2-Score:    {self.f_beta_score(beta=2):.4f}")
        print(f"  MCC:         {self.matthews_correlation_coefficient():.4f}")
        
        print("\nINTERPRETACIÓN:")
        self._interpret()
        
        print("=" * 70)
    
    def _interpret(self):
        """Interpreta las métricas"""
        acc = self.accuracy()
        prec = self.precision()
        rec = self.recall()
        f1 = self.f1_score()
        
        # Balance Precision-Recall
        if abs(prec - rec) < 0.1:
            print("  ✓ Precision y Recall balanceados")
        elif prec > rec + 0.1:
            print("  ⚠️  Precision > Recall:")
            print("      - Modelo conservador (pocos positivos predichos)")
            print("      - Menos falsos positivos, más falsos negativos")
        else:
            print("  ⚠️  Recall > Precision:")
            print("      - Modelo liberal (muchos positivos predichos)")
            print("      - Menos falsos negativos, más falsos positivos")
        
        # Accuracy vs F1
        if acc > f1 + 0.1:
            print("  ⚠️  Accuracy >> F1-Score:")
            print("      - Posible desbalance de clases")
            print("      - Revisar distribución del dataset")

# Ejemplo
metrics = ClassificationMetrics(y_true, y_pred)
metrics.report()
```

### 2.2 Comparación Visual de Métricas

La comparación visual de métricas entre modelos es esencial para la selección de modelos, ya que los números en una tabla pueden resultar difíciles de interpretar en conjunto. Cuando se optimiza un modelo para una única métrica —por ejemplo, maximizar Accuracy— se corre el riesgo de degradar silenciosamente otras métricas igualmente importantes: un modelo que maximiza Accuracy en datos desbalanceados puede tener Recall cercano a cero. Los **gráficos de barras** con múltiples métricas permiten ver de un vistazo el "perfil" del modelo: un modelo bien balanceado mostrará barras de altura similar para Precision y Recall, mientras que un modelo sesgado mostrará una barra alta en una y baja en la otra. Un **perfil ideal** presenta Accuracy, Precision, Recall y F1 todos por encima de 0.85, sin diferencias mayores a 0.10 entre ellos; cuando Accuracy supera a F1 en más de 0.15 puntos, se debe investigar el balance de clases del dataset. La comparación entre un modelo con métricas balanceadas vs. un modelo aleatorio (baseline) también es fundamental para validar que el modelo realmente aprendió algo útil.

```python
def plot_metrics_comparison(y_true, y_pred_list, model_names):
    """
    Compara métricas de múltiples modelos
    
    y_true: etiquetas verdaderas
    y_pred_list: lista de predicciones de diferentes modelos
    model_names: nombres de los modelos
    """
    metrics_dict = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for y_pred in y_pred_list:
        m = ClassificationMetrics(y_true, y_pred)
        metrics_dict['Accuracy'].append(m.accuracy())
        metrics_dict['Precision'].append(m.precision())
        metrics_dict['Recall'].append(m.recall())
        metrics_dict['F1-Score'].append(m.f1_score())
    
    # Graficar
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        bars = ax.bar(model_names, values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(values))))
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} por Modelo', fontsize=13)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Ejemplo: Comparar 3 modelos
model1_pred = y_pred
model2_pred = np.random.binomial(1, 0.5, len(y_true))  # Modelo aleatorio
model3_pred = y_true.copy()  # Modelo perfecto
model3_pred[np.random.choice(len(y_true), 10, replace=False)] = 1 - model3_pred[np.random.choice(len(y_true), 10, replace=False)]

plot_metrics_comparison(
    y_true,
    [model1_pred, model2_pred, model3_pred],
    ['Modelo A', 'Modelo Random', 'Modelo B']
)
```

**Actividad 2.1:** Crea 3 modelos con diferentes balances Precision-Recall y compáralos. Observa cómo el perfil de barras cambia y reflexiona sobre cuál modelo elegirías para cada contexto de aplicación.

### 2.3 Efecto del Umbral de Decisión

En clasificación probabilística, el modelo no produce directamente una etiqueta binaria sino una **probabilidad** entre 0 y 1. El **umbral de decisión** (por defecto 0.5) es el valor a partir del cual se decide predecir "positivo": si p(x) ≥ umbral → Positivo. El valor de 0.5 es una elección arbitraria que asume que ambos tipos de error tienen el mismo costo y que las clases están balanceadas; en la práctica, este umbral rara vez es el óptimo. Cuando se **sube el umbral** (p. ej., a 0.7), el modelo se vuelve más conservador: solo predice "positivo" cuando está muy seguro, lo que aumenta la Precision pero reduce el Recall (más FN). Cuando se **baja el umbral** (p. ej., a 0.3), el modelo es más agresivo: predice "positivo" con menos certeza, aumentando el Recall pero reduciendo la Precision (más FP). La **curva Precision-Recall** visualiza este tradeoff para todos los umbrales posibles, y su área bajo la curva (AUCPR) resume la calidad del modelo independientemente del umbral elegido. El **punto de operación** óptimo se selecciona según los requisitos del negocio: si FN son más costosos, se elige un umbral bajo; si FP son más costosos, se elige un umbral alto. El máximo del F1-Score a lo largo de los umbrales indica el punto de mejor balance.

```python
def analyze_threshold_effect(y_true, y_proba, thresholds=np.linspace(0, 1, 21)):
    """
    Analiza cómo diferentes umbrales afectan las métricas
    
    y_true: etiquetas verdaderas
    y_proba: probabilidades predichas (0 a 1)
    thresholds: umbrales a probar
    """
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if len(np.unique(y_pred)) == 1:  # Solo una clase predicha
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            continue
        
        m = ClassificationMetrics(y_true, y_pred)
        precisions.append(m.precision())
        recalls.append(m.recall())
        f1_scores.append(m.f1_score())
    
    # Visualizar
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'r-s', label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, 'g-^', label='F1-Score', linewidth=2)
    plt.xlabel('Umbral de Clasificación', fontsize=12)
    plt.ylabel('Valor de Métrica', fontsize=12)
    plt.title('Métricas vs Umbral', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(recalls, precisions, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Curva Precision-Recall', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Marcar punto óptimo (max F1)
    max_f1_idx = np.argmax(f1_scores)
    plt.plot(recalls[max_f1_idx], precisions[max_f1_idx], 'r*', 
            markersize=20, label=f'Max F1 (threshold={thresholds[max_f1_idx]:.2f})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Encontrar umbral óptimo
    max_f1_idx = np.argmax(f1_scores)
    print(f"\nUMBRAL ÓPTIMO (maximiza F1):")
    print(f"  Threshold: {thresholds[max_f1_idx]:.2f}")
    print(f"  Precision: {precisions[max_f1_idx]:.4f}")
    print(f"  Recall:    {recalls[max_f1_idx]:.4f}")
    print(f"  F1-Score:  {f1_scores[max_f1_idx]:.4f}")

# Ejemplo
# Generar probabilidades
y_proba = np.random.beta(2, 5, len(y_true))  # Probabilidades sesgadas
y_proba[y_true == 1] += 0.3  # Positivos tienen mayor probabilidad

analyze_threshold_effect(y_true, y_proba)
```

**Actividad 2.2:** Encuentra el umbral óptimo para un problema donde FN son 2x más costosos que FP. Documenta el umbral seleccionado, las métricas resultantes y compara con el umbral que maximiza F1.

## 🔬 Parte 3: Datasets Desbalanceados (40 min)

### 3.1 El Problema del Desbalance

El **desbalance de clases** ocurre cuando una o más clases tienen significativamente más muestras que otras en el conjunto de datos. Este fenómeno es extremadamente común en aplicaciones reales: en detección de fraude bancario, apenas el 0.1–1% de las transacciones son fraudulentas; en diagnóstico de enfermedades raras, los casos positivos pueden representar menos del 1%; en clasificación de tráfico de red, el tráfico malicioso es una fracción mínima del tráfico total legítimo. El problema fundamental es que la **Accuracy se vuelve una métrica completamente engañosa**: si el 95% de las muestras son de clase negativa, un clasificador que **siempre predice negativo** (el "clasificador mayoritario naïve") obtiene 95% de Accuracy sin haber aprendido absolutamente nada. Este clasificador naïve debe usarse siempre como **baseline** en problemas desbalanceados: cualquier modelo real debe superar este umbral trivial en métricas relevantes (Recall, F1, MCC). El verdadero indicador de utilidad en estos contextos es el Recall de la clase minoritaria —si el modelo no detecta al menos una fracción razonable de los casos positivos reales, es inutilizable— junto con el F1-Score que penaliza simultáneamente los falsos positivos y negativos.

```python
# Crear dataset muy desbalanceado (95% negativos, 5% positivos)
n_samples = 1000
n_positives = 50
n_negatives = 950

y_true_imbalanced = np.array([0] * n_negatives + [1] * n_positives)
np.random.shuffle(y_true_imbalanced)

# Modelo "tonto" que siempre predice negativo
y_pred_dummy = np.zeros_like(y_true_imbalanced)

# Modelo real con 80% accuracy en ambas clases
y_pred_real = y_true_imbalanced.copy()
# Errores en negativos
neg_indices = np.where(y_true_imbalanced == 0)[0]
error_neg = np.random.choice(neg_indices, size=int(0.2 * len(neg_indices)), replace=False)
y_pred_real[error_neg] = 1

# Errores en positivos
pos_indices = np.where(y_true_imbalanced == 1)[0]
error_pos = np.random.choice(pos_indices, size=int(0.2 * len(pos_indices)), replace=False)
y_pred_real[error_pos] = 0

# Comparar
print("DATASET DESBALANCEADO (95% negativos, 5% positivos)")
print("\n1. Modelo Dummy (siempre predice negativo):")
metrics_dummy = ClassificationMetrics(y_true_imbalanced, y_pred_dummy)
metrics_dummy.report()

print("\n2. Modelo Real (80% accuracy en cada clase):")
metrics_real = ClassificationMetrics(y_true_imbalanced, y_pred_real)
metrics_real.report()

print("\n¡CONCLUSIÓN!")
print("El modelo dummy tiene 95% accuracy pero es inútil.")
print("El modelo real tiene ~90% accuracy y sí detecta positivos.")
print("→ Accuracy NO es suficiente en datasets desbalanceados!")
```

### 3.2 Técnicas para Datos Desbalanceados

Existen tres grandes estrategias para lidiar con el desbalance de clases, cada una con sus ventajas y desventajas:

**1. Sobremuestreo (Oversampling) de la clase minoritaria:** Duplica o genera nuevas muestras artificiales de la clase minoritaria hasta igualar el número de muestras de la clase mayoritaria. La versión básica (Random Oversampling) simplemente duplica muestras existentes; versiones avanzadas como SMOTE generan muestras sintéticas interpolando entre vecinos. *Pros:* Simple, no pierde información del conjunto original. *Contras:* Riesgo de **overfitting** sobre las muestras duplicadas, ya que el modelo puede memorizar exactamente esas instancias en lugar de generalizar.

**2. Submuestreo (Undersampling) de la clase mayoritaria:** Reduce aleatoriamente la clase mayoritaria hasta igualar el tamaño de la minoritaria, descartando muestras. *Pros:* Reduce el tiempo de entrenamiento, puede eliminar ruido de la clase mayoritaria. *Contras:* **Pérdida de información potencialmente valiosa** al descartar muestras legítimas; no recomendable cuando el dataset ya es pequeño.

**3. Pesos de clase (Class Weights):** Modifica la función de pérdida para asignar un penalización mayor a los errores en la clase minoritaria, sin alterar el dataset en sí. El peso de cada clase es inversamente proporcional a su frecuencia: `w_k = N_total / (K × N_k)`. *Pros:* Usa todas las muestras disponibles, es más estable que el resampling. *Contras:* Requiere que el algoritmo soporte class weights; puede ser más difícil de ajustar el balance correcto.

**¿Cuándo usar cada estrategia?** Si el dataset tiene suficientes muestras de la clase minoritaria (>500), prefiere **class weights** por su simplicidad. Si las muestras son muy pocas (<100), usa **oversampling** para aumentar la diversidad. Si el tiempo de entrenamiento es crítico y el dataset es muy grande, considera **undersampling** con cuidado.

```python
class ImbalancedDataHandler:
    """Herramientas para manejar datos desbalanceados"""
    
    @staticmethod
    def oversample_minority(X, y):
        """Sobremuestreo de la clase minoritaria"""
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        minority_indices = np.where(y == minority_class)[0]
        majority_count = np.max(counts)
        
        # Duplicar minoritarios hasta igualar mayoritarios
        n_to_add = majority_count - len(minority_indices)
        additional_indices = np.random.choice(minority_indices, size=n_to_add, replace=True)
        
        all_indices = np.concatenate([np.arange(len(y)), additional_indices])
        
        return X[all_indices], y[all_indices]
    
    @staticmethod
    def undersample_majority(X, y):
        """Submuestreo de la clase mayoritaria"""
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Reducir mayoritarios al tamaño de minoritarios
        sampled_majority = np.random.choice(
            majority_indices, 
            size=len(minority_indices), 
            replace=False
        )
        
        balanced_indices = np.concatenate([minority_indices, sampled_majority])
        np.random.shuffle(balanced_indices)
        
        return X[balanced_indices], y[balanced_indices]
    
    @staticmethod
    def class_weights(y):
        """
        Calcula pesos para balancear clases
        Peso inversamente proporcional a frecuencia
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        weights = {}
        for cls, count in zip(unique, counts):
            weights[cls] = total / (len(unique) * count)
        
        return weights

# Ejemplo
X_dummy = np.arange(len(y_true_imbalanced)).reshape(-1, 1)

print("TÉCNICAS DE BALANCEO\n")

# Original
print(f"Original: {len(y_true_imbalanced)} muestras")
print(f"  Clase 0: {np.sum(y_true_imbalanced == 0)}")
print(f"  Clase 1: {np.sum(y_true_imbalanced == 1)}")

# Oversample
X_over, y_over = ImbalancedDataHandler.oversample_minority(X_dummy, y_true_imbalanced)
print(f"\nOversample: {len(y_over)} muestras")
print(f"  Clase 0: {np.sum(y_over == 0)}")
print(f"  Clase 1: {np.sum(y_over == 1)}")

# Undersample
X_under, y_under = ImbalancedDataHandler.undersample_majority(X_dummy, y_true_imbalanced)
print(f"\nUndersample: {len(y_under)} muestras")
print(f"  Clase 0: {np.sum(y_under == 0)}")
print(f"  Clase 1: {np.sum(y_under == 1)}")

# Pesos
weights = ImbalancedDataHandler.class_weights(y_true_imbalanced)
print(f"\nPesos de clase:")
for cls, weight in weights.items():
    print(f"  Clase {cls}: {weight:.4f}")
```

**Actividad 3.1:** Compara el rendimiento de un modelo entrenado en datos originales vs balanceados. Documenta específicamente el Recall de la clase minoritaria en cada caso y explica por qué las diferencias observadas tienen sentido.

## 🔬 Parte 4: Validación Cruzada (45 min)

### 4.1 K-Fold Cross-Validation

Evaluar un modelo con una única división train/test tiene un problema fundamental: la **alta varianza en la estimación del rendimiento**. Si el conjunto de test, por azar, contiene muestras "fáciles", la métrica será optimista; si contiene muestras "difíciles", será pesimista. Este problema se conoce en estadística como **varianza del estimador**. La **K-Fold Cross-Validation** resuelve esto dividiendo el dataset en K subconjuntos ("folds") de tamaño similar: en cada iteración, uno de los K folds se usa como conjunto de validación y los K-1 restantes como entrenamiento. Al rotar sistemáticamente cuál fold actúa como validación, **todas las muestras son usadas para validación exactamente una vez**, lo que produce K estimaciones de la métrica. El promedio de estas K estimaciones es un estimador más robusto del rendimiento real del modelo. La elección de K implica un tradeoff bias-varianza: **K=5** es el más utilizado en la práctica porque ofrece un buen balance entre costo computacional y varianza del estimador; **K=10** produce estimaciones más estables pero requiere más tiempo. Con K=N (Leave-One-Out), la varianza del estimador es mínima pero el costo computacional es prohibitivo. La variante **Stratified K-Fold** es especialmente importante en datasets desbalanceados: garantiza que la proporción de clases en cada fold sea representativa del dataset completo, evitando folds donde la clase minoritaria esté ausente o sobrerrepresentada.

```python
class KFoldCrossValidator:
    """Implementación de K-Fold Cross-Validation"""
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        """
        Genera índices de train/test para cada fold
        
        Retorna: generador de tuplas (train_idx, test_idx)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop
    
    def cross_validate(self, model, X, y, metric_fn=None):
        """
        Ejecuta cross-validation completa
        
        model: modelo con métodos fit() y predict()
        X, y: datos
        metric_fn: función de métrica (default: accuracy)
        """
        if metric_fn is None:
            metric_fn = lambda y_true, y_pred: np.mean(y_true == y_pred)
        
        scores = []
        fold_metrics = []
        
        print(f"Ejecutando {self.n_splits}-Fold Cross-Validation...")
        print("=" * 70)
        
        for fold, (train_idx, test_idx) in enumerate(self.split(X, y), 1):
            # Dividir datos
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Entrenar
            model.fit(X_train, y_train)
            
            # Predecir
            y_pred = model.predict(X_test)
            
            # Evaluar
            score = metric_fn(y_test, y_pred)
            scores.append(score)
            
            # Métricas detalladas
            metrics = ClassificationMetrics(y_test, y_pred)
            fold_metrics.append({
                'accuracy': metrics.accuracy(),
                'precision': metrics.precision(),
                'recall': metrics.recall(),
                'f1': metrics.f1_score()
            })
            
            print(f"Fold {fold}/{self.n_splits}: Score = {score:.4f}")
        
        print("=" * 70)
        
        # Resultados
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\nRESULTADOS:")
        print(f"  Mean Score: {mean_score:.4f} (+/- {std_score:.4f})")
        print(f"  Min Score:  {np.min(scores):.4f}")
        print(f"  Max Score:  {np.max(scores):.4f}")
        
        # Promediar métricas
        mean_metrics = {}
        for key in fold_metrics[0].keys():
            values = [m[key] for m in fold_metrics]
            mean_metrics[key] = (np.mean(values), np.std(values))
        
        print(f"\nMÉTRICAS PROMEDIO:")
        for key, (mean, std) in mean_metrics.items():
            print(f"  {key.capitalize():12s}: {mean:.4f} (+/- {std:.4f})")
        
        return {
            'scores': scores,
            'mean': mean_score,
            'std': std_score,
            'fold_metrics': fold_metrics
        }

# Ejemplo con modelo dummy
class DummyClassifier:
    """Clasificador simple para demostración"""
    
    def fit(self, X, y):
        # "Entrenar": guardar la clase más frecuente
        unique, counts = np.unique(y, return_counts=True)
        self.most_common_class = unique[np.argmax(counts)]
        return self
    
    def predict(self, X):
        # Predecir la clase más frecuente
        return np.full(len(X), self.most_common_class)

# Usar
cv = KFoldCrossValidator(n_splits=5, shuffle=True, random_state=42)
model = DummyClassifier()

results = cv.cross_validate(model, X_dummy, y_true_imbalanced)
```

### 4.2 Visualización de Resultados de CV

La varianza en los scores entre folds es una señal diagnóstica crucial sobre la **estabilidad del modelo**. Si los scores varían poco entre folds (desviación estándar < 0.03), el modelo es **robusto**: su rendimiento es predecible independientemente del subconjunto de datos usado, lo que genera confianza para su despliegue en producción. Si la varianza es alta (std > 0.05), el modelo es **sensible a la partición de datos**: puede estar sobreajustando al conjunto de entrenamiento o puede haber subconjuntos del dataset con características muy diferentes (heterogeneidad). Las **barras de error** en los gráficos de métricas promedio representan ±1 desviación estándar entre folds: barras cortas indican consistencia, barras largas indican inestabilidad. Para la **selección de modelos** con cross-validation, no solo se debe preferir el modelo con mayor media, sino también considerar el que tenga menor varianza: un modelo con media=0.87 y std=0.01 es preferible a uno con media=0.89 y std=0.08, especialmente en aplicaciones críticas. Los **intervalos de confianza** al 95% se pueden calcular como `media ± 1.96 × std / √K`, y son la forma correcta de reportar métricas en papers y reportes profesionales.

```python
def plot_cv_results(cv_results):
    """Visualiza resultados de cross-validation"""
    scores = cv_results['scores']
    fold_metrics = cv_results['fold_metrics']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica 1: Scores por fold
    ax1 = axes[0]
    folds = np.arange(1, len(scores) + 1)
    ax1.plot(folds, scores, 'bo-', linewidth=2, markersize=10)
    ax1.axhline(y=cv_results['mean'], color='r', linestyle='--', 
               label=f"Mean: {cv_results['mean']:.4f}")
    ax1.fill_between(folds, 
                     cv_results['mean'] - cv_results['std'],
                     cv_results['mean'] + cv_results['std'],
                     alpha=0.2, color='red')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Score por Fold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Métricas promedio
    ax2 = axes[1]
    metric_names = list(fold_metrics[0].keys())
    metric_means = [np.mean([m[name] for m in fold_metrics]) for name in metric_names]
    metric_stds = [np.std([m[name] for m in fold_metrics]) for name in metric_names]
    
    bars = ax2.bar(metric_names, metric_means, yerr=metric_stds, 
                   capsize=5, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names))))
    ax2.set_ylabel('Valor', fontsize=12)
    ax2.set_title('Métricas Promedio (con std)', fontsize=14)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for bar, mean, std in zip(bars, metric_means, metric_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

plot_cv_results(results)
```

**Actividad 4.1:** Implementa Stratified K-Fold que mantiene la proporción de clases en cada fold. Compara los resultados con el K-Fold estándar en un dataset desbalanceado y documenta las diferencias en la varianza entre folds.

## 📊 Análisis Final de Rendimiento

### Dashboard Completo de Evaluación

Un **reporte de evaluación profesional** debe integrar todas las perspectivas del rendimiento del modelo en un único documento coherente, facilitando tanto la toma de decisiones técnicas como la comunicación con stakeholders no técnicos. El flujo de trabajo estándar es: **entrenar** el modelo con los datos de entrenamiento → **evaluar** con el conjunto de test usando múltiples métricas → **interpretar** los resultados en el contexto del problema → **decidir** si el modelo es apto para producción o requiere ajustes. Para una **audiencia técnica**, el reporte debe incluir la matriz de confusión completa, todas las métricas con intervalos de confianza, la curva Precision-Recall y los resultados de cross-validation. Para una **audiencia no técnica** (gerencia, clientes), conviene traducir las métricas a términos de negocio: "el modelo detecta el 87% de los fraudes reales" en lugar de "Recall = 0.87". Los **intervalos de confianza** para las métricas son especialmente importantes cuando el conjunto de test es pequeño: con 100 muestras, una diferencia de 2% en Accuracy entre dos modelos puede no ser estadísticamente significativa. El dashboard que se implementa a continuación integra matriz de confusión normalizada, barras de métricas, distribución de predicciones, curva Precision-Recall y resumen textual en una única figura de referencia profesional.

```python
class EvaluationDashboard:
    """Dashboard completo para evaluación de modelos"""
    
    def __init__(self, y_true, y_pred, y_proba=None, model_name="Modelo"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.model_name = model_name
        
        self.cm = ConfusionMatrix(y_true, y_pred)
        self.metrics = ClassificationMetrics(y_true, y_pred)
    
    def generate_report(self):
        """Genera reporte completo con visualizaciones"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Matriz de confusión
        ax1 = plt.subplot(2, 3, 1)
        matrix = self.cm.matrix.astype(float)
        matrix_norm = matrix / matrix.sum(axis=1, keepdims=True)
        sns.heatmap(matrix_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax1,
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax1.set_title('Matriz de Confusión', fontsize=13)
        ax1.set_xlabel('Predicción')
        ax1.set_ylabel('Real')
        
        # 2. Métricas principales
        ax2 = plt.subplot(2, 3, 2)
        metrics_data = {
            'Accuracy': self.metrics.accuracy(),
            'Precision': self.metrics.precision(),
            'Recall': self.metrics.recall(),
            'F1-Score': self.metrics.f1_score()
        }
        bars = ax2.barh(list(metrics_data.keys()), list(metrics_data.values()),
                       color=plt.cm.viridis([0.3, 0.5, 0.7, 0.9]))
        ax2.set_xlim(0, 1)
        ax2.set_title('Métricas Principales', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, value) in enumerate(zip(bars, metrics_data.values())):
            ax2.text(value + 0.02, i, f'{value:.3f}', va='center')
        
        # 3. Distribución de predicciones
        ax3 = plt.subplot(2, 3, 3)
        pred_dist = np.bincount(self.y_pred, minlength=2)
        true_dist = np.bincount(self.y_true, minlength=2)
        x = np.arange(2)
        width = 0.35
        ax3.bar(x - width/2, true_dist, width, label='Real', alpha=0.8)
        ax3.bar(x + width/2, pred_dist, width, label='Predicho', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Negativo', 'Positivo'])
        ax3.set_ylabel('Cantidad')
        ax3.set_title('Distribución de Clases', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Precision-Recall si tenemos probabilidades
        if self.y_proba is not None:
            ax4 = plt.subplot(2, 3, 4)
            thresholds = np.linspace(0, 1, 50)
            precisions = []
            recalls = []
            
            for t in thresholds:
                y_pred_t = (self.y_proba >= t).astype(int)
                if len(np.unique(y_pred_t)) == 2:
                    m = ClassificationMetrics(self.y_true, y_pred_t)
                    precisions.append(m.precision())
                    recalls.append(m.recall())
                else:
                    precisions.append(0)
                    recalls.append(0)
            
            ax4.plot(recalls, precisions, 'b-', linewidth=2)
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Curva Precision-Recall', fontsize=13)
            ax4.grid(True, alpha=0.3)
        
        # 5. Errores
        ax5 = plt.subplot(2, 3, 5)
        error_data = {
            'True Neg': self.cm.tn,
            'False Pos': self.cm.fp,
            'False Neg': self.cm.fn,
            'True Pos': self.cm.tp
        }
        colors = ['green', 'red', 'orange', 'green']
        ax5.pie(error_data.values(), labels=error_data.keys(), autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax5.set_title('Distribución de Resultados', fontsize=13)
        
        # 6. Resumen de texto
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary = f"""
        RESUMEN - {self.model_name}
        
        Total de muestras: {len(self.y_true)}
        
        Matriz de Confusión:
          TN: {self.cm.tn}  |  FP: {self.cm.fp}
          FN: {self.cm.fn}  |  TP: {self.cm.tp}
        
        Métricas:
          Accuracy:  {self.metrics.accuracy():.4f}
          Precision: {self.metrics.precision():.4f}
          Recall:    {self.metrics.recall():.4f}
          F1-Score:  {self.metrics.f1_score():.4f}
          
        Interpretación:
          {"Buen balance P/R" if abs(self.metrics.precision() - self.metrics.recall()) < 0.1 else "Desbalance P/R"}
          {"Accuracy confiable" if abs(self.metrics.accuracy() - self.metrics.f1_score()) < 0.1 else "Revisar balance de clases"}
        """
        
        ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.suptitle(f'Dashboard de Evaluación - {self.model_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# Usar dashboard
dashboard = EvaluationDashboard(y_true, y_pred, y_proba, model_name="Detector de Fraude")
dashboard.generate_report()
```

## 🎯 EJERCICIOS PROPUESTOS

### Nivel Básico

**Ejercicio 1:** Matriz de Confusión Manual
```
Dadas predicciones y etiquetas verdaderas:
a) Construye la matriz de confusión manualmente
b) Calcula TP, TN, FP, FN
c) Computa Accuracy, Precision, Recall, F1
```

**Ejercicio 2:** Interpretación de Métricas
```
Dado un problema médico con estas métricas:
- Accuracy: 0.92
- Precision: 0.45
- Recall: 0.95
- F1: 0.61

a) ¿Qué nos dicen estas métricas?
b) ¿El modelo es útil?
c) ¿Qué tipo de errores comete más?
```

**Ejercicio 3:** Selección de Métrica
```
Para cada escenario, indica la métrica más importante:
a) Detector de spam en email
b) Diagnóstico de enfermedad mortal
c) Sistema de aprobación de créditos
d) Clasificación de imágenes balanceadas
```

### Nivel Intermedio

**Ejercicio 4:** Optimización de Umbral
```
Implementa un sistema que:
- Pruebe diferentes umbrales (0.1 a 0.9)
- Grafique Precision y Recall vs Umbral
- Encuentre el umbral que maximiza F1
- Compare con umbral que minimiza FN
```

**Ejercicio 5:** Manejo de Desbalance
```
Dado un dataset 90/10:
- Implementa 3 técnicas de balanceo
- Entrena un modelo en cada versión
- Compara métricas
- Recomienda la mejor aproximación
```

**Ejercicio 6:** K-Fold Completo
```
Implementa K-Fold CV que:
- Use K=5 folds
- Calcule todas las métricas
- Genere intervalo de confianza (95%)
- Visualice resultados por fold
```

### Nivel Avanzado

**Ejercicio 7:** Sistema de Evaluación Completo
```
Crea un sistema que:
- Genere matriz de confusión
- Calcule todas las métricas
- Ejecute K-Fold CV
- Genere dashboard visual
- Produzca reporte PDF
```

**Ejercicio 8:** ROC y AUC
```
Implementa desde cero:
- Curva ROC (TPR vs FPR)
- Cálculo de AUC
- Comparación de múltiples modelos
- Punto óptimo en la curva
```

**Ejercicio 9:** Análisis de Errores
```
Desarrolla un sistema que:
- Identifique patrones en errores (FP y FN)
- Clasifique errores por tipo
- Sugiera mejoras al modelo
- Visualice casos difíciles
```

## 📝 Entregables

### 1. Código Fuente
- `metrics.py`: Implementación de todas las métricas
- `confusion_matrix.py`: Clase de matriz de confusión
- `cross_validation.py`: K-Fold CV
- `evaluation_dashboard.py`: Dashboard visual
- `experiments.ipynb`: Notebook con experimentos

### 2. Experimentos
- Comparación de métricas en diferentes problemas
- Análisis de datasets desbalanceados
- Resultados de cross-validation
- Optimización de umbrales

### 3. Visualizaciones
- Matrices de confusión
- Curvas Precision-Recall
- Comparaciones de modelos
- Dashboards completos

### 4. Reporte (3-4 páginas)
- Análisis de diferentes métricas
- Casos de uso apropiados
- Manejo de desbalance
- Conclusiones y recomendaciones

## 🎯 Criterios de Evaluación (CDIO)

### Conceive (Concebir) - 25%
- [ ] Comprensión de cada métrica y su significado
- [ ] Identificación de métricas apropiadas por problema
- [ ] Diseño de estrategias de evaluación
- [ ] Análisis crítico de resultados

### Design (Diseñar) - 25%
- [ ] Implementación correcta de métricas
- [ ] Código limpio y modular
- [ ] Visualizaciones efectivas
- [ ] Sistema de evaluación robusto

### Implement (Implementar) - 30%
- [ ] Todas las métricas calculadas correctamente
- [ ] K-Fold CV funcional
- [ ] Manejo de casos edge (división por cero, etc.)
- [ ] Resultados reproducibles

### Operate (Operar) - 20%
- [ ] Experimentación exhaustiva
- [ ] Interpretación correcta de resultados
- [ ] Recomendaciones fundamentadas
- [ ] Documentación clara

## 📋 Rúbrica de Evaluación

| Criterio | Excelente (90-100%) | Bueno (75-89%) | Satisfactorio (60-74%) | Insuficiente (<60%) |
|----------|-------------------|--------------|---------------------|------------------|
| **Implementación** | Todas las métricas perfectas | Métricas correctas | Algunas métricas correctas | Errores en cálculos |
| **Comprensión** | Interpretación profunda | Buena interpretación | Interpretación básica | Interpretación pobre |
| **Experimentación** | Análisis exhaustivo | Buenos experimentos | Experimentos básicos | Experimentos insuficientes |
| **Visualización** | Dashboards profesionales | Buenas visualizaciones | Visualizaciones básicas | Visualizaciones pobres |
| **Aplicación** | Selección perfecta de métricas | Buena selección | Selección razonable | Selección inadecuada |

## 📚 Referencias Adicionales

### Papers y Libros
1. Powers, D. M. (2011). "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation"
2. Sokolova, M., & Lapalme, G. (2009). "A systematic analysis of performance measures for classification tasks"
3. "Pattern Recognition and Machine Learning" (Bishop) - Capítulo sobre evaluación

### Recursos Online
- Scikit-learn: Documentación de métricas
- Towards Data Science: Tutoriales sobre métricas
- Google ML Crash Course: Classification metrics
- Confusion Matrix Calculator (online tools)

### Herramientas
- `sklearn.metrics`: Implementación de referencia
- `seaborn`: Visualización de matrices
- `yellowbrick`: Visualizaciones ML avanzadas
- `mlxtend`: Plotting utilities

## 🎓 Notas Finales

### Guía Rápida de Métricas

**¿Qué métrica usar?**

```
Dataset balanceado → Accuracy
Dataset desbalanceado → F1-Score
FP muy costosos → Precision
FN muy costosos → Recall
Balance general → F1-Score
Multiclase → Macro/Weighted F1
```

### Checklist de Evaluación

Antes de confiar en un modelo:
- [ ] Calculé múltiples métricas (no solo accuracy)
- [ ] Analicé la matriz de confusión
- [ ] Consideré el balance de clases
- [ ] Usé cross-validation
- [ ] Interpreté resultados en contexto del problema
- [ ] Visualicé resultados
- [ ] Documenté hallazgos

### Errores Comunes

❌ **Confiar solo en accuracy en datos desbalanceados**
❌ **No considerar el costo de diferentes errores**
❌ **Olvidar normalizar matrices de confusión**
❌ **No usar cross-validation para estimación robusta**
❌ **Ignorar la distribución de clases**

### Reflexión Final

**La evaluación correcta es tan importante como el modelo mismo.**

Un modelo con 99% accuracy puede ser inútil.
Un modelo con 80% accuracy puede ser invaluable.

**Todo depende del contexto y las métricas correctas.**

### Próximos Pasos

En el siguiente laboratorio (Lab 08), aprenderás:
- Frameworks modernos (PyTorch, TensorFlow)
- Implementación eficiente de todo lo aprendido
- Métricas automatizadas
- Productización de modelos

¡La evaluación correcta es tan importante como el entrenamiento! 📊

---

**"Torture the data, and it will confess to anything." - Ronald Coase**

**¡La evaluación correcta es tan importante como el entrenamiento! 📊**
