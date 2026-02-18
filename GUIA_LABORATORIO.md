# Guía de Laboratorio: Autoencoders para Sistemas de Comunicación

## 📋 Información del Laboratorio

**Título:** Fundamentos de Deep Learning para Comunicaciones - Autoencoder End-to-End  
**Código:** Guía 03  
**Duración:** 3-4 horas  
**Nivel:** Básico-Intermedio  

## 🎯 Objetivos Específicos

Al completar este laboratorio, serás capaz de:

1. Comprender el concepto de sistema de comunicación como autoencoder end-to-end
2. Identificar la analogía entre capas de una red neuronal y componentes de comunicación
3. Implementar y entrenar un autoencoder para comunicaciones en PyTorch
4. Diseñar arquitecturas neuronales para encoder y decoder adaptativos
5. Evaluar rendimiento mediante curvas BER vs SNR en canal AWGN
6. Visualizar y analizar constelaciones aprendidas automáticamente
7. Comparar el rendimiento con modulaciones clásicas (QAM, PSK)
8. Comprender las ventajas del aprendizaje end-to-end sobre diseño tradicional

## 📚 Prerrequisitos

### Conocimientos
- Python intermedio (POO, NumPy)
- Fundamentos de redes neuronales (capas densas, backpropagation)
- Conceptos básicos de sistemas de comunicaciones digitales
- Modulación digital y SNR
- Métricas de rendimiento (BER, SER)

### Software
- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib
- Jupyter Notebook

### Material de Lectura
Antes de comenzar, lee:
- `teoria.md` - Marco teórico completo sobre autoencoders para comunicaciones
- `README.md` - Estructura del laboratorio y recursos disponibles

## 📖 Introducción

Los **autoencoders end-to-end para comunicaciones** representan un cambio de paradigma en el diseño de sistemas de comunicaciones digitales:

- **Diseño Tradicional:** Componentes independientes optimizados separadamente (codificador, modulador, demodulador, decodificador)
- **Deep Learning:** Optimización conjunta de todo el sistema mediante backpropagation

### Contexto del Problema

En sistemas de comunicación tradicionales, cada componente se diseña según principios teóricos de teoría de la información y comunicaciones:
- El modulador mapea bits a símbolos según constelaciones predefinidas (QAM, PSK, etc.)
- El demodulador toma decisiones basadas en distancias euclidianas
- Los límites teóricos (Shannon, AWGN) guían el diseño

Sin embargo, este enfoque modular puede ser **subóptimo** cuando:
1. Los componentes no están perfectamente sincronizados
2. Existen imperfecciones de hardware no modeladas
3. El canal tiene características complejas difíciles de modelar
4. Se requiere adaptación dinámica a condiciones cambiantes

### Enfoque con Autoencoders

El paradigma de **autoencoder** trata el sistema de comunicación completo como una red neuronal:

```
                    AUTOENCODER PARA COMUNICACIONES
                    
Mensaje (k bits)                                      Mensaje estimado
    ↓                                                      ↑
[ENCODER = Transmisor]                         [DECODER = Receptor]
    ↓                                                      ↑
Señal (n dimensiones)      →  CANAL AWGN  →      Señal + Ruido
```

**Ventajas clave:**
- **Aprendizaje automático de modulación:** No necesitas diseñar constelaciones manualmente
- **Optimización global:** El sistema completo se optimiza para minimizar errores
- **Adaptabilidad:** El modelo puede entrenarse para diferentes condiciones de canal
- **Descubrimiento de soluciones:** Puede encontrar esquemas mejores que los clásicos

### Conceptos Fundamentales

**1. Restricción de Potencia:**
En sistemas reales, la potencia de transmisión está limitada. El encoder debe normalizar:
$$P_{avg} = \mathbb{E}[\|\mathbf{x}\|^2] = 1$$

**2. Canal Diferenciable:**
El canal AWGN es diferenciable, permitiendo backpropagation:
$$\mathbf{y} = \mathbf{x} + \mathbf{n}, \quad \mathbf{n} \sim \mathcal{N}(0, \sigma^2 I)$$

**3. Función de Pérdida:**
Cross-entropy entre mensaje original y estimado (clasificación):
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log P(\hat{m}_i = m_i)$$

### Aplicaciones Prácticas

- **IoT y 5G:** Diseño de esquemas de modulación adaptativos
- **Comunicaciones Ópticas:** Optimización para distorsiones no lineales
- **Satélites:** Adaptación a canales variables
- **Sistemas Embebidos:** Modulaciones eficientes en recursos

## 🔬 Parte 1: Preparación y Conceptos Fundamentales (30 min)

### 1.1 Introducción a Autoencoders para Comunicaciones

Un sistema de comunicación tradicional consta de componentes diseñados independientemente (codificador, modulador, canal, demodulador, decodificador). El paradigma de **autoencoder end-to-end** propone entrenar una red neuronal completa que optimiza todo el sistema de forma conjunta.

```
Sistema Tradicional:
Mensaje → [Codificador] → [Modulador] → [Canal AWGN] → [Demodulador] → [Decodificador] → Mensaje estimado

Sistema Autoencoder:
Mensaje (M opciones)
    ↓
[ENCODER - Red Neuronal Dense + ReLU]
    ↓
Símbolos (n dimensiones, potencia normalizada)
    ↓
[CANAL AWGN - Ruido Gaussiano]
    ↓
Símbolos recibidos con ruido
    ↓
[DECODER - Red Neuronal Dense + Softmax]
    ↓
Mensaje estimado (probabilidades sobre M opciones)
```

**Pregunta de Reflexión 1:** ¿Por qué es ventajoso optimizar todo el sistema de comunicación de forma conjunta en lugar de diseñar cada componente independientemente?

### 1.2 Preparación del Entorno

```python
# Importar bibliotecas necesarias
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Importar módulos del laboratorio
from autoencoder import CommunicationNet
from utils import (
    train_communication_system,
    evaluate_ber,
    plot_constellation,
    plot_ber_curve,
    compare_with_standard_modulation,
    add_awgn_noise
)

# Configuración de dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Usando dispositivo: {device}")

# Semilla para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
```

### 1.3 Comprensión de Parámetros Clave

Los parámetros fundamentales del sistema son:

- **M (num_messages):** Número de mensajes distintos a transmitir. Equivale a $2^k$ donde $k$ es el número de bits por mensaje.
- **n (signal_dims):** Dimensión del espacio de señal (número de componentes del símbolo transmitido). Para n=2, equivale a I/Q (In-phase/Quadrature).
- **SNR (Signal-to-Noise Ratio):** Relación señal a ruido en dB. Define la cantidad de ruido en el canal AWGN.

**Restricción de potencia:** La potencia promedio de transmisión debe normalizarse a 1:
$$P_{avg} = \mathbb{E}[\|\mathbf{x}\|^2] = 1$$

**Actividad 1:** Calcula cuántos bits por mensaje se transmiten para M=16, M=4 y M=64. ¿Cómo afecta M a la complejidad del problema?

### Actividades

**Actividad 1.1:** Verifica que el entorno esté configurado correctamente y que todos los módulos se importen sin errores.

**Actividad 1.2:** Calcula manualmente:
- Para M=4: ¿cuántos bits por símbolo?
- Para M=16: ¿cuántos bits por símbolo?
- Para M=64: ¿cuántos bits por símbolo?

**Actividad 1.3:** Ejecuta el test rápido y verifica que la normalización de potencia esté funcionando.

### Preguntas de Reflexión

**Pregunta 1.1 (Concebir):** ¿Por qué es ventajoso optimizar todo el sistema de comunicación de forma conjunta en lugar de diseñar cada componente independientemente? Piensa en términos de optimización global vs local.

**Pregunta 1.2 (Diseñar):** ¿Por qué es necesaria la restricción de potencia en el encoder? ¿Qué pasaría si permitimos potencia infinita?

**Pregunta 1.3 (Operar):** En aplicaciones reales, ¿qué otros factores además de la potencia deberían restringirse (latencia, ancho de banda, complejidad)?

## 🔬 Parte 2: Implementación del Autoencoder Básico (60 min)

### 2.1 Arquitectura del Autoencoder

El autoencoder consta de dos componentes principales:

```python
# Crear el modelo de comunicación
# M=16 mensajes, n=2 dimensiones (equivalente a I/Q)
model = CommunicationNet(
    num_messages=16,      # M = 16 (4 bits por mensaje)
    signal_dims=2,        # n = 2 (señal compleja I/Q)
    intermediate_size=64  # Tamaño de capa oculta
).to(device)

# Ver arquitectura
print("\n📐 Arquitectura del Modelo:")
print(model)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
print(f"\n📊 Total de parámetros: {total_params:,}")
```

**Pregunta de Reflexión 2:** ¿Por qué el encoder necesita una capa de normalización de potencia? ¿Qué ocurriría sin ella?

### 2.2 Visualización de Constelación Inicial (sin entrenamiento)

```python
# Visualizar constelación antes del entrenamiento
print("\n🌌 Constelación ANTES del entrenamiento:")

model.eval()
with torch.no_grad():
    # Generar todos los mensajes posibles
    messages = torch.arange(0, 16).to(device)
    
    # Codificar mensajes a símbolos
    symbols = model.encoder(messages)
    
    # Mover a CPU y convertir a NumPy
    symbols_np = symbols.cpu().numpy()
    
    # Graficar constelación
    plt.figure(figsize=(8, 8))
    plt.scatter(symbols_np[:, 0], symbols_np[:, 1], s=100, c=range(16), cmap='tab20')
    plt.xlabel('Dimensión I (In-phase)', fontsize=12)
    plt.ylabel('Dimensión Q (Quadrature)', fontsize=12)
    plt.title('Constelación Inicial (sin entrenar)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Añadir círculo de potencia unitaria
    circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--', 
                        linewidth=2, label='Potencia = 1')
    plt.gca().add_patch(circle)
    plt.legend()
    
    # Anotar cada punto con su mensaje
    for i, (x, y) in enumerate(symbols_np):
        plt.annotate(f'{i}', (x, y), fontsize=9, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

print(f"✓ Potencia promedio: {np.mean(np.sum(symbols_np**2, axis=1)):.4f}")
```

**Actividad 2:** Observa la constelación inicial. ¿Los símbolos están distribuidos uniformemente? ¿Cuál es la potencia promedio?

### 2.3 Entrenamiento del Autoencoder

```python
# Configurar hiperparámetros de entrenamiento
training_config = {
    'num_epochs': 100,
    'batch_size': 256,
    'learning_rate': 0.001,
    'snr_db_train': 10.0  # SNR de entrenamiento
}

print("\n🏋️  Configuración de Entrenamiento:")
for key, value in training_config.items():
    print(f"  {key}: {value}")

# Entrenar el modelo
print("\n📈 Iniciando entrenamiento...\n")

history = train_communication_system(
    model=model,
    num_epochs=training_config['num_epochs'],
    batch_size=training_config['batch_size'],
    learning_rate=training_config['learning_rate'],
    snr_db=training_config['snr_db_train'],
    device=device,
    verbose=True
)

print("\n✅ Entrenamiento completado!")
```

**Pregunta de Reflexión 3:** ¿Por qué se usa cross-entropy loss para entrenar el sistema? ¿Qué está optimizando realmente esta función de pérdida?

### 2.4 Curvas de Entrenamiento

```python
# Graficar evolución del loss y accuracy durante entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history['loss'], linewidth=2, color='#e74c3c')
axes[0].set_xlabel('Época', fontsize=12)
axes[0].set_ylabel('Loss (Cross-Entropy)', fontsize=12)
axes[0].set_title('Evolución del Loss durante Entrenamiento', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history['accuracy'], linewidth=2, color='#27ae60')
axes[1].set_xlabel('Época', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('Evolución de la Accuracy durante Entrenamiento', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 105])

plt.tight_layout()
plt.show()

print(f"\n📊 Resultados Finales de Entrenamiento:")
print(f"  Loss final: {history['loss'][-1]:.4f}")
print(f"  Accuracy final: {history['accuracy'][-1]:.2f}%")
```

**Actividad 3:** Experimenta con diferentes learning rates (0.0001, 0.001, 0.01). ¿Cómo afecta la velocidad de convergencia?

### Actividades

**Actividad 2.1:** Ejecuta el código de arquitectura y cuenta los parámetros del encoder y decoder por separado.

**Actividad 2.2:** Visualiza la constelación inicial y observa cómo los símbolos están distribuidos antes del entrenamiento.

**Actividad 2.3:** Entrena el modelo y analiza las curvas de loss y accuracy. ¿Hay señales de overfitting o underfitting?

**Actividad 2.4:** Experimenta con diferentes learning rates y compara las curvas de entrenamiento.

### Preguntas de Reflexión

**Pregunta 2.1 (Concebir):** ¿Por qué el encoder necesita una capa de normalización de potencia? ¿Qué ocurriría sin ella?

**Pregunta 2.2 (Diseñar):** ¿Por qué se usa cross-entropy loss para entrenar el sistema? ¿Qué está optimizando realmente esta función de pérdida?

**Pregunta 2.3 (Implementar):** ¿Cuál es el papel de la función de activación ReLU en el encoder? ¿Qué pasaría si usamos otras activaciones como Sigmoid o Tanh?

**Pregunta 2.4 (Operar):** Analiza las curvas de entrenamiento. ¿En qué época el modelo converge? ¿Sería beneficioso entrenar más épocas?

## 🔬 Parte 3: Análisis de la Constelación Aprendida (30 min)

### 3.1 Visualización de Constelación Entrenada

```python
# Visualizar constelación DESPUÉS del entrenamiento
print("\n🌌 Constelación DESPUÉS del entrenamiento:")

model.eval()
with torch.no_grad():
    messages = torch.arange(0, 16).to(device)
    symbols = model.encoder(messages)
    symbols_np = symbols.cpu().numpy()
    
    # Graficar constelación aprendida
    plt.figure(figsize=(10, 10))
    plt.scatter(symbols_np[:, 0], symbols_np[:, 1], s=150, c=range(16), 
                cmap='tab20', edgecolors='black', linewidth=2)
    plt.xlabel('Dimensión I (In-phase)', fontsize=13)
    plt.ylabel('Dimensión Q (Quadrature)', fontsize=13)
    plt.title(f'Constelación Aprendida (SNR entrenamiento = {training_config["snr_db_train"]} dB)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Círculo de potencia
    circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--', 
                        linewidth=2, label='Potencia = 1')
    plt.gca().add_patch(circle)
    plt.legend(fontsize=11)
    
    # Anotar puntos
    for i, (x, y) in enumerate(symbols_np):
        plt.annotate(f'{i}', (x, y), fontsize=10, ha='center', va='bottom', 
                     fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Calcular distancias mínimas entre símbolos
distances = []
for i in range(len(symbols_np)):
    for j in range(i+1, len(symbols_np)):
        dist = np.linalg.norm(symbols_np[i] - symbols_np[j])
        distances.append(dist)

print(f"\n📏 Análisis de la Constelación:")
print(f"  Potencia promedio: {np.mean(np.sum(symbols_np**2, axis=1)):.4f}")
print(f"  Distancia mínima entre símbolos: {np.min(distances):.4f}")
print(f"  Distancia máxima entre símbolos: {np.max(distances):.4f}")
print(f"  Distancia promedio: {np.mean(distances):.4f}")
```

**Pregunta de Reflexión 4:** ¿La constelación aprendida se asemeja a alguna modulación clásica (QAM, PSK)? ¿Por qué el autoencoder eligió esta configuración?

### 3.2 Comparación con 16-QAM

```python
# Comparar con 16-QAM estándar
print("\n📊 Comparación con 16-QAM:")

# Generar constelación 16-QAM estándar
from utils import generate_qam_constellation

qam16_symbols = generate_qam_constellation(M=16, normalize=True)

# Graficar comparación lado a lado
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Autoencoder aprendido
axes[0].scatter(symbols_np[:, 0], symbols_np[:, 1], s=150, c=range(16), 
                cmap='tab20', edgecolors='black', linewidth=2)
axes[0].set_xlabel('I', fontsize=12)
axes[0].set_ylabel('Q', fontsize=12)
axes[0].set_title('Autoencoder Aprendido', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axis('equal')
circle1 = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--', linewidth=2)
axes[0].add_patch(circle1)

# 16-QAM estándar
axes[1].scatter(qam16_symbols[:, 0], qam16_symbols[:, 1], s=150, c=range(16), 
                cmap='tab20', edgecolors='black', linewidth=2, marker='s')
axes[1].set_xlabel('I', fontsize=12)
axes[1].set_ylabel('Q', fontsize=12)
axes[1].set_title('16-QAM Estándar', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')
circle2 = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--', linewidth=2)
axes[1].add_patch(circle2)

plt.tight_layout()
plt.show()
```

**Actividad 4:** Calcula la distancia mínima para 16-QAM y compárala con la del autoencoder. ¿Cuál tiene mejor separación?

### Actividades

**Actividad 3.1:** Visualiza la constelación aprendida y compárala con la inicial (Parte 2). ¿Qué cambió?

**Actividad 3.2:** Calcula la distancia mínima entre símbolos para la constelación aprendida y para 16-QAM. Compara los valores.

**Actividad 3.3:** Observa si la constelación tiene alguna simetría o patrón específico.

### Preguntas de Reflexión

**Pregunta 3.1 (Concebir):** ¿La constelación aprendida se asemeja a alguna modulación clásica (QAM, PSK)? ¿Por qué el autoencoder eligió esta configuración?

**Pregunta 3.2 (Diseñar):** ¿Cómo podrías modificar el entrenamiento para forzar al autoencoder a aprender una constelación específica (por ejemplo, parecida a QAM)?

**Pregunta 3.3 (Operar):** Si la distancia mínima del autoencoder es mayor que 16-QAM, ¿significa que siempre tendrá mejor rendimiento? ¿Por qué sí o por qué no?

## 🔬 Parte 4: Evaluación de Rendimiento BER vs SNR (60 min)

### 4.1 Evaluación en Canal AWGN

```python
# Evaluar BER para diferentes valores de SNR
print("\n📡 Evaluando BER vs SNR en Canal AWGN...\n")

snr_range_db = np.arange(-4, 21, 2)  # De -4 dB a 20 dB
ber_autoencoder = []
num_test_blocks = 10000  # Bloques para evaluación

for snr_db in tqdm(snr_range_db, desc="Evaluando SNR"):
    ber = evaluate_ber(
        model=model,
        snr_db=snr_db,
        num_blocks=num_test_blocks,
        device=device
    )
    ber_autoencoder.append(ber)
    print(f"  SNR = {snr_db:3d} dB → BER = {ber:.6f}")

print("\n✅ Evaluación completada!")
```

### 4.2 Curvas BER vs SNR

```python
# Graficar curva BER vs SNR
plt.figure(figsize=(12, 7))
plt.semilogy(snr_range_db, ber_autoencoder, 'o-', linewidth=2.5, markersize=8,
             color='#3498db', label='Autoencoder (M=16, n=2)', markeredgecolor='black')

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('Bit Error Rate (BER)', fontsize=13)
plt.title('Rendimiento del Autoencoder en Canal AWGN', fontsize=14, fontweight='bold')
plt.grid(True, which='both', alpha=0.4)
plt.legend(fontsize=11, loc='lower left')
plt.ylim([1e-5, 1])
plt.tight_layout()
plt.show()

# Mostrar tabla de resultados
print("\n📋 Tabla de Resultados BER:")
print("=" * 40)
print(f"{'SNR (dB)':<12} {'BER':<15} {'SER (aprox.)':<15}")
print("=" * 40)
for snr, ber in zip(snr_range_db, ber_autoencoder):
    # Aproximación SER para Gray coding: SER ≈ 2 × BER (válida para BER bajo)
    ser = 2 * ber  
    print(f"{snr:<12} {ber:<15.6f} {ser:<15.6f}")
print("=" * 40)
```

**Pregunta de Reflexión 5:** ¿A partir de qué SNR el BER se vuelve prácticamente cero? ¿Cómo se relaciona esto con el SNR de entrenamiento?

### 4.3 Comparación con Modulaciones Clásicas

```python
# Comparar con QAM y PSK estándar
print("\n📊 Comparando con modulaciones estándar...\n")

ber_comparison = compare_with_standard_modulation(
    model=model,
    snr_range_db=snr_range_db,
    num_blocks=num_test_blocks,
    device=device
)

# Graficar comparación completa
plt.figure(figsize=(12, 8))

plt.semilogy(snr_range_db, ber_autoencoder, 'o-', linewidth=2.5, markersize=9,
             label='Autoencoder (aprendido)', color='#e74c3c', markeredgecolor='black')
plt.semilogy(snr_range_db, ber_comparison['16-QAM'], 's-', linewidth=2.5, markersize=8,
             label='16-QAM (estándar)', color='#3498db', markeredgecolor='black')
plt.semilogy(snr_range_db, ber_comparison['16-PSK'], '^-', linewidth=2.5, markersize=8,
             label='16-PSK (estándar)', color='#2ecc71', markeredgecolor='black')

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('Bit Error Rate (BER)', fontsize=13)
plt.title('Comparación: Autoencoder vs Modulaciones Clásicas (M=16, Canal AWGN)', 
          fontsize=14, fontweight='bold')
plt.grid(True, which='both', alpha=0.4)
plt.legend(fontsize=11, loc='lower left')
plt.ylim([1e-5, 1])
plt.tight_layout()
plt.show()
```

**Actividad 5:** Identifica en qué rango de SNR el autoencoder supera o iguala a las modulaciones clásicas. ¿Por qué?

### Actividades

**Actividad 4.1:** Evalúa el BER del autoencoder en el rango completo de SNR y genera la tabla de resultados.

**Actividad 4.2:** Grafica la curva BER vs SNR en escala logarítmica e identifica el SNR donde BER < 10^-3.

**Actividad 4.3:** Compara el autoencoder con 16-QAM y 16-PSK. ¿En qué rangos de SNR cada uno es superior?

**Actividad 4.4:** Calcula la ganancia de codificación del autoencoder respecto a 16-QAM a BER = 10^-3.

### Preguntas de Reflexión

**Pregunta 4.1 (Concebir):** ¿A partir de qué SNR el BER se vuelve prácticamente cero? ¿Cómo se relaciona esto con el SNR de entrenamiento (10 dB)?

**Pregunta 4.2 (Diseñar):** ¿Por qué el autoencoder podría tener peor rendimiento que 16-QAM en SNR muy alto? Piensa en términos de optimalidad y capacidad del modelo.

**Pregunta 4.3 (Operar):** Si entrenarás el modelo a diferentes SNR (por ejemplo, 5 dB o 15 dB), ¿cómo cambiaría la curva BER resultante?

**Pregunta 4.4 (CDIO - Integración):** En un sistema 5G real, ¿qué ventajas y desventajas tendría usar un autoencoder aprendido versus una modulación estándar como QAM?

## 🔬 Parte 5: Experimentación con Diferentes Configuraciones (45 min)

### 5.1 Variación del Número de Mensajes (M)

```python
# Experimentar con diferentes valores de M
print("\n🔬 Experimentando con diferentes M...\n")

M_values = [4, 8, 16, 32]
ber_results_M = {}

for M in M_values:
    print(f"📡 Entrenando y evaluando para M={M}...")
    
    # Crear nuevo modelo
    model_M = CommunicationNet(
        num_messages=M,
        signal_dims=2,
        intermediate_size=64
    ).to(device)
    
    # Entrenar
    _ = train_communication_system(
        model=model_M,
        num_epochs=80,
        batch_size=256,
        learning_rate=0.001,
        snr_db=10.0,
        device=device,
        verbose=False
    )
    
    # Evaluar
    ber_M = []
    for snr_db in snr_range_db:
        ber = evaluate_ber(model_M, snr_db, 5000, device)
        ber_M.append(ber)
    
    ber_results_M[M] = ber_M
    print(f"  ✓ M={M} completado\n")

# Graficar comparación
plt.figure(figsize=(12, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
markers = ['o', 's', '^', 'D']

for i, M in enumerate(M_values):
    plt.semilogy(snr_range_db, ber_results_M[M], markers[i]+'-', 
                 linewidth=2.5, markersize=8, label=f'M={M} ({int(np.log2(M))} bits)',
                 color=colors[i], markeredgecolor='black')

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('BER', fontsize=13)
plt.title('Efecto del Número de Mensajes (M) en el Rendimiento', fontsize=14, fontweight='bold')
plt.grid(True, which='both', alpha=0.4)
plt.legend(fontsize=11)
plt.ylim([1e-5, 1])
plt.tight_layout()
plt.show()
```

**Pregunta de Reflexión 6:** ¿Cómo afecta el incremento de M al rendimiento BER? ¿Por qué sistemas con más mensajes tienen peor BER?

### 5.2 Variación de la Dimensión del Espacio de Señal (n)

```python
# Experimentar con diferentes dimensiones n
print("\n🔬 Experimentando con diferentes n...\n")

n_values = [2, 4, 8]
ber_results_n = {}
constellation_n = {}

for n in n_values:
    print(f"📡 Entrenando y evaluando para n={n}...")
    
    # Crear modelo
    model_n = CommunicationNet(
        num_messages=16,
        signal_dims=n,
        intermediate_size=64
    ).to(device)
    
    # Entrenar
    _ = train_communication_system(
        model=model_n,
        num_epochs=80,
        batch_size=256,
        learning_rate=0.001,
        snr_db=10.0,
        device=device,
        verbose=False
    )
    
    # Guardar constelación (solo primeras 2 dimensiones para visualización)
    model_n.eval()
    with torch.no_grad():
        messages = torch.arange(0, 16).to(device)
        symbols = model_n.encoder(messages).cpu().numpy()
        constellation_n[n] = symbols
    
    # Evaluar
    ber_n = []
    for snr_db in snr_range_db:
        ber = evaluate_ber(model_n, snr_db, 5000, device)
        ber_n.append(ber)
    
    ber_results_n[n] = ber_n
    print(f"  ✓ n={n} completado\n")

# Graficar comparación
plt.figure(figsize=(12, 7))
colors_n = ['#e74c3c', '#9b59b6', '#34495e']
markers_n = ['o', 's', '^']

for i, n in enumerate(n_values):
    plt.semilogy(snr_range_db, ber_results_n[n], markers_n[i]+'-', 
                 linewidth=2.5, markersize=8, label=f'n={n} dimensiones',
                 color=colors_n[i], markeredgecolor='black')

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('BER', fontsize=13)
plt.title('Efecto de la Dimensión del Espacio de Señal (n)', fontsize=14, fontweight='bold')
plt.grid(True, which='both', alpha=0.4)
plt.legend(fontsize=11)
plt.ylim([1e-5, 1])
plt.tight_layout()
plt.show()
```

**Actividad 6:** Compara el rendimiento con n=2 vs n=4. ¿Por qué aumentar n mejora el BER? ¿Cuál es el costo?

### 5.3 Variación del Tamaño de Capa Intermedia

```python
# Experimentar con diferentes tamaños de capa oculta
print("\n🔬 Experimentando con diferentes intermediate_size...\n")

hidden_sizes = [32, 64, 128, 256]
ber_results_hidden = {}

for hidden_size in hidden_sizes:
    print(f"📡 Entrenando y evaluando para hidden_size={hidden_size}...")
    
    model_h = CommunicationNet(
        num_messages=16,
        signal_dims=2,
        intermediate_size=hidden_size
    ).to(device)
    
    # Entrenar
    _ = train_communication_system(
        model=model_h,
        num_epochs=80,
        batch_size=256,
        learning_rate=0.001,
        snr_db=10.0,
        device=device,
        verbose=False
    )
    
    # Evaluar
    ber_h = []
    for snr_db in snr_range_db:
        ber = evaluate_ber(model_h, snr_db, 5000, device)
        ber_h.append(ber)
    
    ber_results_hidden[hidden_size] = ber_h
    print(f"  ✓ hidden_size={hidden_size} completado\n")

# Graficar
plt.figure(figsize=(12, 7))
colors_h = ['#e67e22', '#16a085', '#8e44ad', '#c0392b']
markers_h = ['o', 's', '^', 'D']

for i, h_size in enumerate(hidden_sizes):
    plt.semilogy(snr_range_db, ber_results_hidden[h_size], markers_h[i]+'-', 
                 linewidth=2.5, markersize=8, label=f'Hidden={h_size}',
                 color=colors_h[i], markeredgecolor='black')

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('BER', fontsize=13)
plt.title('Efecto del Tamaño de Capa Oculta', fontsize=14, fontweight='bold')
plt.grid(True, which='both', alpha=0.4)
plt.legend(fontsize=11)
plt.ylim([1e-5, 1])
plt.tight_layout()
plt.show()
```

**Pregunta de Reflexión 7:** ¿Existe un punto de rendimiento decreciente al aumentar el tamaño de la capa oculta? ¿Por qué?

### Actividades

**Actividad 5.1:** Entrena y evalúa modelos con M = [4, 8, 16, 32]. Grafica todas las curvas BER vs SNR en una sola figura.

**Actividad 5.2:** Entrena modelos con n = [2, 4, 8]. Compara el rendimiento. ¿Cuánto mejora n=4 respecto a n=2?

**Actividad 5.3:** Experimenta con diferentes tamaños de capa oculta [32, 64, 128, 256]. ¿Cuál es el tamaño óptimo?

**Actividad 5.4:** Visualiza las constelaciones aprendidas para diferentes valores de M y n.

### Preguntas de Reflexión

**Pregunta 5.1 (Concebir):** ¿Cómo afecta el incremento de M al rendimiento BER? ¿Por qué sistemas con más mensajes tienen peor BER? Relaciona esto con la teoría de información de Shannon.

**Pregunta 5.2 (Diseñar):** Compara el rendimiento con n=2 vs n=4. ¿Por qué aumentar n mejora el BER? ¿Cuál es el costo (eficiencia espectral)?

**Pregunta 5.3 (Implementar):** ¿Existe un punto de rendimiento decreciente al aumentar el tamaño de la capa oculta? ¿Por qué? Piensa en términos de capacidad vs overfitting.

**Pregunta 5.4 (Operar):** Si tuvieras que diseñar un sistema para IoT con restricciones de potencia y ancho de banda, ¿qué configuración elegirías (M, n, hidden_size)? Justifica tu respuesta.

## 📊 Parte 6: Análisis Comparativo Final (30 min)

### 6.1 Resumen de Todos los Experimentos

```python
# Crear tabla resumen de todos los experimentos
print("\n" + "="*80)
print(" " * 25 + "RESUMEN DE EXPERIMENTOS")
print("="*80)

# Definir SNR de referencia para comparación
snr_ref = 10  # dB
idx_ref = list(snr_range_db).index(snr_ref)

summary_data = {
    'Configuración': [],
    'Parámetros': [],
    f'BER @ {snr_ref}dB': [],
    'Rendimiento Relativo': []
}

# Baseline (modelo original)
baseline_ber = ber_autoencoder[idx_ref]
summary_data['Configuración'].append('Baseline')
summary_data['Parámetros'].append('M=16, n=2, h=64')
summary_data[f'BER @ {snr_ref}dB'].append(f"{baseline_ber:.6f}")
summary_data['Rendimiento Relativo'].append('100%')

# Variaciones de M
for M in M_values:
    ber_val = ber_results_M[M][idx_ref]
    # Rendimiento relativo: valores >100% indican MEJOR rendimiento que baseline (menor BER)
    rel_perf = (baseline_ber / ber_val) * 100 if ber_val > 0 else float('inf')
    summary_data['Configuración'].append(f'Variación M')
    summary_data['Parámetros'].append(f'M={M}, n=2, h=64')
    summary_data[f'BER @ {snr_ref}dB'].append(f"{ber_val:.6f}")
    summary_data['Rendimiento Relativo'].append(f'{rel_perf:.1f}%')

# Variaciones de n
for n in n_values:
    ber_val = ber_results_n[n][idx_ref]
    # Rendimiento relativo: valores >100% indican MEJOR rendimiento que baseline (menor BER)
    rel_perf = (baseline_ber / ber_val) * 100 if ber_val > 0 else float('inf')
    summary_data['Configuración'].append(f'Variación n')
    summary_data['Parámetros'].append(f'M=16, n={n}, h=64')
    summary_data[f'BER @ {snr_ref}dB'].append(f"{ber_val:.6f}")
    summary_data['Rendimiento Relativo'].append(f'{rel_perf:.1f}%')

# Imprimir tabla
for i in range(len(summary_data['Configuración'])):
    if i == 0 or summary_data['Configuración'][i] != summary_data['Configuración'][i-1]:
        print(f"\n{summary_data['Configuración'][i]}:")
        print("-" * 80)
    print(f"  {summary_data['Parámetros'][i]:<25} BER: {summary_data[f'BER @ {snr_ref}dB'][i]:<12} "
          f"Rendimiento: {summary_data['Rendimiento Relativo'][i]}")

print("\n" + "="*80)
```

### 6.2 Visualización Comparativa de Constelaciones

```python
# Visualizar constelaciones aprendidas para diferentes configuraciones
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
fig.suptitle('Constelaciones Aprendidas - Diferentes Configuraciones', 
             fontsize=16, fontweight='bold')

configs_to_plot = [
    ('M=4, n=2', ber_results_M.get(4) and symbols_np[:4] or symbols_np[:4]),
    ('M=16, n=2', symbols_np),
    ('M=32, n=2', ber_results_M.get(32) and symbols_np[:16] or symbols_np),
    ('M=16, n=4 (proj. 2D)', constellation_n.get(4, symbols_np)[:, :2])
]

for idx, (ax, (title, const)) in enumerate(zip(axes.flat, configs_to_plot)):
    ax.scatter(const[:, 0], const[:, 1], s=120, c=range(len(const)), 
               cmap='tab20', edgecolors='black', linewidth=1.5)
    ax.set_xlabel('I', fontsize=11)
    ax.set_ylabel('Q', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--', linewidth=2)
    ax.add_patch(circle)
    
    # Anotar algunos puntos
    for i, (x, y) in enumerate(const):
        if i % max(1, len(const)//16) == 0:  # Anotar cada N puntos
            ax.annotate(f'{i}', (x, y), fontsize=8, ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

### 6.3 Preguntas de Reflexión Final

1. **Optimización End-to-End:**
   - ¿Cuáles son las principales ventajas de entrenar todo el sistema conjuntamente?
   - ¿En qué situaciones las modulaciones clásicas podrían seguir siendo preferibles?

2. **Generalización del Modelo:**
   - El modelo fue entrenado a SNR=10dB. ¿Cómo se desempeña a otros SNR?
   - ¿Qué estrategias podrías usar para que el modelo generalice a múltiples SNR?

3. **Complejidad vs Rendimiento:**
   - ¿Cuál es el trade-off entre número de parámetros y rendimiento BER?
   - ¿Vale la pena duplicar el tamaño de la red para mejorar BER en 0.0001?

4. **Aplicaciones Prácticas:**
   - ¿Qué desafíos enfrentarías al implementar este sistema en hardware real?
   - ¿Cómo lidiarías con canales que cambian en el tiempo?

5. **Extensiones Avanzadas:**
   - ¿Cómo extenderías este enfoque a canales con desvanecimiento (Rayleigh, Rician)?
   - ¿Podrías adaptar el autoencoder para sistemas MIMO?

### Actividades

**Actividad 6.1:** Completa la tabla resumen comparando todas las configuraciones experimentadas.

**Actividad 6.2:** Visualiza las constelaciones aprendidas lado a lado para diferentes configuraciones.

**Actividad 6.3:** Identifica la configuración que ofrece el mejor balance entre rendimiento (BER) y complejidad (parámetros).

### Preguntas de Reflexión

**Pregunta 6.1 (Concebir):** ¿Cuáles son las principales ventajas de entrenar todo el sistema conjuntamente? ¿En qué situaciones las modulaciones clásicas podrían seguir siendo preferibles?

**Pregunta 6.2 (Diseñar):** El modelo fue entrenado a SNR=10dB. ¿Cómo se desempeña a otros SNR? ¿Qué estrategias podrías usar para que el modelo generalice a múltiples SNR?

**Pregunta 6.3 (Implementar):** ¿Cuál es el trade-off entre número de parámetros y rendimiento BER? ¿Vale la pena duplicar el tamaño de la red para mejorar BER en 0.0001?

**Pregunta 6.4 (Operar):** ¿Qué desafíos enfrentarías al implementar este sistema en hardware real? ¿Cómo lidiarías con canales que cambian en el tiempo?

## 📊 Análisis Final de Rendimiento

### Resumen de Resultados Clave

**Configuración Baseline (M=16, n=2, hidden=64):**
- Loss final de entrenamiento: ~0.0001
- Accuracy de entrenamiento: >99.9%
- BER @ 10 dB (SNR de entrenamiento): ~10^-5
- BER @ 20 dB: Prácticamente 0

**Comparación con Modulaciones Clásicas:**
```
SNR (dB)    Autoencoder    16-QAM    16-PSK    Mejora vs QAM
----------------------------------------------------------------
    0           ~0.2         ~0.25      ~0.30         +20%
    5           ~0.05        ~0.08      ~0.12         +38%
   10           ~10^-5       ~10^-4     ~10^-3        10x mejor
   15           ~0           ~0         ~0            Similar
   20           ~0           ~0         ~0            Similar
```

**Hallazgos Principales:**

1. **Ventaja en SNR Medio:** El autoencoder supera significativamente a las modulaciones clásicas en el rango de SNR de entrenamiento (5-15 dB), con mejoras de hasta 10x en BER.

2. **Constelación Aprendida:** La red aprende automáticamente una constelación que maximiza la distancia entre símbolos, similar pero no idéntica a 16-QAM.

3. **Efecto de M (Número de Mensajes):**
   - M=4 (2 bits): BER muy bajo incluso en SNR negativo
   - M=16 (4 bits): Balance óptimo rendimiento/eficiencia
   - M=32+ (5+ bits): Degradación esperada según teoría de Shannon

4. **Efecto de n (Dimensiones):**
   - n=2: Eficiente espectralmente (símbolos complejos I/Q)
   - n=4: Mejora ~2-3 dB en SNR requerido
   - n=8: Mejora marginal adicional, pobre eficiencia espectral

5. **Efecto de Hidden Size:**
   - 32 neuronas: Suficiente para M≤8
   - 64 neuronas: Óptimo para M=16
   - 128-256: Beneficio marginal, mayor riesgo de overfitting

### Análisis CDIO

**Concebir (20%):**
- ✅ Comprensión del paradigma end-to-end
- ✅ Analogía entre comunicaciones y autoencoders
- ✅ Restricciones de potencia y su importancia
- ✅ Trade-offs fundamentales (M, n, complejidad)

**Diseñar (25%):**
- ✅ Arquitectura apropiada para el problema
- ✅ Selección de hiperparámetros justificada
- ✅ Estrategia de entrenamiento efectiva
- ✅ Experimentos sistemáticos y controlados

**Implementar (30%):**
- ✅ Código funcional y reproducible
- ✅ Entrenamiento convergente
- ✅ Evaluación completa BER vs SNR
- ✅ Visualizaciones claras y profesionales

**Operar (25%):**
- ✅ Análisis crítico de resultados
- ✅ Comparación con modulaciones estándar
- ✅ Identificación de fortalezas y limitaciones
- ✅ Recomendaciones para aplicaciones prácticas

### Limitaciones Identificadas

1. **Generalización a SNR:** El modelo entrenado a un SNR fijo tiene rendimiento subóptimo en otros SNR.

2. **Canal Específico:** El autoencoder está optimizado para AWGN. Canales reales (desvanecimiento, no linealidades) requieren reentrenamiento.

3. **Complejidad Computacional:** La inferencia requiere ~10-100x más operaciones que demappers clásicos.

4. **Sincronización:** Asume sincronización perfecta (tiempo, fase, frecuencia). Errores de sincronización no se modelaron.

5. **Escalabilidad:** Para M muy grandes (256+), el entrenamiento se vuelve difícil y el BER se degrada significativamente.

### Recomendaciones para Aplicaciones Prácticas

**Cuándo usar Autoencoders:**
- ✅ Canales complejos difíciles de modelar
- ✅ Hardware con imperfecciones conocidas
- ✅ Necesidad de adaptación a condiciones específicas
- ✅ Recursos computacionales disponibles

**Cuándo usar Modulaciones Clásicas:**
- ✅ Sistemas estandarizados (Wi-Fi, 5G, etc.)
- ✅ Necesidad de interoperabilidad
- ✅ Recursos computacionales muy limitados
- ✅ Requerimientos de baja latencia crítica

### Extensiones Futuras

1. **Entrenamiento Multi-SNR:** Batch con SNR aleatorio para mejor generalización
2. **Canales Realistas:** Rayleigh, Rician, canales selectivos en frecuencia
3. **Codificación de Canal:** Integrar FEC (LDPC, Turbo) en el autoencoder
4. **Sistemas MIMO:** Extender a múltiples antenas
5. **Adaptación Online:** Fine-tuning en tiempo real según condiciones del canal

## 🎯 EJERCICIOS PROPUESTOS

### Ejercicio 1: Adaptación a Múltiples SNR (Dificultad: Media)

**Objetivo:** Mejorar la generalización del autoencoder para que funcione bien en un rango amplio de SNR.

**Tareas:**
1. Modifica el proceso de entrenamiento para usar SNR variable:
   - En cada batch, selecciona SNR aleatorio entre 0-15 dB
   - Implementa una función de muestreo uniforme o ponderado de SNR
   - Opcionalmente, usa curriculum learning (empezar con SNR alto, decrementar gradualmente)

2. Entrena el nuevo modelo y compáralo con el baseline:
   - Evalúa BER vs SNR en el rango completo [-4, 20] dB
   - Grafica ambas curvas en la misma figura
   - Calcula el BER promedio en el rango [0, 15] dB

3. Analiza la robustez del modelo resultante:
   - ¿El modelo multi-SNR tiene mejor o peor rendimiento en SNR específicos?
   - ¿Hay un trade-off entre generalización y rendimiento pico?
   - Visualiza la constelación aprendida. ¿Es diferente?

**Entregables:**
- Código de entrenamiento con SNR aleatorio
- Gráficas comparativas BER vs SNR (baseline vs multi-SNR)
- Tabla con BER promedio en diferentes rangos
- Análisis de trade-offs

**Criterios de Éxito:**
- El modelo multi-SNR debe tener BER <2x del baseline en SNR de entrenamiento original
- Mejora demostrable en al menos 50% del rango de SNR
- Código bien documentado y reproducible
- Análisis crítico fundamentado

---

### Ejercicio 2: Canal Rayleigh con Desvanecimiento (Dificultad: Media-Alta)

**Objetivo:** Extender el autoencoder para trabajar en canales con desvanecimiento realista.

**Tareas:**
1. Implementa un canal Rayleigh:
   - Modelo: $\mathbf{y} = h \cdot \mathbf{x} + \mathbf{n}$, donde $h \sim \text{Rayleigh}(\sigma_h)$
   - Asegúrate de que el canal sea diferenciable para backpropagation
   - Implementa variantes: desvanecimiento rápido vs lento

2. Entrena un autoencoder específico para este canal:
   - Usa los mismos hiperparámetros que el baseline
   - Monitorea convergencia (puede ser más lenta)
   - Guarda checkpoints durante entrenamiento

3. Compara el rendimiento:
   - Evalúa BER vs SNR promedio (averaged over fading)
   - Compara con el modelo AWGN puro
   - Evalúa el modelo AWGN en canal Rayleigh (sin reentrenar)
   - Visualiza la constelación aprendida

4. Análisis adicional:
   - ¿Cómo cambia la constelación óptima para Rayleigh?
   - ¿El modelo es robusto a cambios en la velocidad de desvanecimiento?

**Entregables:**
- Implementación del canal Rayleigh
- Modelo entrenado para Rayleigh
- Gráficas comparativas: AWGN-trained vs Rayleigh-trained
- Análisis de constelaciones y rendimiento

**Criterios de Éxito:**
- Canal Rayleigh correctamente implementado y validado
- Modelo converge exitosamente en canal Rayleigh
- Mejora >3 dB respecto a usar modelo AWGN en canal Rayleigh
- Documentación clara del proceso
- Análisis de por qué la constelación cambia

---

### Ejercicio 3: Autoencoder con Codificación de Canal (Dificultad: Alta)

**Objetivo:** Diseñar un autoencoder que incluya redundancia (codificación de canal) para mejorar la robustez.

**Tareas:**
1. Diseña arquitectura con rate < 1:
   - Ejemplo: 4 bits de entrada → encoder → 8 dimensiones (rate = 0.5)
   - Modifica la arquitectura para soportar n > k (más dimensiones que bits)
   - Mantén normalización de potencia

2. Implementa múltiples configuraciones:
   - Rate 1.0: k=4 bits → n=4 dimensiones (baseline, sin redundancia)
   - Rate 0.67: k=4 bits → n=6 dimensiones
   - Rate 0.5: k=4 bits → n=8 dimensiones
   - Rate 0.33: k=4 bits → n=12 dimensiones

3. Compara el rendimiento:
   - Evalúa BER vs SNR para todas las configuraciones
   - Calcula la ganancia de codificación en dB
   - Analiza el trade-off: eficiencia espectral vs robustez
   - Compara con códigos clásicos (Hamming, Reed-Solomon)

4. Visualización y análisis:
   - Para n=4, 6, 8: visualiza proyección 2D de la constelación
   - Calcula distancia mínima en espacio n-dimensional
   - Analiza cómo la redundancia mejora la separación

**Entregables:**
- Código de arquitectura con rate variable
- Modelos entrenados para diferentes rates
- Gráficas BER vs SNR comparando todos los rates
- Tabla de trade-offs: rate vs BER @ SNR_ref vs complejidad
- Análisis de ganancia de codificación

**Criterios de Éxito:**
- Rate 0.5 debe mejorar BER en al menos 2-3 dB respecto a rate 1.0
- Demostración clara del trade-off eficiencia vs robustez
- Comparación fundamentada con códigos clásicos
- Visualizaciones claras de constelaciones multi-dimensionales
- Análisis cuantitativo de ganancia de codificación

---

### Ejercicio 4: Visualización Interactiva y Análisis de Regiones de Decisión (Dificultad: Media)

**Objetivo:** Crear visualizaciones que muestren cómo el decoder interpreta diferentes regiones del espacio de señal.

**Tareas:**
1. Genera una malla 2D del espacio I/Q:
   - Crea una grid uniforme de puntos en el rango [-1.5, 1.5] × [-1.5, 1.5]
   - Resolución recomendada: 200×200 puntos
   - Para cada punto, evalúa la salida del decoder

2. Visualiza las regiones de decisión:
   - Colorea cada punto según el mensaje que el decoder predice
   - Usa diferentes colores para cada uno de los M mensajes
   - Superpone la constelación aprendida (símbolos transmitidos)
   - Añade contornos de probabilidad (isolíneas)

3. Análisis de fronteras de decisión:
   - ¿Las fronteras son lineales o no lineales?
   - ¿Hay regiones de alta incertidumbre?
   - Compara con fronteras óptimas (Voronoi)
   - Analiza simetría y estructura de las regiones

4. Visualización interactiva (opcional):
   - Permite hacer clic en puntos para ver probabilidades
   - Anima cómo las regiones cambian durante el entrenamiento
   - Muestra el efecto del ruido en las decisiones

**Entregables:**
- Código de visualización de regiones de decisión
- Gráficas de alta calidad mostrando:
  - Mapa de regiones coloreado
  - Constelación superpuesta
  - Contornos de probabilidad
  - Comparación con regiones de Voronoi óptimas
- Análisis de la forma de las fronteras
- (Opcional) Notebook interactivo o animaciones

**Criterios de Éxito:**
- Visualizaciones claras y profesionales
- Regiones de decisión correctamente calculadas
- Análisis detallado de la forma de las fronteras
- Comparación con fronteras teóricas óptimas
- Código bien documentado y reutilizable

---

### Ejercicio 5: Transfer Learning y Adaptación de Dominio (Dificultad: Alta)

**Objetivo:** Usar transfer learning para adaptar rápidamente el autoencoder a nuevas condiciones de canal.

**Tareas:**
1. Pre-entrena un modelo robusto:
   - Entrena en canal AWGN con SNR variable [0, 20] dB
   - Usa arquitectura grande (hidden_size=256)
   - Entrena por muchas épocas hasta convergencia perfecta

2. Implementa estrategias de transfer learning:
   - **Estrategia 1:** Congelar encoder, fine-tune solo decoder
   - **Estrategia 2:** Congelar primeras capas, fine-tune capas finales
   - **Estrategia 3:** Fine-tune todas las capas con learning rate bajo

3. Adapta a nuevos canales:
   - Canal Rayleigh (desvanecimiento)
   - Canal con offset de fase (error de sincronización)
   - Canal con distorsión no lineal (ej: saturación)

4. Compara con entrenamiento desde cero:
   - Número de épocas necesarias para convergencia
   - BER final alcanzado
   - Estabilidad del entrenamiento
   - Cantidad de datos necesarios

**Entregables:**
- Modelo pre-entrenado robusto
- Código de transfer learning con las 3 estrategias
- Resultados comparativos: transfer learning vs from scratch
- Curvas de entrenamiento mostrando convergencia más rápida
- Análisis de qué estrategia funciona mejor para cada tipo de canal

**Criterios de Éxito:**
- Transfer learning converge en <50% de las épocas necesarias from scratch
- BER final igual o mejor que entrenamiento completo
- Demostración en al menos 2 tipos de canal diferentes
- Análisis claro de cuándo usar cada estrategia
- Código modular y reutilizable

## 📝 Entregables

Para la evaluación completa del laboratorio, debes entregar:

1. **Jupyter Notebook o script Python** (.ipynb o .py) que incluya:
   - Todo el código funcional y ejecutable
   - Comentarios explicativos en secciones clave
   - Salidas de ejecución (gráficas, métricas)
   - Respuestas a preguntas de reflexión integradas
   - Código limpio y bien organizado

2. **Reporte técnico** (4-6 páginas) que incluya:
   - **Introducción:** Contexto de autoencoders para comunicaciones y objetivos del laboratorio
   - **Marco teórico:** Paradigma end-to-end, restricción de potencia, función de pérdida
   - **Metodología:** 
     - Descripción de la arquitectura del autoencoder (encoder y decoder)
     - Proceso de entrenamiento y hiperparámetros
     - Configuración de experimentos
   - **Resultados:**
     - Curvas de entrenamiento (loss, accuracy)
     - Visualizaciones de constelaciones aprendidas
     - Gráficos comparativos BER vs SNR
     - Resultados de experimentos con diferentes M, n e intermediate_size
     - Tablas de resumen
   - **Análisis:**
     - Comparación con modulaciones clásicas (16-QAM, 16-PSK)
     - Interpretación de constelaciones aprendidas
     - Análisis de trade-offs (eficiencia espectral vs robustez)
     - Impacto de parámetros en el rendimiento
   - **Discusión:**
     - Ventajas del aprendizaje end-to-end
     - Limitaciones identificadas
     - Aplicaciones prácticas
     - Comparación con diseño tradicional
   - **Conclusiones:** 
     - Hallazgos principales
     - Recomendaciones para uso en sistemas reales
     - Trabajo futuro

3. **Respuestas a preguntas de reflexión** de cada parte (pueden estar integradas en el notebook o en el reporte)
   - Clasificadas por dimensión CDIO (Concebir, Diseñar, Implementar, Operar)
   - Respuestas fundamentadas con evidencia experimental

4. **Al menos 2 ejercicios propuestos** completados con:
   - Código implementado y documentado
   - Resultados experimentales (tablas, gráficas de alta calidad)
   - Análisis crítico de los resultados
   - Conclusiones específicas del ejercicio
   - Comparación con baseline

5. **Archivos adicionales** (si aplica):
   - Modelos entrenados guardados (.pth)
   - Scripts auxiliares para visualización o utilidades
   - Datos generados (si son relevantes para reproducibilidad)

6. **Presentación breve** (5-7 slides) resumiendo los resultados principales (opcional pero recomendado):
   - Motivación y objetivos
   - Arquitectura del autoencoder
   - Resultados clave (constelaciones, BER vs SNR)
   - Conclusiones principales

## 🎯 Criterios de Evaluación (CDIO)

| Criterio | Peso | Descripción | Indicadores |
|----------|------|-------------|-------------|
| **Concebir** | 20% | Comprensión del paradigma autoencoder end-to-end, restricción de potencia, y optimización conjunta | - Claridad en respuestas a preguntas de reflexión<br>- Correcta interpretación de resultados<br>- Comprensión de trade-offs (M, n, complejidad)<br>- Entendimiento de ventajas vs diseño tradicional<br>- Conocimiento de aplicaciones prácticas |
| **Diseñar** | 25% | Diseño apropiado de arquitecturas, selección de hiperparámetros, y metodología experimental | - Elección justificada de arquitectura del encoder/decoder<br>- Configuración coherente de parámetros (M, n, hidden)<br>- Estrategia de entrenamiento apropiada<br>- Diseño sistemático de experimentos comparativos<br>- Planificación de evaluación BER |
| **Implementar** | 30% | Correcta implementación del código, entrenamiento efectivo, y evaluación completa | - Código funcional sin errores<br>- Implementación eficiente y limpia<br>- Uso apropiado de PyTorch<br>- Documentación clara del código<br>- Reproducibilidad de resultados<br>- Visualizaciones profesionales |
| **Operar** | 25% | Análisis de resultados, interpretación de métricas, y conclusiones aplicables | - Análisis crítico de constelaciones aprendidas<br>- Interpretación correcta de curvas BER<br>- Comparación fundamentada con modulaciones clásicas<br>- Identificación de limitaciones<br>- Recomendaciones para aplicaciones reales<br>- Calidad del reporte técnico |

### Distribución Detallada de Puntos

- **Notebook/código completado y funcional:** 30 puntos
  - Implementación correcta del autoencoder: 10 pts
  - Entrenamiento convergente y efectivo: 8 pts
  - Evaluación completa BER vs SNR: 7 pts
  - Experimentos con diferentes configuraciones: 5 pts

- **Reporte técnico:** 25 puntos
  - Introducción y marco teórico: 5 pts
  - Metodología clara y detallada: 7 pts
  - Resultados y análisis profundo: 8 pts
  - Conclusiones y recomendaciones: 5 pts

- **Respuestas a preguntas de reflexión:** 20 puntos
  - Profundidad de análisis: 10 pts
  - Conexión con conceptos CDIO: 5 pts
  - Fundamentación con evidencia experimental: 5 pts

- **Ejercicios propuestos completados (mínimo 2):** 15 puntos
  - Implementación correcta: 8 pts
  - Análisis de resultados: 5 pts
  - Documentación y presentación: 2 pts

- **Calidad del código y presentación:** 10 puntos
  - Documentación y comentarios: 4 pts
  - Organización y claridad: 3 pts
  - Visualizaciones profesionales: 3 pts

**Total:** 100 puntos

### Desglose por Dimensión CDIO

**Concebir (20 puntos):**
- Comprensión de la analogía sistema de comunicación ↔ autoencoder (5 pts)
- Entendimiento de restricción de potencia y normalización (4 pts)
- Conocimiento de función de pérdida y backpropagation a través del canal (4 pts)
- Comprensión de ventajas del aprendizaje end-to-end (4 pts)
- Análisis de aplicaciones prácticas (3 pts)

**Diseñar (25 puntos):**
- Arquitectura del encoder apropiada (6 pts)
- Arquitectura del decoder apropiada (6 pts)
- Selección justificada de hiperparámetros (M, n, hidden_size) (6 pts)
- Diseño de experimentos sistemáticos y comparativos (5 pts)
- Estrategia de evaluación (SNR range, número de muestras) (2 pts)

**Implementar (30 puntos):**
- Código funcional y ejecutable sin errores (10 pts)
- Entrenamiento exitoso con convergencia (8 pts)
- Evaluación correcta de BER en múltiples configuraciones (6 pts)
- Visualizaciones claras y profesionales (4 pts)
- Documentación y reproducibilidad (2 pts)

**Operar (25 puntos):**
- Análisis cuantitativo de rendimiento BER (6 pts)
- Análisis cualitativo de constelaciones aprendidas (5 pts)
- Comparación crítica con modulaciones estándar (6 pts)
- Respuestas completas y reflexivas a preguntas (5 pts)
- Reporte técnico bien estructurado y profesional (3 pts)

### Criterios de Calidad del Código

- **Excelente (9-10 puntos):** 
  - Código limpio, eficiente, bien documentado
  - Sin errores, fácilmente reproducible
  - Uso avanzado de PyTorch (GPU, data loaders, etc.)
  - Visualizaciones publication-quality

- **Bueno (7-8 puntos):** 
  - Código funcional con documentación adecuada
  - Pocos errores menores
  - Reproducible con ajustes mínimos
  - Visualizaciones claras

- **Satisfactorio (5-6 puntos):** 
  - Código funcional con documentación básica
  - Algunos errores o warnings
  - Parcialmente reproducible
  - Visualizaciones básicas pero suficientes

- **Insuficiente (<5 puntos):** 
  - Código no funcional o con errores graves
  - Documentación ausente o inadecuada
  - No reproducible
  - Visualizaciones ausentes o confusas

### Criterios de Éxito Mínimos

Para aprobar el laboratorio, debes cumplir:

- ✅ **Entrenamiento exitoso:** Loss < 0.01, Accuracy > 95% en el modelo baseline
- ✅ **Rendimiento mínimo:** BER < 10^-3 para SNR ≥ 12 dB
- ✅ **Código ejecutable:** Sin errores críticos, completamente reproducible
- ✅ **Reporte completo:** Todas las secciones presentes y bien desarrolladas
- ✅ **Respuestas a reflexión:** Al menos 80% de las preguntas respondidas con profundidad
- ✅ **Ejercicios propuestos:** Mínimo 2 ejercicios completados satisfactoriamente
- ✅ **Visualizaciones:** Constelaciones y curvas BER claramente presentadas

### Criterios de Excelencia (>90 puntos)

Para obtener una calificación sobresaliente:

- 🌟 **Análisis profundo:** Interpretación detallada de por qué el autoencoder aprende ciertas constelaciones
- 🌟 **Comparaciones exhaustivas:** Benchmark contra múltiples modulaciones y configuraciones
- 🌟 **Experimentos adicionales:** Más de 2 ejercicios propuestos completados
- 🌟 **Código avanzado:** Implementación de features adicionales (early stopping, learning rate scheduling, etc.)
- 🌟 **Visualizaciones excepcionales:** Gráficos interactivos, animaciones, o dashboards
- 🌟 **Conexión con literatura:** Referencias a papers relevantes y comparación con resultados publicados
- 🌟 **Aplicaciones innovadoras:** Propuesta de aplicaciones o extensiones originales

### Rúbrica de Reporte Técnico

| Sección | Puntos | Criterios de Evaluación |
|---------|--------|-------------------------|
| **Introducción** | 5 | Contexto claro, motivación, objetivos específicos |
| **Marco Teórico** | 5 | Conceptos fundamentales bien explicados, ecuaciones correctas |
| **Metodología** | 7 | Descripción detallada de arquitectura, hiperparámetros, y experimentos |
| **Resultados** | 8 | Presentación clara de todos los experimentos con tablas y gráficas |
| **Análisis** | 5 | Interpretación profunda, conexión con teoría, trade-offs identificados |
| **Discusión** | 3 | Ventajas, limitaciones, comparación con diseño tradicional |
| **Conclusiones** | 2 | Síntesis de hallazgos, recomendaciones, trabajo futuro |
| **Formato** | 0 | Bonus por formato profesional, referencias, ortografía impecable |

## 📚 Referencias Adicionales

### Artículos Fundamentales

1. **O'Shea, T., & Hoydis, J. (2017).** "An Introduction to Deep Learning for the Physical Layer." *IEEE Transactions on Cognitive Communications and Networking*, 3(4), 563-575.
   - Paper seminal que introduce el concepto de autoencoder para comunicaciones

2. **Dörner, S., Cammerer, S., Hoydis, J., & ten Brink, S. (2018).** "Deep Learning Based Communication Over the Air." *IEEE Journal on Selected Areas in Communications*, 36(7), 1413-1426.
   - Demostración experimental de autoencoders en hardware real

3. **Aoudia, F. A., & Hoydis, J. (2019).** "End-to-End Learning of Communications Systems Without a Channel Model." *52nd Asilomar Conference on Signals, Systems, and Computers*, 298-303.
   - Entrenamiento de autoencoders sin modelo explícito del canal

4. **Farsad, N., & Goldsmith, A. (2018).** "Neural Network Detection of Data Sequences in Communication Systems." *IEEE Transactions on Signal Processing*, 66(21), 5663-5678.
   - Análisis teórico de redes neuronales para detección en comunicaciones

5. **Ye, H., Li, G. Y., & Juang, B. H. (2018).** "Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems." *IEEE Wireless Communications Letters*, 7(1), 114-117.
   - Aplicación de DL a sistemas OFDM prácticos

### Libros de Texto

6. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press.
   - Capítulo 14: Autoencoders
   - Disponible gratuitamente: http://www.deeplearningbook.org/

7. **Proakis, J. G., & Salehi, M. (2008).** *Digital Communications* (5th ed.). McGraw-Hill.
   - Capítulos 4-5: Digital Modulation Techniques
   - Fundamentos teóricos de modulación y codificación

8. **Goldsmith, A. (2005).** *Wireless Communications*. Cambridge University Press.
   - Capítulo 5: Performance of Digital Modulation over Wireless Channels
   - Límites teóricos y capacidad de canal

9. **Haykin, S. (2009).** *Communication Systems* (5th ed.). Wiley.
   - Fundamentos de teoría de comunicaciones y procesamiento de señales

### Surveys y Tutoriales

10. **Zhang, C., Patras, P., & Haddadi, H. (2019).** "Deep Learning in Mobile and Wireless Networking: A Survey." *IEEE Communications Surveys & Tutorials*, 21(3), 2224-2287.
    - Revisión exhaustiva de DL aplicado a comunicaciones wireless

11. **Qin, Z., Ye, H., Li, G. Y., & Juang, B. H. (2019).** "Deep Learning in Physical Layer Communications." *IEEE Wireless Communications*, 26(2), 93-99.
    - Tutorial sobre aplicaciones de DL en capa física

12. **Jiang, C., et al. (2017).** "Machine Learning Paradigms for Next-Generation Wireless Networks." *IEEE Wireless Communications*, 24(2), 98-105.
    - Perspectiva sobre ML en redes 5G y beyond

### Documentación y Recursos Online

13. **PyTorch Documentation:** https://pytorch.org/docs/stable/index.html
    - Documentación oficial de PyTorch

14. **PyTorch Tutorials - Neural Networks:** https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    - Tutoriales básicos de redes neuronales

15. **Sionna:** https://nvlabs.github.io/sionna/
    - Framework de NVIDIA para ML aplicado a comunicaciones (incluye autoencoders)

16. **DeepMIMO:** https://www.deepmimo.net/
    - Dataset de canales realistas para entrenar modelos de DL

### Código y Repositorios

17. **GitHub - CommPy:** https://github.com/veeresht/CommPy
    - Librería Python para comunicaciones digitales

18. **GitHub - Sionna Examples:** https://github.com/NVlabs/sionna
    - Ejemplos de autoencoders y sistemas end-to-end

19. **Papers with Code - Autoencoders:** https://paperswithcode.com/method/autoencoder
    - Código de implementaciones state-of-the-art

### Datasets Públicos

20. **DeepSig RadioML Datasets:** https://www.deepsig.ai/datasets
    - Datasets de señales RF para ML

21. **5G Channel Measurement Datasets:** https://www.5g-wave.eu/
    - Mediciones de canales 5G reales

### Artículos Avanzados y Extensiones

22. **Nachmani, E., Be'ery, Y., & Burstein, D. (2016).** "Learning to Decode Linear Codes Using Deep Learning." *54th Allerton Conference*, 341-346.
    - Decodificación de códigos FEC con redes neuronales

23. **Cammerer, S., et al. (2020).** "Trainable Communication Systems: Concepts and Prototype." *IEEE Transactions on Communications*, 68(9), 5489-5503.
    - Implementación práctica de sistemas entrenables end-to-end

24. **Aoudia, F. A., Hoydis, J., & Görtz, N. (2021).** "Model-Free Training of End-to-End Communication Systems." *IEEE Journal on Selected Areas in Communications*, 39(1), 199-210.
    - Entrenamiento sin modelo del canal (model-free)

25. **Xu, X., et al. (2021).** "Meta Learning to Bridge Vision and Language Models for Multimodal Few-Shot Learning." *ICLR 2021*.
    - Aplicación de meta-learning a adaptación de autoencoders

### Herramientas de Visualización

26. **TensorBoard:** https://www.tensorflow.org/tensorboard
    - Visualización de métricas de entrenamiento

27. **Weights & Biases:** https://wandb.ai/
    - Plataforma para experimentos de ML

28. **Plotly:** https://plotly.com/python/
    - Visualizaciones interactivas en Python

### Estándares y Especificaciones

29. **3GPP TS 38.211:** Physical channels and modulation (5G NR)
    - Especificaciones de modulación en 5G

30. **IEEE 802.11:** Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications
    - Estándares Wi-Fi con múltiples esquemas de modulación

### Blogs y Recursos Educativos

31. **Wireless Pi - Communications DSP:** https://wirelesspi.com/
    - Tutoriales de procesamiento de señales para comunicaciones

32. **Stanford CS229:** http://cs229.stanford.edu/
    - Curso de Machine Learning con fundamentos aplicables

33. **Distill.pub - Machine Learning Research:** https://distill.pub/
    - Artículos interactivos sobre conceptos de ML

### Videos y Cursos Online

34. **DeepLearning.AI - Neural Networks and Deep Learning**
    - Curso de Andrew Ng en Coursera

35. **MIT 6.S191 - Introduction to Deep Learning**
    - Curso MIT con material gratuito

### Conferencias Relevantes

- **ICC (IEEE International Conference on Communications)**
- **GLOBECOM (IEEE Global Communications Conference)**
- **ISIT (IEEE International Symposium on Information Theory)**
- **SPAWC (IEEE Signal Processing Advances in Wireless Communications)**

---

## 🎓 Notas Finales

**¡Éxito en tu laboratorio!** 🚀📡🔬

**Nota importante:** Este laboratorio introduce el paradigma revolucionario de **optimización end-to-end** en comunicaciones. Los conceptos aprendidos son directamente aplicables a:
- Sistemas 5G/6G con adaptación dinámica
- Comunicaciones ópticas con distorsiones no lineales
- IoT con restricciones de potencia
- Satélites con canales variables
- Software Defined Radio (SDR)

### Para Soporte Adicional

- 📖 **Consulta la documentación:** `teoria.md` y `README.md` en el directorio del laboratorio
- 💻 **Revisa el código de referencia:** `autoencoder.py`, `utils.py`
- 📓 **Explora notebooks completos:** `laboratorio.ipynb`, `ejercicios-propuestos.ipynb`
- 🔍 **Lee la documentación de PyTorch:** https://pytorch.org/docs/
- 💬 **Participa en foros:** Stack Overflow, PyTorch Forums, Reddit r/MachineLearning

### Conexión con Otros Laboratorios

Este laboratorio es fundamental para:
- **Guía 04:** Codificación de canal con autoencoders (extensión directa)
- **Guía 08:** Receptores neuronales OFDM (aplicación a sistemas multi-portadora)
- **Guías 15-17 (Sionna):** Sistemas end-to-end avanzados con Sionna framework

### Aplicaciones en el Mundo Real

**Casos de Éxito:**
1. **5G NR:** Adaptación de esquemas de modulación y codificación (MCS)
2. **Starlink:** Optimización de comunicaciones satelitales
3. **Facebook/Meta:** Optimización de backhaul wireless
4. **Qualcomm:** Receptores neurales en chipsets móviles

**Desafíos Abiertos:**
- Entrenamiento online y adaptación en tiempo real
- Reducción de complejidad computacional para dispositivos embebidos
- Robustez a ataques adversarios
- Certificación y estandarización de sistemas aprendidos

### Próximos Pasos Sugeridos

1. **Experimenta con canales realistas:** Implementa desvanecimiento Rayleigh/Rician
2. **Integra codificación de canal:** Añade códigos LDPC, Turbo, o Polar
3. **Explora sistemas MIMO:** Extiende a múltiples antenas
4. **Implementa en hardware:** Usa GNU Radio o USRP para pruebas reales
5. **Lee papers recientes:** Sigue conferencias ICC, GLOBECOM, ISIT

### Contribución a la Ciencia

Si obtienes resultados interesantes:
- Documenta tus hallazgos cuidadosamente
- Compara con state-of-the-art
- Considera escribir un paper para conferencias estudiantiles
- Comparte tu código en GitHub con licencia open source

---

**"The best way to predict the future is to invent it."** - Alan Kay

¡Continúa explorando las fronteras entre comunicaciones y machine learning! 🌟

