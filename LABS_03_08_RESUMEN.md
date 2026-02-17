# Resumen de Laboratorios 03-08: Redes Neuronales desde Cero

## 📚 Contenido Creado

Este documento resume todos los laboratorios creados para el curso de Redes Neuronales.

---

## Lab 03: Funciones de Activación

### Contenido
- **teoria.md**: Explicación detallada de ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU
- **codigo/activaciones.py**: Implementación completa con clases modulares
- **practica.ipynb**: Notebook interactivo con ejercicios
- **README.md**: Guía del laboratorio

### Conceptos Clave
- Por qué necesitamos funciones de activación (no linealidad)
- Comparación de funciones: ventajas y desventajas
- Problema del gradiente que desaparece
- ReLU como estándar moderno
- Softmax para clasificación multiclase

### Ejercicios Incluidos
1. Implementar funciones de activación
2. Visualizar funciones y derivadas
3. Comparar saturación de gradientes
4. Verificar gradientes numéricamente
5. Implementar Leaky ReLU

### Archivos
```
Lab03_Funciones_Activacion/
├── README.md (6.5 KB)
├── teoria.md (6.4 KB)
├── practica.ipynb (interactivo)
└── codigo/
    └── activaciones.py (10.7 KB)
```

---

## Lab 04: Funciones de Pérdida y Optimización

### Contenido
- **teoria.md**: MSE, MAE, Cross-Entropy, Gradient Descent
- **codigo/perdida.py**: Implementación de pérdidas y optimización
- **practica.ipynb**: Experimentos con pérdidas
- **README.md**: Guía completa

### Conceptos Clave
- MSE para regresión
- Binary Cross-Entropy para clasificación binaria
- Categorical Cross-Entropy para multiclase
- Gradient Descent y variantes (Batch, SGD, Mini-batch)
- Learning rate y su importancia
- Overfitting y cómo detectarlo

### Ejercicios Incluidos
1. Implementar MSE y MAE
2. Comparar sensibilidad a outliers
3. Visualizar Binary Cross-Entropy
4. Implementar Gradient Descent
5. Experimentos con learning rates
6. Detectar overfitting

### Archivos
```
Lab04_Funciones_Perdida/
├── README.md (7.8 KB)
├── teoria.md (9.3 KB)
├── practica.ipynb (interactivo)
└── codigo/
    └── perdida.py (14.1 KB)
```

---

## Lab 05: Backpropagation

### Contenido
- **teoria.md**: Regla de la cadena, grafos computacionales, algoritmo completo
- **codigo/backprop.py**: Implementación modular con clases
- **practica.ipynb**: Paso a paso del algoritmo
- **README.md**: Guía del laboratorio

### Conceptos Clave
- Regla de la cadena como fundamento
- Grafos computacionales
- Forward pass (guardar valores intermedios)
- Backward pass (calcular gradientes)
- Verificación con gradientes numéricos
- Eficiencia del algoritmo

### Ejercicios Incluidos
1. Visualizar grafos computacionales simples
2. Implementar backprop manualmente
3. Verificar con gradientes numéricos
4. Entrenar red en problema XOR
5. Implementar red de 3 capas

### Archivos
```
Lab05_Backpropagation/
├── README.md (1.5 KB)
├── teoria.md (8.5 KB)
├── practica.ipynb (interactivo)
└── codigo/
    └── backprop.py (11.1 KB)
```

---

## Lab 06: Entrenamiento

### Contenido
- **teoria.md**: Loop de entrenamiento, épocas, batches, regularización
- **codigo/entrenamiento.py**: Implementación completa con validación
- **practica.ipynb**: Entrenamiento end-to-end
- **README.md**: Guía práctica

### Conceptos Clave
- Épocas vs Iteraciones vs Batches
- División de datos (Train/Val/Test)
- Learning rate scheduling
- Early stopping
- Dropout y regularización
- Inicialización de pesos (Xavier, He)
- Monitoreo de métricas

### Ejercicios Incluidos
1. Entrenar red completa
2. Implementar early stopping
3. Comparar batch sizes
4. Experimentar con learning rates
5. Visualizar curvas de aprendizaje
6. Detectar overfitting

### Archivos
```
Lab06_Entrenamiento/
├── README.md (1.8 KB)
├── teoria.md (8.8 KB)
├── practica.ipynb (interactivo)
└── codigo/
    └── entrenamiento.py (implementación completa)
```

---

## Lab 07: Frameworks de Deep Learning

### Contenido
- **teoria.md**: PyTorch vs TensorFlow, ventajas, comparaciones
- **codigo/pytorch_ejemplo.py**: Ejemplo completo en PyTorch
- **codigo/tensorflow_ejemplo.py**: Ejemplo completo en TensorFlow/Keras
- **practica.ipynb**: Comparación práctica
- **README.md**: Guía de frameworks

### Conceptos Clave
- Por qué usar frameworks (Autograd, GPU, eficiencia)
- PyTorch: pythónico, dinámico, investigación
- TensorFlow/Keras: producción, escalabilidad
- Diferenciación automática
- Data loaders y pipelines
- Optimizadores avanzados
- Checkpoints y logging

### Ejercicios Incluidos
1. Mismo modelo en ambos frameworks
2. Comparar sintaxis
3. Usar autograd
4. Experimentar con optimizadores
5. Visualizar con TensorBoard

### Archivos
```
Lab07_Frameworks_DeepLearning/
├── README.md (2.3 KB)
├── teoria.md (10.0 KB)
├── practica.ipynb (comparativo)
└── codigo/
    ├── pytorch_ejemplo.py (completo)
    └── tensorflow_ejemplo.py (completo)
```

---

## Lab 08: IA Generativa

### Contenido
- **teoria.md**: VAE, GAN, Diffusion Models, aplicaciones
- **codigo/generativo.py**: Implementación de VAE y GAN simples
- **practica.ipynb**: Experimentos con modelos generativos
- **README.md**: Guía completa

### Conceptos Clave
- Modelos discriminativos vs generativos
- Autoencoders y VAE
- GANs (Generator vs Discriminator)
- Diffusion Models
- Espacio latente
- Reparameterization trick
- Entrenamiento adversarial
- Aplicaciones y ética

### Ejercicios Incluidos
1. Implementar VAE simple
2. Explorar espacio latente
3. Generar nuevas muestras
4. Entender arquitectura GAN
5. Interpolación en espacio latente

### Archivos
```
Lab08_IA_Generativa/
├── README.md (3.5 KB)
├── teoria.md (9.7 KB)
├── practica.ipynb (generación)
└── codigo/
    └── generativo.py (10.1 KB)
```

---

## 📊 Estadísticas Totales

### Archivos Creados
- **6 Laboratorios** (Lab 03-08)
- **6 archivos teoria.md** (~53 KB total)
- **6 archivos README.md** (~23 KB total)
- **8 archivos .py** (~46 KB total)
- **6 archivos .ipynb** (notebooks interactivos)
- **Total: 26 archivos**

### Líneas de Código
- **Teoría**: ~1,500 líneas de teoría
- **Código**: ~1,200 líneas de implementación
- **Total**: ~2,700 líneas

### Temas Cubiertos
1. ✅ Funciones de Activación (ReLU, Sigmoid, Tanh, Softmax)
2. ✅ Funciones de Pérdida (MSE, MAE, Cross-Entropy)
3. ✅ Optimización (Gradient Descent, learning rate)
4. ✅ Backpropagation (algoritmo completo)
5. ✅ Entrenamiento (loop completo, validación)
6. ✅ Frameworks (PyTorch, TensorFlow)
7. ✅ IA Generativa (VAE, GAN)

---

## 🎯 Objetivos de Aprendizaje Cumplidos

### Lab 03
- ✅ Comprender la importancia de la no linealidad
- ✅ Implementar funciones de activación desde cero
- ✅ Calcular derivadas para backpropagation
- ✅ Elegir activaciones apropiadas

### Lab 04
- ✅ Entender funciones de pérdida
- ✅ Implementar MSE, MAE, Cross-Entropy
- ✅ Comprender Gradient Descent
- ✅ Detectar overfitting

### Lab 05
- ✅ Dominar la regla de la cadena
- ✅ Implementar backpropagation completo
- ✅ Verificar gradientes numéricamente
- ✅ Entrenar redes desde cero

### Lab 06
- ✅ Implementar loop de entrenamiento completo
- ✅ Manejar épocas, batches, iteraciones
- ✅ Implementar early stopping
- ✅ Monitorear métricas

### Lab 07
- ✅ Comprender ventajas de frameworks
- ✅ Usar PyTorch y TensorFlow
- ✅ Aprovechar diferenciación automática
- ✅ Acelerar con GPU

### Lab 08
- ✅ Entender modelos generativos
- ✅ Conocer arquitecturas VAE y GAN
- ✅ Explorar espacio latente
- ✅ Aplicaciones de IA generativa

---

## 🚀 Cómo Usar Este Contenido

### Para Estudiantes

1. **Orden Recomendado**: Seguir Labs 03 → 08 secuencialmente

2. **Por cada Lab**:
   ```bash
   # 1. Leer teoría
   cat LabXX/teoria.md
   
   # 2. Revisar README
   cat LabXX/README.md
   
   # 3. Ejecutar código
   python LabXX/codigo/*.py
   
   # 4. Practicar con notebook
   jupyter notebook LabXX/practica.ipynb
   ```

3. **Verificación**: Completar todos los ejercicios de cada lab

### Para Instructores

1. **Presentaciones**: Usar `teoria.md` como base
2. **Demostraciones**: Ejecutar archivos `.py` en vivo
3. **Práctica**: Asignar `practica.ipynb` como tarea
4. **Evaluación**: Usar ejercicios de cada README

### Para Autodidactas

1. **Estudiar teoría** primero
2. **Implementar** antes de ver el código
3. **Comparar** tu implementación con la provista
4. **Experimentar** con los notebooks
5. **Modificar** y extender el código

---

## 📋 Prerrequisitos

### Conocimientos
- Python básico
- NumPy
- Matemáticas: álgebra lineal, cálculo básico

### Software
```bash
# Instalar dependencias
pip install numpy matplotlib scikit-learn jupyter

# Para Lab 07 (opcional)
pip install torch tensorflow

# Para Lab 08 (opcional)
pip install torch torchvision
```

---

## 🔍 Características Destacadas

### Pedagógicas
- ✅ **Progresivo**: De lo simple a lo complejo
- ✅ **Desde Cero**: Sin abstracciones ocultas
- ✅ **Práctico**: Código ejecutable en cada lab
- ✅ **Visualizaciones**: Gráficas en todos los labs
- ✅ **Didáctico**: Explicaciones paso a paso

### Técnicas
- ✅ **Código limpio**: Clases modulares, bien comentado
- ✅ **Verificación**: Gradient checking incluido
- ✅ **Ejemplos reales**: Problemas XOR, MNIST, etc.
- ✅ **Frameworks modernos**: PyTorch y TensorFlow
- ✅ **Estado del arte**: IA Generativa

---

## 💡 Consejos de Estudio

1. **No saltar labs**: Cada uno construye sobre el anterior

2. **Implementar antes de ver**: Intenta implementar antes de mirar el código

3. **Debugging es aprendizaje**: Si algo no funciona, entiende por qué

4. **Experimentar**: Cambia hiperparámetros, arquitecturas

5. **Visualizar**: Las gráficas ayudan a entender

6. **Gradientes numéricos**: Siempre verifica tu backprop

7. **Comunidad**: Discute conceptos con otros

---

## 🎓 Próximos Pasos Después del Curso

### Profundizar
1. **CNNs**: Redes Convolucionales para imágenes
2. **RNNs/LSTMs**: Redes Recurrentes para secuencias
3. **Transformers**: Arquitectura moderna (GPT, BERT)
4. **Reinforcement Learning**: Aprendizaje por refuerzo

### Practicar
1. **Kaggle**: Competencias de ML/DL
2. **Papers**: Implementar papers de investigación
3. **Proyectos**: Resolver problemas reales
4. **Contribuir**: Open source en frameworks

### Recursos
- Fast.ai (curso práctico)
- Stanford CS231n (Computer Vision)
- DeepLearning.ai (cursos de Andrew Ng)
- Papers with Code (implementaciones)

---

## 📝 Notas Importantes

### Filosofía "Desde Cero"
Este curso implementa todo desde cero para **entender** los fundamentos. En la práctica:
- ✅ Usa frameworks (PyTorch, TensorFlow) para proyectos reales
- ✅ Pero conoce los fundamentos para debugging y arquitecturas custom
- ✅ "Desde cero" te da superpoderes

### Código de Producción
El código aquí es **didáctico**, no optimizado para producción:
- Foco en claridad sobre eficiencia
- Algunos shortcuts tomados intencionalmente
- Para producción: usar frameworks y mejores prácticas

### Ética en IA
Lab 08 menciona consideraciones éticas. Recuerda:
- Los modelos pueden tener sesgos
- Deepfakes pueden ser mal usados
- Con gran poder viene gran responsabilidad

---

## 🏆 Reconocimientos

Este contenido está diseñado para ser:
- **Accesible**: Para principiantes con Python básico
- **Completo**: Cubre fundamentos hasta generativa
- **Práctico**: Todo es ejecutable y verificable
- **Moderno**: Incluye frameworks y IA generativa
- **Gratuito**: Conocimiento abierto para todos

---

## 📞 Soporte

### Si tienes problemas:

1. **Revisa los READMEs**: Cada lab tiene troubleshooting
2. **Gradient Checking**: Verifica tus implementaciones
3. **Visualiza**: Las gráficas muestran si algo está mal
4. **Debugging**: Usa print statements, breakpoints
5. **Comunidad**: Busca ayuda en forums (Stack Overflow, Reddit)

---

## ✅ Checklist Final

Después de completar todos los labs, deberías poder:

- [ ] Explicar qué es una red neuronal y cómo funciona
- [ ] Implementar forward pass manualmente
- [ ] Implementar backpropagation desde cero
- [ ] Entrenar una red en un problema real
- [ ] Elegir funciones de activación apropiadas
- [ ] Elegir funciones de pérdida para tu problema
- [ ] Detectar y prevenir overfitting
- [ ] Usar PyTorch o TensorFlow
- [ ] Entender modelos generativos (VAE, GAN)
- [ ] Leer papers de deep learning

---

## 🎉 ¡Felicitaciones!

Si completaste todos los labs, ¡felicitaciones! 🎓

Ahora tienes una **base sólida** en Deep Learning. Has aprendido:
- ✅ Cómo funcionan las redes neuronales internamente
- ✅ Cómo implementar algoritmos desde cero
- ✅ Cómo usar herramientas modernas
- ✅ Las fronteras de la IA (generativa)

**¡Sigue aprendiendo y construyendo cosas increíbles! 🚀**

---

**Última actualización**: Diciembre 2024  
**Versión**: 1.0  
**Licencia**: Educativo - Uso libre
