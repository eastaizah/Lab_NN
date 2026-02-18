# Lab 07: Frameworks de Deep Learning

## Objetivos
1. Comprender ventajas de frameworks
2. Implementar redes en PyTorch
3. Implementar redes en TensorFlow/Keras
4. Comparar ambos frameworks
5. Usar diferenciación automática

## Estructura
```
Lab07_Frameworks_DeepLearning/
├── README.md
├── teoria.md
├── practica.ipynb
└── codigo/
    ├── pytorch_ejemplo.py
    └── tensorflow_ejemplo.py
```

## ¿Por qué Frameworks?

- **Autograd**: Backpropagation automático
- **GPU**: Aceleración automática
- **Optimizaciones**: 10-100x más rápido
- **Ecosistema**: Modelos pre-entrenados, herramientas

## PyTorch vs TensorFlow

| Característica | PyTorch | TensorFlow/Keras |
|----------------|---------|------------------|
| **Filosofía** | Pythónico | Producción |
| **Debugging** | Más fácil | Más difícil |
| **Comunidad** | Investigación | Industria |
| **API** | Imperativo | Keras (alto nivel) |

## Práctica

### PyTorch:
```bash
python codigo/pytorch_ejemplo.py
```

### TensorFlow:
```bash
python codigo/tensorflow_ejemplo.py
```

### Notebook:
```bash
jupyter notebook practica.ipynb
```

## Conceptos Clave

### Tensores
```python
# PyTorch
x = torch.tensor([1, 2, 3])

# TensorFlow
x = tf.constant([1, 2, 3])
```

### Autograd
```python
# PyTorch
x.requires_grad = True
y = x ** 2
y.backward()
print(x.grad)

# TensorFlow
with tf.GradientTape() as tape:
    y = x ** 2
dy_dx = tape.gradient(y, x)
```

### Modelos
```python
# PyTorch
class MiModelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

# TensorFlow/Keras
model = keras.Sequential([
    keras.layers.Dense(2, input_shape=(10,))
])
```

## Ejercicios

1. Implementar mismo modelo en ambos frameworks
2. Comparar tiempos de entrenamiento
3. Usar GPU (si disponible)
4. Experimentar con optimizadores

## Instalación

### PyTorch:
```bash
pip install torch torchvision
```

### TensorFlow:
```bash
pip install tensorflow
```

## Verificación
- [ ] Entiendo ventajas de frameworks
- [ ] Puedo usar PyTorch
- [ ] Puedo usar TensorFlow/Keras
- [ ] Entiendo autograd

## Próximo Lab
**Lab 09**: IA Generativa (VAEs, GANs)

---
**¡Los frameworks hacen el deep learning accesible! 🚀**
