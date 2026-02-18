"""
Ejemplo de Red Neuronal en TensorFlow/Keras
Lab 07: Frameworks de Deep Learning
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def crear_modelo_keras(input_size, hidden_sizes, output_size):
    """Crear modelo en Keras."""
    
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_size,)))
    
    for hidden_size in hidden_sizes:
        model.add(keras.layers.Dense(hidden_size, activation='relu'))
    
    model.add(keras.layers.Dense(output_size, activation='softmax'))
    
    return model


def entrenar_tensorflow():
    """Ejemplo completo de entrenamiento en TensorFlow/Keras."""
    
    print("=" * 70)
    print("EJEMPLO DE TENSORFLOW/KERAS")
    print("=" * 70)
    
    # 1. Preparar datos
    print("\n1. Preparando datos...")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test: {len(X_test)} muestras")
    
    # 2. Crear modelo
    print("\n2. Creando modelo...")
    model = crear_modelo_keras(input_size=2, hidden_sizes=[16, 8], output_size=2)
    
    print("   Arquitectura:")
    model.summary()
    
    # 3. Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Entrenar
    print("\n3. Entrenando...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Imprimir cada 20 épocas
    for epoch in range(0, 100, 20):
        print(f"   Época {epoch+1}/100 - "
              f"Loss: {history.history['loss'][epoch]:.4f} - "
              f"Acc: {history.history['accuracy'][epoch]:.4f} - "
              f"Val Acc: {history.history['val_accuracy'][epoch]:.4f}")
    
    # 5. Resultados finales
    print(f"\n4. Resultados finales:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # 6. Visualizar
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Test', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Curva de Aprendizaje - TensorFlow/Keras')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Test', linewidth=2, color='green')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.title('Accuracy - TensorFlow/Keras')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tensorflow_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("✓ Entrenamiento en TensorFlow/Keras completado!")
    print("=" * 70)


if __name__ == "__main__":
    entrenar_tensorflow()
