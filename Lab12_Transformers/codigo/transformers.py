"""
Implementación de Transformers y Self-Attention desde Cero
Laboratorio 11: Transformers
============================================

Este módulo implementa:
- Self-Attention mechanism desde cero con NumPy
- Multi-Head Attention
- Positional Encoding (sinusoidal y aprendido)
- Transformer Block completo
- Transformer Encoder-Decoder
- Ejemplos con Hugging Face (BERT, GPT-2)
- Utilidades para visualización de atención
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Union
import warnings

# Importar PyTorch de manera opcional
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Advertencia: PyTorch no está disponible.")

# Importar transformers (Hugging Face)
try:
    from transformers import (
        BertTokenizer, BertModel, BertForSequenceClassification,
        GPT2Tokenizer, GPT2LMHeadModel,
        AutoTokenizer, AutoModel,
        Trainer, TrainingArguments
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Advertencia: Transformers (Hugging Face) no está disponible.")


# =============================================================================
# PARTE 1: SELF-ATTENTION DESDE CERO (NUMPY)
# =============================================================================

class SelfAttentionNumPy:
    """
    Implementación de Self-Attention desde cero con NumPy.
    
    Self-Attention permite que cada elemento de una secuencia atienda
    a todos los demás elementos para determinar su representación.
    
    Fórmula:
        Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
    """
    
    def __init__(self, d_model: int, d_k: int = None, seed: int = 42):
        """
        Args:
            d_model: Dimensión del modelo (tamaño de embeddings)
            d_k: Dimensión de queries y keys (por defecto = d_model)
            seed: Semilla para reproducibilidad
        """
        np.random.seed(seed)
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = self.d_k  # Típicamente d_v = d_k
        
        # Matrices de proyección para Q, K, V
        # Inicialización Xavier/Glorot
        limit = np.sqrt(6 / (d_model + self.d_k))
        self.W_q = np.random.uniform(-limit, limit, (d_model, self.d_k))
        self.W_k = np.random.uniform(-limit, limit, (d_model, self.d_k))
        self.W_v = np.random.uniform(-limit, limit, (d_model, self.d_v))
    
    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None,
                return_attention: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass de self-attention.
        
        Args:
            X: Entrada (seq_len, d_model)
            mask: Máscara opcional (seq_len, seq_len). Valores True se enmascaran
            return_attention: Si True, retorna también los pesos de atención
            
        Returns:
            output: (seq_len, d_v)
            attention_weights (opcional): (seq_len, seq_len)
        """
        # 1. Proyectar a Q, K, V
        Q = X @ self.W_q  # (seq_len, d_k)
        K = X @ self.W_k  # (seq_len, d_k)
        V = X @ self.W_v  # (seq_len, d_v)
        
        # 2. Calcular scores de atención
        scores = Q @ K.T  # (seq_len, seq_len)
        
        # 3. Escalar por √d_k
        scores = scores / np.sqrt(self.d_k)
        
        # 4. Aplicar máscara (opcional)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        
        # 5. Aplicar softmax
        attention_weights = self.softmax(scores)  # (seq_len, seq_len)
        
        # 6. Aplicar atención a valores
        output = attention_weights @ V  # (seq_len, d_v)
        
        if return_attention:
            return output, attention_weights
        return output
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax numéricamente estable."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def __call__(self, X: np.ndarray, mask: Optional[np.ndarray] = None,
                 return_attention: bool = False):
        return self.forward(X, mask, return_attention)


class MultiHeadAttentionNumPy:
    """
    Multi-Head Attention: Múltiples cabezas de atención en paralelo.
    
    Cada cabeza aprende diferentes patrones de atención.
    Los resultados se concatenan y proyectan.
    
    Fórmula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        donde head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, seed: int = 42):
        """
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            seed: Semilla para reproducibilidad
        """
        np.random.seed(seed)
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimensión por cabeza
        
        # Crear cabezas de atención
        self.heads = [
            SelfAttentionNumPy(d_model, self.d_k, seed=seed+i)
            for i in range(num_heads)
        ]
        
        # Proyección de salida
        limit = np.sqrt(6 / (d_model + d_model))
        self.W_o = np.random.uniform(-limit, limit, (d_model, d_model))
    
    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None,
                return_attention: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Forward pass de multi-head attention.
        
        Args:
            X: Entrada (seq_len, d_model)
            mask: Máscara opcional (seq_len, seq_len)
            return_attention: Si True, retorna también pesos de atención
            
        Returns:
            output: (seq_len, d_model)
            attention_weights (opcional): Lista de (seq_len, seq_len) por cabeza
        """
        # Aplicar cada cabeza en paralelo
        head_outputs = []
        attention_weights_list = []
        
        for head in self.heads:
            if return_attention:
                head_out, attn_weights = head(X, mask, return_attention=True)
                head_outputs.append(head_out)
                attention_weights_list.append(attn_weights)
            else:
                head_out = head(X, mask, return_attention=False)
                head_outputs.append(head_out)
        
        # Concatenar salidas de todas las cabezas
        # (seq_len, num_heads * d_k) = (seq_len, d_model)
        concat_output = np.concatenate(head_outputs, axis=-1)
        
        # Proyección final
        output = concat_output @ self.W_o  # (seq_len, d_model)
        
        if return_attention:
            return output, attention_weights_list
        return output
    
    def __call__(self, X: np.ndarray, mask: Optional[np.ndarray] = None,
                 return_attention: bool = False):
        return self.forward(X, mask, return_attention)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Crea una máscara causal (para decoders).
    
    Previene que las posiciones atiendan a posiciones futuras.
    
    Args:
        seq_len: Longitud de la secuencia
        
    Returns:
        mask: (seq_len, seq_len) con True en posiciones futuras
    """
    # Matriz triangular superior (sin diagonal)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return mask


# =============================================================================
# PARTE 2: POSITIONAL ENCODING
# =============================================================================

class PositionalEncodingSinusoidal:
    """
    Positional Encoding Sinusoidal (Vaswani et al., 2017).
    
    Añade información de posición usando funciones seno y coseno
    de diferentes frecuencias.
    
    Fórmula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Dimensión del modelo
            max_len: Longitud máxima de secuencia
        """
        self.d_model = d_model
        self.max_len = max_len
        
        # Pre-calcular encoding para todas las posiciones
        self.encoding = self._create_encoding()
    
    def _create_encoding(self) -> np.ndarray:
        """Crea la matriz de positional encoding."""
        # Posiciones: (max_len, 1)
        position = np.arange(self.max_len)[:, np.newaxis]
        
        # Dimensiones: (1, d_model // 2)
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
        )
        
        # Inicializar encoding
        pe = np.zeros((self.max_len, self.d_model))
        
        # Aplicar seno a dimensiones pares
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Aplicar coseno a dimensiones impares
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def get_encoding(self, seq_len: int) -> np.ndarray:
        """
        Obtiene encoding para una secuencia de longitud dada.
        
        Args:
            seq_len: Longitud de la secuencia
            
        Returns:
            encoding: (seq_len, d_model)
        """
        return self.encoding[:seq_len]
    
    def __call__(self, seq_len: int) -> np.ndarray:
        return self.get_encoding(seq_len)


def visualize_positional_encoding(d_model: int = 128, max_len: int = 100):
    """
    Visualiza el positional encoding sinusoidal.
    
    Args:
        d_model: Dimensión del modelo
        max_len: Longitud máxima de secuencia
    """
    pe = PositionalEncodingSinusoidal(d_model, max_len)
    encoding = pe.get_encoding(max_len)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Encoding completo
    plt.subplot(1, 2, 1)
    plt.imshow(encoding.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Valor de encoding')
    plt.xlabel('Posición en secuencia')
    plt.ylabel('Dimensión de embedding')
    plt.title('Positional Encoding Sinusoidal')
    
    # Plot 2: Algunas dimensiones específicas
    plt.subplot(1, 2, 2)
    positions = np.arange(max_len)
    plt.plot(positions, encoding[:, 4], label='dim 4 (sin)')
    plt.plot(positions, encoding[:, 5], label='dim 5 (cos)')
    plt.plot(positions, encoding[:, 32], label='dim 32 (sin)')
    plt.plot(positions, encoding[:, 33], label='dim 33 (cos)')
    plt.xlabel('Posición')
    plt.ylabel('Valor')
    plt.title('Encoding por dimensión')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()


# =============================================================================
# PARTE 3: TRANSFORMER BLOCK (PYTORCH)
# =============================================================================

if PYTORCH_AVAILABLE:
    
    class PositionwiseFeedForward(nn.Module):
        """
        Feed-Forward Network aplicado a cada posición independientemente.
        
        FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
             = ReLU(Linear(x)) @ Linear
        """
        
        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
            """
            Args:
                d_model: Dimensión del modelo
                d_ff: Dimensión de la capa oculta (típicamente 4 * d_model)
                dropout: Tasa de dropout
            """
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, d_model)
            Returns:
                output: (batch, seq_len, d_model)
            """
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return x
    
    
    class TransformerEncoderBlock(nn.Module):
        """
        Bloque de Transformer Encoder.
        
        Componentes:
        1. Multi-Head Self-Attention
        2. Add & Norm
        3. Position-wise Feed-Forward
        4. Add & Norm
        """
        
        def __init__(self, d_model: int, num_heads: int, d_ff: int,
                     dropout: float = 0.1):
            """
            Args:
                d_model: Dimensión del modelo
                num_heads: Número de cabezas de atención
                d_ff: Dimensión de feed-forward network
                dropout: Tasa de dropout
            """
            super().__init__()
            
            # Multi-head self-attention
            self.self_attention = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
            
            # Feed-forward network
            self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
            
            # Layer normalization
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            # Dropout
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        
        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, d_model)
                mask: Máscara de atención opcional
            Returns:
                output: (batch, seq_len, d_model)
            """
            # Self-attention con residual connection y layer norm
            attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
            x = x + self.dropout1(attn_output)
            x = self.norm1(x)
            
            # Feed-forward con residual connection y layer norm
            ff_output = self.feed_forward(x)
            x = x + self.dropout2(ff_output)
            x = self.norm2(x)
            
            return x
    
    
    class TransformerDecoderBlock(nn.Module):
        """
        Bloque de Transformer Decoder.
        
        Componentes:
        1. Masked Multi-Head Self-Attention
        2. Add & Norm
        3. Multi-Head Cross-Attention (con encoder)
        4. Add & Norm
        5. Position-wise Feed-Forward
        6. Add & Norm
        """
        
        def __init__(self, d_model: int, num_heads: int, d_ff: int,
                     dropout: float = 0.1):
            """
            Args:
                d_model: Dimensión del modelo
                num_heads: Número de cabezas de atención
                d_ff: Dimensión de feed-forward network
                dropout: Tasa de dropout
            """
            super().__init__()
            
            # Masked self-attention
            self.self_attention = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
            
            # Cross-attention (encoder-decoder attention)
            self.cross_attention = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
            
            # Feed-forward network
            self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
            
            # Layer normalization
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            
            # Dropout
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
        
        def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                    src_mask: Optional[torch.Tensor] = None,
                    tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Args:
                x: Entrada del decoder (batch, tgt_len, d_model)
                encoder_output: Salida del encoder (batch, src_len, d_model)
                src_mask: Máscara para encoder (padding)
                tgt_mask: Máscara causal para decoder
            Returns:
                output: (batch, tgt_len, d_model)
            """
            # Masked self-attention
            attn_output, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
            x = x + self.dropout1(attn_output)
            x = self.norm1(x)
            
            # Cross-attention con encoder
            attn_output, _ = self.cross_attention(
                x, encoder_output, encoder_output, attn_mask=src_mask
            )
            x = x + self.dropout2(attn_output)
            x = self.norm2(x)
            
            # Feed-forward
            ff_output = self.feed_forward(x)
            x = x + self.dropout3(ff_output)
            x = self.norm3(x)
            
            return x
    
    
    class PositionalEncodingPyTorch(nn.Module):
        """Positional Encoding en PyTorch."""
        
        def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Crear positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            # Registrar como buffer (no es parámetro entrenable)
            self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, d_model)
            Returns:
                output: (batch, seq_len, d_model)
            """
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)
    
    
    class TransformerModel(nn.Module):
        """
        Transformer completo (Encoder-Decoder).
        
        Arquitectura del paper "Attention is All You Need".
        """
        
        def __init__(self, vocab_size: int, d_model: int = 512,
                     num_heads: int = 8, num_layers: int = 6,
                     d_ff: int = 2048, max_len: int = 5000,
                     dropout: float = 0.1):
            """
            Args:
                vocab_size: Tamaño del vocabulario
                d_model: Dimensión del modelo
                num_heads: Número de cabezas de atención
                num_layers: Número de capas encoder/decoder
                d_ff: Dimensión de feed-forward network
                max_len: Longitud máxima de secuencia
                dropout: Tasa de dropout
            """
            super().__init__()
            
            self.d_model = d_model
            
            # Embeddings
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = PositionalEncodingPyTorch(d_model, max_len, dropout)
            
            # Encoder
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            
            # Decoder
            self.decoder_layers = nn.ModuleList([
                TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
            
            # Output projection
            self.output_projection = nn.Linear(d_model, vocab_size)
            
            # Inicialización
            self._init_weights()
        
        def _init_weights(self):
            """Inicialización de pesos (Xavier)."""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Encoder forward pass.
            
            Args:
                src: (batch, src_len) índices de tokens
                src_mask: Máscara opcional
            Returns:
                encoder_output: (batch, src_len, d_model)
            """
            # Embedding + positional encoding
            x = self.embedding(src) * np.sqrt(self.d_model)
            x = self.pos_encoding(x)
            
            # Aplicar capas de encoder
            for layer in self.encoder_layers:
                x = layer(x, src_mask)
            
            return x
        
        def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
                   src_mask: Optional[torch.Tensor] = None,
                   tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Decoder forward pass.
            
            Args:
                tgt: (batch, tgt_len) índices de tokens
                encoder_output: (batch, src_len, d_model)
                src_mask: Máscara para encoder
                tgt_mask: Máscara causal para decoder
            Returns:
                decoder_output: (batch, tgt_len, d_model)
            """
            # Embedding + positional encoding
            x = self.embedding(tgt) * np.sqrt(self.d_model)
            x = self.pos_encoding(x)
            
            # Aplicar capas de decoder
            for layer in self.decoder_layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)
            
            return x
        
        def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                    src_mask: Optional[torch.Tensor] = None,
                    tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Forward pass completo.
            
            Args:
                src: (batch, src_len) secuencia de entrada
                tgt: (batch, tgt_len) secuencia objetivo
                src_mask: Máscara para encoder
                tgt_mask: Máscara causal para decoder
            Returns:
                logits: (batch, tgt_len, vocab_size)
            """
            # Encoder
            encoder_output = self.encode(src, src_mask)
            
            # Decoder
            decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
            
            # Proyección a vocabulario
            logits = self.output_projection(decoder_output)
            
            return logits
        
        @staticmethod
        def generate_square_subsequent_mask(sz: int, device: str = 'cpu') -> torch.Tensor:
            """
            Genera máscara causal para decoder.
            
            Args:
                sz: Tamaño de la secuencia
                device: Dispositivo (cpu/cuda)
            Returns:
                mask: (sz, sz) con -inf en triángulo superior
            """
            mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
            return mask


# =============================================================================
# PARTE 4: EJEMPLOS CON HUGGING FACE TRANSFORMERS
# =============================================================================

if TRANSFORMERS_AVAILABLE and PYTORCH_AVAILABLE:
    
    class BERTSentimentClassifier:
        """
        Clasificador de sentimientos usando BERT pre-entrenado.
        
        Fine-tuning de BERT para análisis de sentimiento.
        """
        
        def __init__(self, model_name: str = 'bert-base-uncased',
                     num_labels: int = 2, device: str = None):
            """
            Args:
                model_name: Nombre del modelo BERT pre-entrenado
                num_labels: Número de clases (2 para binario: pos/neg)
                device: Dispositivo (cpu/cuda)
            """
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Cargar tokenizer y modelo
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            ).to(self.device)
        
        def preprocess(self, texts: List[str], max_length: int = 128) -> Dict:
            """
            Preprocesa textos para BERT.
            
            Args:
                texts: Lista de textos
                max_length: Longitud máxima de secuencia
            Returns:
                encoding: Diccionario con input_ids, attention_mask, etc.
            """
            encoding = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            return {k: v.to(self.device) for k, v in encoding.items()}
        
        def train_step(self, texts: List[str], labels: List[int],
                       optimizer: torch.optim.Optimizer) -> float:
            """
            Un paso de entrenamiento.
            
            Args:
                texts: Lista de textos
                labels: Lista de etiquetas
                optimizer: Optimizador
            Returns:
                loss: Pérdida del batch
            """
            self.model.train()
            
            # Preprocesar
            inputs = self.preprocess(texts)
            labels_tensor = torch.tensor(labels, device=self.device)
            
            # Forward pass
            outputs = self.model(**inputs, labels=labels_tensor)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            return loss.item()
        
        def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
            """
            Predice sentimiento de textos.
            
            Args:
                texts: Lista de textos
            Returns:
                predictions: Lista de predicciones (0 o 1)
                probabilities: Lista de probabilidades
            """
            self.model.eval()
            
            # Preprocesar
            inputs = self.preprocess(texts)
            
            # Predecir
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
            
            return predictions.cpu().tolist(), probs.cpu().tolist()
        
        def get_attention_weights(self, text: str, layer: int = -1) -> np.ndarray:
            """
            Extrae pesos de atención de BERT.
            
            Args:
                text: Texto de entrada
                layer: Capa de la que extraer atención (-1 = última)
            Returns:
                attention: (num_heads, seq_len, seq_len)
            """
            self.model.eval()
            
            # Preprocesar
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            
            # Forward con atención
            with torch.no_grad():
                outputs = self.model.bert(**inputs, output_attentions=True)
                attentions = outputs.attentions  # Tuple de (batch, heads, seq, seq)
            
            # Extraer capa específica
            attention = attentions[layer].squeeze(0).cpu().numpy()  # (heads, seq, seq)
            
            return attention
    
    
    class GPT2TextGenerator:
        """
        Generador de texto usando GPT-2.
        
        Uso de GPT-2 pre-entrenado para generación de texto.
        """
        
        def __init__(self, model_name: str = 'gpt2', device: str = None):
            """
            Args:
                model_name: Nombre del modelo GPT-2
                device: Dispositivo (cpu/cuda)
            """
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Cargar tokenizer y modelo
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            
            # Configurar pad token (GPT-2 no tiene por defecto)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def generate(self, prompt: str, max_length: int = 100,
                     temperature: float = 1.0, top_k: int = 50,
                     top_p: float = 0.95, num_return_sequences: int = 1) -> List[str]:
            """
            Genera texto dado un prompt.
            
            Args:
                prompt: Texto inicial
                max_length: Longitud máxima de generación
                temperature: Temperatura (> 1 más aleatorio, < 1 más determinista)
                top_k: Top-k sampling
                top_p: Nucleus sampling (top-p)
                num_return_sequences: Número de secuencias a generar
            Returns:
                generated_texts: Lista de textos generados
            """
            self.model.eval()
            
            # Codificar prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generar
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decodificar
            generated_texts = [
                self.tokenizer.decode(seq, skip_special_tokens=True)
                for seq in output_sequences
            ]
            
            return generated_texts
        
        def get_next_token_probabilities(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
            """
            Obtiene probabilidades del siguiente token.
            
            Args:
                text: Texto de entrada
                top_k: Top-k tokens a retornar
            Returns:
                token_probs: Lista de (token, probabilidad)
            """
            self.model.eval()
            
            # Codificar
            input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]  # (1, vocab_size)
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Top-k tokens
            top_probs, top_indices = torch.topk(next_token_probs[0], top_k)
            
            # Decodificar tokens
            token_probs = [
                (self.tokenizer.decode(idx), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
            
            return token_probs


# =============================================================================
# PARTE 5: UTILIDADES PARA VISUALIZACIÓN
# =============================================================================

def visualize_attention_weights(attention_weights: np.ndarray,
                                 tokens: List[str],
                                 title: str = "Attention Weights"):
    """
    Visualiza matriz de pesos de atención.
    
    Args:
        attention_weights: (seq_len, seq_len) matriz de atención
        tokens: Lista de tokens
        title: Título del plot
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Peso de atención')
    
    # Configurar ejes
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel('Key (tokens atendidos)')
    plt.ylabel('Query (tokens que atienden)')
    plt.title(title)
    
    # Añadir valores en celdas
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text = plt.text(j, i, f'{attention_weights[i, j]:.2f}',
                           ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    return plt.gcf()


def visualize_multi_head_attention(attention_weights_list: List[np.ndarray],
                                   tokens: List[str],
                                   num_heads_to_show: int = 4):
    """
    Visualiza múltiples cabezas de atención.
    
    Args:
        attention_weights_list: Lista de matrices (seq_len, seq_len)
        tokens: Lista de tokens
        num_heads_to_show: Número de cabezas a mostrar
    """
    num_heads = min(len(attention_weights_list), num_heads_to_show)
    fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 5))
    
    if num_heads == 1:
        axes = [axes]
    
    for idx, (ax, attn) in enumerate(zip(axes, attention_weights_list[:num_heads])):
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        ax.set_title(f'Head {idx+1}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        plt.colorbar(im, ax=ax, label='Atención')
    
    plt.tight_layout()
    return fig


def compare_attention_patterns(text: str, model_name: str = 'bert-base-uncased',
                               layers_to_show: List[int] = [0, 5, 11]):
    """
    Compara patrones de atención en diferentes capas de BERT.
    
    Args:
        text: Texto de entrada
        model_name: Modelo BERT a usar
        layers_to_show: Capas a visualizar
    """
    if not (TRANSFORMERS_AVAILABLE and PYTORCH_AVAILABLE):
        print("Se requiere transformers y PyTorch")
        return
    
    # Cargar modelo
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    
    # Tokenizar
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions  # Tuple de tensores
    
    # Visualizar capas especificadas
    fig, axes = plt.subplots(1, len(layers_to_show), figsize=(6*len(layers_to_show), 5))
    
    if len(layers_to_show) == 1:
        axes = [axes]
    
    for ax, layer_idx in zip(axes, layers_to_show):
        # Promediar sobre todas las cabezas
        layer_attn = attentions[layer_idx].squeeze(0).mean(dim=0).numpy()
        
        im = ax.imshow(layer_attn, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        plt.colorbar(im, ax=ax, label='Atención promedio')
    
    plt.tight_layout()
    return fig


# =============================================================================
# FUNCIONES DE DEMOSTRACIÓN
# =============================================================================

def demo_self_attention():
    """Demostración de self-attention básico."""
    print("=" * 60)
    print("DEMO: Self-Attention desde cero")
    print("=" * 60)
    
    # Crear datos de ejemplo
    seq_len, d_model = 5, 8
    X = np.random.randn(seq_len, d_model)
    
    print(f"\nEntrada: {X.shape}")
    print("Palabras: ['El', 'gato', 'bebió', 'la', 'leche']")
    
    # Crear self-attention
    attention = SelfAttentionNumPy(d_model, d_k=8)
    
    # Forward pass
    output, attn_weights = attention(X, return_attention=True)
    
    print(f"\nSalida: {output.shape}")
    print(f"Pesos de atención: {attn_weights.shape}")
    print("\nMatriz de atención:")
    print(attn_weights.round(3))
    
    # Visualizar
    tokens = ['El', 'gato', 'bebió', 'la', 'leche']
    visualize_attention_weights(attn_weights, tokens, "Self-Attention Demo")
    plt.savefig('attention_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualización guardada en 'attention_demo.png'")


def demo_multi_head_attention():
    """Demostración de multi-head attention."""
    print("\n" + "=" * 60)
    print("DEMO: Multi-Head Attention")
    print("=" * 60)
    
    # Crear datos
    seq_len, d_model, num_heads = 5, 64, 4
    X = np.random.randn(seq_len, d_model)
    
    print(f"\nEntrada: {X.shape}")
    print(f"Número de cabezas: {num_heads}")
    print(f"Dimensión por cabeza: {d_model // num_heads}")
    
    # Crear multi-head attention
    mha = MultiHeadAttentionNumPy(d_model, num_heads)
    
    # Forward pass
    output, attn_list = mha(X, return_attention=True)
    
    print(f"\nSalida: {output.shape}")
    print(f"Atención por cabeza: {len(attn_list)} x {attn_list[0].shape}")
    
    # Visualizar
    tokens = ['El', 'gato', 'bebió', 'la', 'leche']
    visualize_multi_head_attention(attn_list, tokens, num_heads_to_show=4)
    plt.savefig('multi_head_attention_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualización guardada en 'multi_head_attention_demo.png'")


def demo_positional_encoding():
    """Demostración de positional encoding."""
    print("\n" + "=" * 60)
    print("DEMO: Positional Encoding")
    print("=" * 60)
    
    # Crear positional encoding
    d_model, max_len = 128, 100
    pe = PositionalEncodingSinusoidal(d_model, max_len)
    
    print(f"\nDimensión: {d_model}")
    print(f"Longitud máxima: {max_len}")
    
    encoding = pe.get_encoding(max_len)
    print(f"Shape del encoding: {encoding.shape}")
    
    # Visualizar
    visualize_positional_encoding(d_model, max_len)
    plt.savefig('positional_encoding_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualización guardada en 'positional_encoding_demo.png'")


if __name__ == "__main__":
    print("Módulo de Transformers cargado correctamente")
    print(f"PyTorch disponible: {PYTORCH_AVAILABLE}")
    print(f"Transformers (Hugging Face) disponible: {TRANSFORMERS_AVAILABLE}")
    
    # Ejecutar demos
    demo_self_attention()
    demo_multi_head_attention()
    demo_positional_encoding()
    
    print("\n" + "=" * 60)
    print("¡Todas las demos completadas!")
    print("=" * 60)
