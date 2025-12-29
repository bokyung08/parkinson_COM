'''from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
#LSTM
def build_pose_model(input_shape):
    """Pose 기반 LSTM 모델 정의"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
# LSTM+Attention
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Attention, Concatenate

def build_pose_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x2 = Bidirectional(LSTM(64, return_sequences=True))(x)
    
    # Attention
    attn_out = Attention()([x2, x2])
    concat = Concatenate()([x2, attn_out])
    x = Dense(64, activation='relu')(concat[:, -1, :])
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, Flatten
# CNN-LSTM
def build_pose_model(input_shape):
    inputs = Input(shape=input_shape)  # (time, features)
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Transformer Encoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Dropout, Add

def transformer_block(x, num_heads, key_dim, ff_dim):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(x.shape[-1])(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)
    return x

def build_pose_model(input_shape):
    inputs = Input(shape=input_shape)
    x = transformer_block(inputs, num_heads=4, key_dim=64, ff_dim=128)
    x = transformer_block(x, num_heads=4, key_dim=64, ff_dim=128)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

    '''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, MultiHeadAttention, 
    GlobalAveragePooling1D, Dropout, Add, Layer, 
    TimeDistributed, Reshape
)


# ----------------------------------------------------------------------
# Spatio-Temporal Transformer 모델 
# ----------------------------------------------------------------------

# 1. Spatial-Transformer-Block
class SpatialTransformerBlock(Layer):
    """
    같은 프레임 내의 관절(노드) 간의 공간적 관계를 학습하는 커스텀 레이어
    """
    def __init__(self, num_heads, key_dim, ff_dim, dropout_rate=0.1, **kwargs):
        super(SpatialTransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # 레이어 초기화
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dropout_attn = Dropout(dropout_rate)
        self.add_attn = Add()
        self.norm_attn = LayerNormalization(epsilon=1e-6)
        
        self.ff_dense1 = Dense(ff_dim, activation="relu")
        self.dropout_ff = Dropout(dropout_rate)
        self.add_ff = Add()
        self.norm_ff = LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        # build 시점에 출력 차원을 입력 차원과 동일하게 맞추는 Dense 레이어 생성
        # input_shape: (Batch, Nodes, Features)
        self.embed_dim = input_shape[-1]
        self.ff_dense2 = Dense(self.embed_dim)
        super(SpatialTransformerBlock, self).build(input_shape)

    def call(self, inputs, training=False):
        # 1. Multi-Head Attention (공간)
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout_attn(attn_output, training=training)
        x = self.add_attn([inputs, attn_output])
        x = self.norm_attn(x)
        
        # 2. Feed-Forward Network
        ff_output = self.ff_dense1(x)
        ff_output = self.ff_dense2(ff_output)
        ff_output = self.dropout_ff(ff_output, training=training)
        x = self.add_ff([x, ff_output])
        x = self.norm_ff(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


# 2. Temporal-Transformer-Block
def temporal_transformer_block(x, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    """
    시간 축(프레임) 간의 관계를 학습하는 트랜스포머 블록
    x shape = (Batch, num_frames, hidden_dim)
    """
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dense(x.shape[-1])(ff_output) # 입력 차원과 동일하게
    ff_output = Dropout(dropout_rate)(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


# 3. train_model.py에서 호출할 최종 모델 빌더
def build_pose_model(input_shape, num_heads=4, key_dim=32, ff_dim=64, num_transformer_blocks=2):
    """
    Spatio-Temporal Transformer 모델 빌드
    input_shape = (Frames, Nodes, Features) e.g. (100, 33, 3)
    """
    # 입력 차원 (Frames, Nodes, Features)
    inputs = Input(shape=input_shape) 
    
    # 1. Spatial Attention
    # (Batch, Frames, Nodes, Features) -> (Batch, Frames, Nodes, EmbedDim)
    # 임베딩 차원을 MultiHeadAttention 헤드 수에 맞게 조정
    embed_dim = key_dim * num_heads 
    x = Dense(embed_dim)(inputs)
    spatial_block_instance = SpatialTransformerBlock(
        num_heads=num_heads, 
        key_dim=key_dim, 
        ff_dim=ff_dim
    )
    # TimeDistributed를 사용해 각 프레임(시간)별로 Spatial-Transformer를 독립 적용
    spatial_x = TimeDistributed(spatial_block_instance)(x)
    
    # 2. Temporal Attention
    # (Batch, Frames, Nodes, EmbedDim) -> (Batch, Frames, Nodes*EmbedDim)
    # 시간 축으로 어텐션을 적용하기 위해 노드와 특징을 펼침
    # (spatial_x.shape[2] = Nodes, spatial_x.shape[3] = EmbedDim)
    # Keras는 input_shape에서 Batch를 제외하므로 input_shape[0]=Frames, input_shape[1]=Nodes
    x_flat = Reshape((input_shape[0], input_shape[1] * embed_dim))(spatial_x)
    
    # 시간 축 트랜스포머 블록 적용
    temporal_x = x_flat
    for _ in range(num_transformer_blocks):
        temporal_x = temporal_transformer_block(
            temporal_x, num_heads, key_dim, ff_dim
        )
        
    # 3. Classification
    x = GlobalAveragePooling1D()(temporal_x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
