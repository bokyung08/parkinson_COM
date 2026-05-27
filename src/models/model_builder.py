from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Dropout, Add, Layer,
    TimeDistributed, Reshape
)


# ----------------------------------------------------------------------
# Spatio-Temporal Transformer 블록들
# ----------------------------------------------------------------------

class SpatialTransformerBlock(Layer):
    """
    동일 프레임 내 관절들 사이의 공간적 관계를 학습하는 블록
    """
    def __init__(self, num_heads, key_dim, ff_dim, dropout_rate=0.1, **kwargs):
        super(SpatialTransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dropout_attn = Dropout(dropout_rate)
        self.add_attn = Add()
        self.norm_attn = LayerNormalization(epsilon=1e-6)

        self.ff_dense1 = Dense(ff_dim, activation="relu")
        self.dropout_ff = Dropout(dropout_rate)
        self.add_ff = Add()
        self.norm_ff = LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.embed_dim = input_shape[-1]
        self.ff_dense2 = Dense(self.embed_dim)
        super(SpatialTransformerBlock, self).build(input_shape)

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout_attn(attn_output, training=training)
        x = self.add_attn([inputs, attn_output])
        x = self.norm_attn(x)

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

    def compute_output_shape(self, input_shape):
        # 입력과 동일한 shape을 유지하는 블록이므로 그대로 반환
        return input_shape


def temporal_transformer_block(x, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    """
    시계열 상의 프레임 간 관계를 학습하는 블록
    x shape = (Batch, num_frames, hidden_dim)
    """
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dense(x.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def build_pose_model(input_shape, num_heads=4, key_dim=32, ff_dim=64, num_transformer_blocks=2, optimizer='adam'):
    """
    Spatio-Temporal Transformer 기반 멀티태스크 모델
    - 입력: (Frames, Nodes, Features)
    - 출력:
        cls : HY stage 기반 이진 분류 (중등도 이상 여부)
        reg : MDS-UPDRS Part III 총점 회귀
    """
    inputs = Input(shape=input_shape)

    # Spatial Attention (프레임별로 관절 간 관계)
    embed_dim = key_dim * num_heads
    x = Dense(embed_dim)(inputs)
    spatial_block_instance = SpatialTransformerBlock(
        num_heads=num_heads,
        key_dim=key_dim,
        ff_dim=ff_dim
    )
    spatial_x = TimeDistributed(spatial_block_instance)(x)

    # Temporal Attention (프레임 간 관계)
    x_flat = Reshape((input_shape[0], input_shape[1] * embed_dim))(spatial_x)
    temporal_x = x_flat
    for _ in range(num_transformer_blocks):
        temporal_x = temporal_transformer_block(
            temporal_x, num_heads, key_dim, ff_dim
        )

    # 공유 임베딩
    x = GlobalAveragePooling1D()(temporal_x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)

    # 단일 회귀 헤드 (UPDRS 예측)
    reg_output = Dense(1, activation='linear', name='reg')(x)

    model = Model(inputs, reg_output)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model
