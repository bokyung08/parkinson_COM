from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Dropout, Add, Layer,
    TimeDistributed, Reshape
)


# ----------------------------------------------------------------------
# Spatio-Temporal Transformer blocks
# ----------------------------------------------------------------------

class SpatialTransformerBlock(Layer):
    """
    Spatial block that learns relationships between joints in the same frame.
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
        return input_shape


def temporal_transformer_block(x, num_heads, key_dim, ff_dim, dropout_rate=0.1):
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
    Spatio-Temporal Transformer based model (regression head)
    """
    inputs = Input(shape=input_shape)

    embed_dim = key_dim * num_heads
    x = Dense(embed_dim)(inputs)
    spatial_block_instance = SpatialTransformerBlock(
        num_heads=num_heads,
        key_dim=key_dim,
        ff_dim=ff_dim
    )
    spatial_x = TimeDistributed(spatial_block_instance)(x)

    x_flat = Reshape((input_shape[0], input_shape[1] * embed_dim))(spatial_x)
    temporal_x = x_flat
    for _ in range(num_transformer_blocks):
        temporal_x = temporal_transformer_block(
            temporal_x, num_heads, key_dim, ff_dim
        )

    x = GlobalAveragePooling1D()(temporal_x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)

    reg_output = Dense(1, activation='linear', name='reg')(x)

    model = Model(inputs, reg_output)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model
