import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization
from keras.optimizers import Adam

np.random.seed(42)

# Generate synthetic sales data
days = np.arange(365)
baseline = 150
trend = 0.05 * days
seasonal = 50 * np.sin(2 * np.pi * days / 365)
noise = np.random.normal(1, 2, size=365)

for _ in range(5):
    random_indices = np.random.choice(365, size=30, replace=False)
    noise[random_indices] += np.random.uniform(0, 12, size=30)

sales_data = baseline + trend + seasonal + noise

# Define date list
start_date = datetime(2023, 1, 1)
date_list = [start_date + timedelta(days=i) for i in range(365)]

# Normalize data
scaler = MinMaxScaler()
sales_data_normalized = scaler.fit_transform(sales_data.reshape(-1, 1))

# Split data into train and test sets
X_train, X_test = train_test_split(sales_data_normalized, test_size=0.2, random_state=42)
from keras.layers import Dense, LayerNormalization, Dropout, Input, Add, MultiHeadAttention, Reshape
from keras.models import Model

def transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs

    # Transformer Encoder
    encoder_x = Dense(64, activation='relu')(x)
    encoder_x = LayerNormalization()(encoder_x)
    encoder_x = Dropout(0.1)(encoder_x)

    # Post-Encode layer
    post_encode_x = Dense(64, activation='relu')(encoder_x)
    post_encode_x = LayerNormalization()(post_encode_x)
    post_encode_x = Dropout(0.1)(post_encode_x)

    # Transformer Decoder
    decoder_x = Dense(64, activation='relu')(post_encode_x)
    decoder_x = LayerNormalization()(decoder_x)
    decoder_x = Dropout(0.1)(decoder_x)

    # Reshape decoder_x for MultiHeadAttention layer
    decoder_x = Reshape((1, 64))(decoder_x)

    for _ in range(2):
        # Self-Attention layer
        decoder_x = MultiHeadAttention(num_heads=2, key_dim=16)(decoder_x, decoder_x)
        decoder_x = Add()([decoder_x, decoder_x])  # Residual connection
        decoder_x = LayerNormalization()(decoder_x)

    outputs = Dense(1)(decoder_x)

    model = Model(inputs, outputs)
    return model

# Create Transformer model
# input_shape = (input_shape,)  # Assuming input_shape is defined elsewhere
# model = transformer_model(input_shape)


# Create Transformer model
# input_shape = (input_shape,)  # Assuming input_shape is defined elsewhere
# model = transformer_model(input_shape)


# Create Transformer model
input_shape = X_train.shape[1:]
model = transformer_model(input_shape)

# Compile model
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train model
history = model.fit(X_train, X_train, epochs=100, batch_size=16, shuffle=True, validation_data=(X_test, X_test))

# Predictions
predicted_sales_data_normalized = model.predict(sales_data_normalized)

# Inverse transform to get the predicted sales data
predicted_sales_data = scaler.inverse_transform(predicted_sales_data_normalized.reshape(-1, 1))

# Plot original and predicted sales data
# Plotting the original and predicted sales data
plt.figure(figsize=(14, 6))
plt.plot(date_list, sales_data, label='Original  Data', linewidth=2)
plt.plot(date_list, predicted_sales_data, label='Predicted  Data', linestyle='dashed', linewidth=2)
plt.title('Original vs Predicted  Data')
plt.xlabel('Time')
plt.ylabel('')
plt.legend()
plt.grid(True)
plt.show()