import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# 
np.random.seed(42)
days = np.arange(365)
baseline = 150
trend = 0.05 * days
seasonal = 50 * np.sin(2 * np.pi * days / 365)
noise = np.random.normal(0, 2, size=365)

#
for _ in range(5):
    random_indices = np.random.choice(365, size=30, replace=False)
    noise[random_indices] += np.random.uniform(0, 12, size=30)
    

sales_data = baseline + trend + seasonal + noise

# 
start_date = datetime(2023, 1, 1)
date_list = [start_date + timedelta(days=i) for i in range(365)]

# 
# plt.figure(figsize=(14, 6))
# plt.plot(date_list, sales_data, label='', linewidth=2)
# plt.title('')
# plt.xlabel( '')
# plt.ylabel('')
# plt.legend()
# plt.grid(True)
# plt.show()



import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense



# Normalize data
scaler = MinMaxScaler()
sales_data_normalized = scaler.fit_transform(sales_data.reshape(-1, 1))

# Split data into train and test sets
X_train, X_test = train_test_split(sales_data_normalized, test_size=0.2, random_state=42)

# Autoencoder model
input_layer = Input(shape=(1,))
encoded = Dense(32, activation='relu')(input_layer)
decoded = Dense(1, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Training the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, shuffle=True, validation_data=(X_test, X_test))

# Encoding data using the trained autoencoder
encoded_sales_data = autoencoder.predict(sales_data_normalized)

# Inverse transform to get the predicted sales data
predicted_sales_data = scaler.inverse_transform(encoded_sales_data)

# Calculate reconstruction error for the training data
reconstruction_error = np.mean(np.square(sales_data_normalized - encoded_sales_data))

# Print reconstruction error
print(f'Reconstruction Error: {reconstruction_error}')

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