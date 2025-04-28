import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X = np.random.rand(1000, 10, 10)
    y = np.random.choice([0, 1], size=(1000,))
    X_flat = X.reshape((1000, -1))
    
    model = create_model(X_flat.shape[1])
    model.fit(X_flat, y, epochs=10, batch_size=32, validation_split=0.2)
    
    new_scaffold = np.random.rand(1, 10, 10).reshape((1, -1))
    prediction = model.predict(new_scaffold)
    print(f"Probability of Hair Cell Regeneration: {prediction[0][0]:.2f}")
