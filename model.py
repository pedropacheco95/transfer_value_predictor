import pandas as pd
import numpy as np
import seaborn as sns
import csv
import unicodedata
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Softmax, Permute, Multiply, Lambda, BatchNormalization
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

class FNNAM:

    def __init__(self):
        self.model = None
        self.target_column_name = None

        self.train_X = None
        self.test_X = None
        self.validation_X = None

        self.train_y = None
        self.test_y = None
        self.validation_y = None

    def data_preparation(self,df):
        # Preprocessing: Select features and the target
        features = df.drop(self.target_column_name, axis=1)  # Assuming all other columns are features
        target = df[self.target_column_name]
        target_transformed = np.log1p(target)

        # Normalize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        total_length = len(df)

        # Calculate split points
        split_point1 = int(total_length * 0.85)
        split_point2 = int(total_length * 0.95)

        # Split the dataset into training, validation, and test sets
        self.train_X = features_scaled[:split_point1, :]
        self.test_X = features_scaled[split_point1:split_point2, :]
        self.validation_X = features_scaled[split_point2:, :]

        self.train_y = target_transformed[:split_point1]
        self.test_y = target_transformed[split_point1:split_point2]
        self.validation_y = target_transformed[split_point2:]

        # Reshape for LSTM: [samples, time steps, features]
        self.train_X = self.train_X.reshape((self.train_X.shape[0], 1, self.train_X.shape[1]))
        self.test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))
        self.validation_X = self.validation_X.reshape((self.validation_X.shape[0], 1, self.validation_X.shape[1]))

    def create_model(self):
        def attention_layer(inputs, name):
            input_dim = int(inputs.shape[1])
            attention_probs = Dense(input_dim, activation='sigmoid', name=name)(inputs)
            attention_mul = Multiply()([inputs, attention_probs])
            return attention_mul

        # Feedforward Model with Attention
        input_layer = Input(shape=(self.train_X.shape[1], self.train_X.shape[2]))
        attention_mul = attention_layer(input_layer, 'attention_probs')
        hidden_layer1 = Dense(64, activation='sigmoid')(attention_mul)
        hidden_layer1 = BatchNormalization()(hidden_layer1)
        dropout_layer1 = Dropout(0.4)(hidden_layer1)
        hidden_layer2 = Dense(128, activation='sigmoid')(dropout_layer1)
        hidden_layer2 = BatchNormalization()(hidden_layer2)
        dropout_layer2 = Dropout(0.3)(hidden_layer2)
        hidden_layer3 = Dense(64, activation='sigmoid')(dropout_layer2)
        hidden_layer3 = BatchNormalization()(hidden_layer3)
        dropout_layer3 = Dropout(0.2)(hidden_layer3)
        output_layer = Dense(1)(dropout_layer3)

        fnnam_model = Model(inputs=[input_layer], outputs=output_layer)

        # Compile the model
        fnnam_model.compile(optimizer='rmsprop', loss='mean_absolute_error')

        self.model = fnnam_model

        return fnnam_model

    def train_model(self,plot_graph=False):
        # Train the model
        history = self.model.fit(self.train_X, self.train_y, epochs=400, batch_size=256, validation_data=(self.test_X, self.test_y), verbose=2)

        if plot_graph:
            # Plotting the loss
            sns.set(style="darkgrid")
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss Over Epochs')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            output_directory = 'images/'
            output_filename = 'loss.png'
            plt.savefig(output_directory + output_filename)
            plt.close()

        return history
    
    def get_trained_model(self,df,target_column_name):
        self.target_column_name = target_column_name
        self.data_preparation(df)
        self.create_model()
        self.train_model(plot_graph=True)
        self.save_model()
        return self.evaluate_model(plot_graph=True)
    
    def make_prediction(self,X,already_scaled=False):
        if not already_scaled:
            # Normalize the features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        return np.expm1(self.model.predict(X).flatten())
    
    def evaluate_model(self,plot_graph=False):
        y_pred = np.expm1(self.model.predict(self.validation_X).flatten())
        exp_validation = np.expm1(self.validation_y)
        mse = mean_squared_error(exp_validation, y_pred)
        mae = mean_absolute_error(exp_validation, y_pred)
        r2 = r2_score(exp_validation, y_pred)

        if plot_graph:
            plt.figure(figsize=(10, 6))
            plt.scatter(exp_validation, y_pred, alpha=0.5)

            plt.title('Scatter Plot of Predictions vs Validation Feature')
            plt.xlabel('Selected Feature from Validation Data')
            plt.ylabel('Predicted Value')
            output_directory = 'images/'
            output_filename = 'evaluation.png'
            plt.savefig(output_directory + output_filename)
            plt.close()

        return {'mse':mse,'mae':mae,'r2':r2}
    
    def save_model(self):
        self.model.save('fnnam.keras')
        return True