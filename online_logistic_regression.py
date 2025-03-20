import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class OnlineLogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = np.zeros(n_features)
        self.learning_rate = learning_rate
        self.fixed_weights_mask = np.ones(n_features, dtype=bool)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights))
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def fix_weights(self, feature_indices):
        """Fix weights for specified features"""
        self.fixed_weights_mask[feature_indices] = False
    
    def unfix_weights(self, feature_indices):
        """Unfix weights for specified features"""
        self.fixed_weights_mask[feature_indices] = True
    
    def partial_fit(self, X, y):
        """Online learning with support for fixed weights"""
        y_pred = self.predict_proba(X)
        error = y - y_pred
        
        # Update only unfixed weights
        gradient = error.reshape(-1, 1) * X  # Reshape error to match X dimensions
        gradient_sum = np.sum(gradient, axis=0)  # Sum across samples
        self.weights[self.fixed_weights_mask] += self.learning_rate * gradient_sum[self.fixed_weights_mask]
        
        return self

def generate_sample_data(n_samples=1000):
    """Generate synthetic data for binary classification"""
    np.random.seed(42)
    X = np.random.randn(n_samples, 3)
    true_weights = np.array([1, -2, 0.5])
    y = (1 / (1 + np.exp(-np.dot(X, true_weights))) >= 0.5).astype(int)
    return X, y

def main():
    # Generate data
    X, y = generate_sample_data()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data for online learning
    X_train, X_test = X_scaled[:800], X_scaled[800:]
    y_train, y_test = y[:800], y[800:]
    
    # Initialize model
    model = OnlineLogisticRegression(n_features=3)
    
    # Fix the weight for the third feature
    model.fix_weights([2])
    
    # Training history
    train_accuracy = []
    
    # Online learning
    for i in range(len(X_train)):
        model.partial_fit(X_train[i:i+1], y_train[i:i+1])
        
        if (i + 1) % 100 == 0:
            y_pred = model.predict(X_train[:i+1])
            acc = accuracy_score(y_train[:i+1], y_pred)
            train_accuracy.append(acc)
    
    # Final evaluation
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(100, 801, 100), train_accuracy)
    plt.xlabel('Training samples')
    plt.ylabel('Accuracy')
    plt.title('Online Learning Progress')
    plt.grid(True)
    plt.show()
    
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print("Final weights:", model.weights)
    print("Fixed weights mask:", model.fixed_weights_mask)

if __name__ == "__main__":
    main() 