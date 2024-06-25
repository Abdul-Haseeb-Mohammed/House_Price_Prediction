# Import accuracy score
from sklearn.metrics import mean_absolute_error

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Predict the price of the property on train set
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)

    # Predict the price of the property on test set    
    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    return train_mae, test_mae