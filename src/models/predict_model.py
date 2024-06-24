# Import accuracy score
from sklearn.metrics import mean_absolute_error

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Predict the price of the property on train set
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(train_pred, X_test)

    # Predict the price of the property on test set    
    test_pred = model.predict(y_train)
    test_mae = mean_absolute_error(test_pred, y_test)
    
    return train_mae, test_mae