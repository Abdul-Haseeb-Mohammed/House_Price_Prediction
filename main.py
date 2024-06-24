from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import train_test_split
from src.models.train_model import train_linear_regression, train_decision_tree_regressor, train_random_forest_regressor
from src.models.predict_model import evaluate_model
from src.visualization.visualize import plot_tree
if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "Productionalize_Regression_Models/data/raw/final.csv"
    df = load_and_preprocess_data(data_path)

    # Split the dataset into train test
    X_train, X_test, y_train, y_test = train_test_split(df)
    
    #Train the Linear regression model
    lrmodel = train_linear_regression(X_train, y_train)

    # Show the metrics of models
    print("Coefficients of Linear Regression:", lrmodel.coef_)
    print("Intercept of Linear Regression:", lrmodel.intercept_)
    
    # Display evaluation metrics for linear regression
    lr_train_mae, lr_test_mae = evaluate_model(lrmodel, X_train, X_test, y_train, y_test)
    print('Linear Regression Train error is', lr_train_mae)
    print('Linear Regression Test error is', lr_test_mae)
    
    #Train the Decision Tree Regressor model
    dtr_model = train_decision_tree_regressor(X_train, y_train)
        
    # Display evaluation metrics for Decision Tree Regressor
    dtr_train_mae, dtr_test_mae = evaluate_model(dtr_model, X_train, X_test, y_train, y_test)
    print('Decision Tree Regressor Train error is', dtr_train_mae)
    print('Decision Tree Regressor Test error is', dtr_test_mae)
    
    #Plot the decision tree
    plot_tree(dtr_model, dtr_model.feature_names_in_, save_path='reports/figures/decision_tree.png')
    
    #Train the Random Forest Regressor
    rfr_model = train_random_forest_regressor(X_train, y_train)
        
    # Display evaluation metrics for Random Forest Regressor
    rfr_train_mae, rfr_test_mae = evaluate_model(rfr_model, X_train, X_test, y_train, y_test)
    print('Random Forest Regressor Train error is', rfr_train_mae)
    print('Random Forest Regressor Test error is', rfr_test_mae)
    
    #Plot the Random Forest Regressor
    plot_tree(rfr_model.estimators_[2], dtr_model.feature_names_in_, save_path='reports/figures/decision_tree.png')
    
