# import module
from sklearn.model_selection import train_test_split

def train_test_splitter(dataset):
    # seperate input features in x
    x = dataset.drop('price', axis=1)

    # store the target variable in y
    y = dataset['price']
    
   
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234)
    
    print("Train test split shape: ")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test