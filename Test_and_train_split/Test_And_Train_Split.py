from sklearn.model_selection import train_test_split

def test_and_train_split(features, labels, df):
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.30, random_state=1)

    return X_train, X_test, y_train, y_test, indices_train, indices_test