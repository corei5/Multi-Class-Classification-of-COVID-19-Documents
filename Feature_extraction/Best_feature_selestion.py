from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def best_feature_selestion(df,labels):
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df, labels, test_size=0.30, random_state=1)
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
    sel.fit(X_train_1, y_train_1)
    selected_feat = X_train_1.columns[(sel.get_support())]
    return selected_feat