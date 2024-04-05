from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

DATA_SIZE = 6000
TEST_RATIO = 0.3
SEED = 40


def hyper_tune(tuner, features_train, features_test, target_train, target_test):
    tuner.fit(features_train, target_train)
    best_estimator = tuner.best_estimator_
    score = best_estimator.score(features_test, target_test)
    print(f'best estimator: {best_estimator}\naccuracy: {score}\n')


(x, y), _ = mnist.load_data(path="mnist.npz")
x, y = x[:DATA_SIZE], y[:DATA_SIZE]
x = x.reshape((x.shape[0], -1))
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=TEST_RATIO,
                                                    random_state=SEED)
prep_norm = Normalizer()
x_train_norm = prep_norm.fit_transform(x_train)
x_test_norm = prep_norm.fit_transform(x_test)

param_grid_KNN = {'n_neighbors': [3, 4, 5],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'brute']}
param_grid_RF = {'n_estimators': [550, 500],
                 'max_features': ['sqrt', 'log2'],
                 'class_weight': ['balanced', 'balanced_subsample']}

grid_search_KNN = GridSearchCV(estimator=KNeighborsClassifier(),
                               param_grid=param_grid_KNN,
                               n_jobs=-1)
grid_search_RF = GridSearchCV(estimator=RandomForestClassifier(random_state=SEED),
                              param_grid=param_grid_RF,
                              n_jobs=-1)
print('K-nearest neighbours algorithm')
hyper_tune(grid_search_KNN, x_train_norm, x_test_norm, y_train, y_test)
print('Random forest algorithm')
hyper_tune(grid_search_RF, x_train_norm, x_test_norm, y_train, y_test)
