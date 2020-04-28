import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

Encode = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import KFold

import xlsxwriter


def splitValidation(datasetValidation):
    x_validation = datasetValidation.iloc[:, 0]
    y_validation = datasetValidation.iloc[:, 1]
    return x_validation, y_validation


def adjust(opis_train, kategoria_train, opis_test, kategoria_test):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', clf), ])
    text_clf.fit(opis_train, kategoria_train)

    predicted = text_clf.predict(opis_test)
    accuracy = metrics.accuracy_score(kategoria_test, predicted)
    return accuracy


# Start programu
dataset = pd.read_csv('Files\Dane-do-walidacji.csv', delimiter=';', encoding='utf-8')
datasetValidation = pd.read_csv('Files\Dane-do-walidacji.csv', delimiter=';', encoding='utf-8')

results = xlsxwriter.Workbook("Wyniki.xlsx")
worksheet = results.add_worksheet("Wyniki")
row = 0
col = 0

worksheet.write(row, col, "Nazwa_algorytmu")
worksheet.write(row, col + 1, "Acc")
worksheet.write(row, col + 2, "Acc_walidacji")
worksheet.write(row, col + 3, "Fold1")
worksheet.write(row, col + 4, "Fold2")
worksheet.write(row, col + 5, "Fold3")
worksheet.write(row, col + 6, "Fold4")
worksheet.write(row, col + 7, "Fold5")
worksheet.write(row, col + 8, "Fold_sredni")
row += 1

X = dataset.iloc[:, 0]
y = dataset.iloc[:, 1]

# Podaje nazwy klasyfikatorow
names = ["RandomForestClassifier", "KNeighborsClassifier", "MultinomialNB", "DecisionTreeClassifier", "BernoulliNB",
         "AdaBoostClassifier", "LogisticRegression", "SVC1", "SVC2", "SVC3"]

# Okreslam klasyfikatory
classifiers = [
    RandomForestClassifier(max_depth=10000, n_estimators=100, max_features=100),
    KNeighborsClassifier(3),
    MultinomialNB(),
    DecisionTreeClassifier(max_depth=10000),
    BernoulliNB(),
    AdaBoostClassifier(),
    LogisticRegression(),
    SVC(kernel="linear", C=1),
    SVC(gamma=2, C=1),
    SVC(kernel="sigmoid", C=1)]

# Okreslam liczbe foldow
cv = KFold(n_splits=5, random_state=100, shuffle=True)

# Dataset jest dzielony na dane do nauki i testu
dataset['class_label'] = Encode.fit_transform(dataset['klasa'])
x_train2, x_test2, y_train2, y_test2 = train_test_split(dataset["opis"],
                                                        dataset['class_label'],
                                                        random_state=1, train_size=0.8)

datasetValidation['class_label'] = Encode.fit_transform(datasetValidation['klasa'])
x_trainValdation, x_validation, y_trainValdation, y_validation = train_test_split(datasetValidation["opis"],
                                                                                  datasetValidation['class_label'],
                                                                                  random_state=1, train_size=0.0025)

for names, clf in zip(names, classifiers):
    count = 1
    average = 0
    print(names)

    name = clf.__class__.__name__
    acc = adjust(x_train2, y_train2, x_test2, y_test2)
    acc2 = adjust(x_train2, y_train2, x_validation, y_validation)

    print(names + '  accuracy = ' + str(acc * 100) + '%')
    print(names + '  Validation accuracy = ' + str(acc2 * 100) + '%')

    # Zapisuje do pliku
    worksheet.write(row, col, names)
    worksheet.write(row, col + 1, acc)
    worksheet.write(row, col + 2, acc2)

    n_col = 3
    for train_index, test_index in cv.split(X):
        x_train, x_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        name = clf.__class__.__name__
        acc3 = adjust(x_train, y_train, x_test, y_test)

        # print("Fold " + str(count) + ": " + names + '  accuracy = ' + str(acc * 100) + '%')
        worksheet.write(row, n_col, acc3)
        n_col += 1

        average = average + acc3
        count = count + 1
    print("Average Fold: accuracy = " + str(average / (count - 1) * 100) + '%')
    print("Train size: ", len(x_train), len(x_train2), "Test size: ", len(x_test), "Walidation test size: ",
          len(x_validation))
    worksheet.write(row, col + 8, average / (count - 1))
    row += 1

results.close()
