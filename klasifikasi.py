from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def klasifikasi_svm(c, X_train, y_train):
    model = svm.SVC(kernel="linear", gamma="scale", C=c).fit(X_train, y_train)
    return model

def prediksi_svm(model, X_test):
    predicted = model.predict(X_test)
    return predicted

def metrik_klasifikasi(y_test, predicted):
    cm = confusion_matrix(y_test, predicted)
    precision = precision_score(y_test, predicted, average='binary', pos_label=1)
    recall = recall_score(y_test, predicted, average='binary', pos_label=1)
    f1 = f1_score(y_test, predicted, average='binary', pos_label=1)

    return cm, precision, recall, f1


