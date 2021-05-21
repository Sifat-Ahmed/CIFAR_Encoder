from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def copy_weights(model, classifier):
    model = model
    classifier = classifier

    count_layers = 0
    for layer in model.layers:
        count_layers += 1
        if layer.name == 'encoded':
            break

    for l1, l2 in zip(classifier.layers[:count_layers], model.layers[:count_layers]):
        l1.set_weights(l2.get_weights())

    for layer in classifier.layers[:count_layers]:
        layer.trainable=False


    return classifier


def classification_stats(y_pred, y_true):
    print(classification_report(y_pred=y_pred, y_true=y_true))
    cms = confusion_matrix(y_pred=y_pred, y_true=y_true)

    sns.heatmap(cms, linewidths=1, annot=True, fmt='g')
