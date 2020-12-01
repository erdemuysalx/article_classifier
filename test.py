import train
#from train import X_test, y_test, test_df, model, history
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score


def decode_label(score):
    return 'Material Science' if score > 0.5 else 'Chemistry'


def plot_confusion_matrix(y_test, y_pred):
    '''
    Plots confusion matrix.
    '''

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False, cmap="YlGnBu")
    plt.ylabel('True label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    plt.show()


def plot_loss_accuracy(history, save=True):
    '''
    Plots loss vs epoch and accuracy vs epoch graphs.
    '''

    figures_dir = os.path.join(os.getcwd(), 'dcipher-nlp-challenge/figures/')
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)

    with open(os.path.join(os.getcwd(), 'dcipher-nlp-challenge/log/training.log'), 'r') as f:
        log = f.read()

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(['acc', 'loss']):
        ax[i].plot(history.history[metric])
        ax[i].plot(history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])
        ax[i].grid(True)

        if save:
            fig.savefig(figures_dir + metric + '_plot.jpg')

    plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    model = load_model('./model/model.03-0.70.h5')
    print("Loaded model from disk.")

    y_pred = model.predict(train.X_test, verbose=1)
    y_pred_decoded = [decode_label(pred) for pred in y_pred]
    loss, score = model.evaluate(x=train.X_test, y=train.y_test)
    print("Test accuracy:", score)
    print("Test AUC score:", roc_auc_score(train.y_test, y_pred))
    print(classification_report(list(train.test_df.label), y_pred_decoded))
    plot_loss_accuracy(train.history)
    plot_confusion_matrix(train.test_df.label.to_list(), y_pred_decoded)
