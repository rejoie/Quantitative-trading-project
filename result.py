from matplotlib import font_manager
from openpyxl import load_workbook
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

path = './预测结果.xlsx'
dataset = []
X = []
y = []

wb = load_workbook(path)
ws = wb['Sheet1']
RF = [i.value for i in list(ws.columns)[0]][1:]
LR = [i.value for i in list(ws.columns)[1]][1:]
SVM = [i.value for i in list(ws.columns)[2]][1:]
ANN = [i.value for i in list(ws.columns)[3]][1:]


def plot_result(x, y, x_name, y_name):
    confmat = confusion_matrix(y_true=y, y_pred=x)

    # 将混淆矩阵可视化
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=round(confmat[i, j]/len(x),2), va='center', ha='center')

    plt.title('预测结果',fontproperties=font_manager.FontProperties(fname='C:\Windows\Fonts\msyh.ttc'))
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


plot_result(RF, LR, 'RF', 'LR')
plot_result(RF, SVM, 'RF', 'SVM')
plot_result(RF, ANN, 'RF', 'ANN')
plot_result(LR, SVM, 'LR', 'SVM')
plot_result(LR, ANN, 'LR', 'ANN')
plot_result(SVM, ANN, 'SVM', 'ANN')
