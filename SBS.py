from collections import Counter
from imblearn.over_sampling import SMOTE
from openpyxl import load_workbook
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

pathTrain = './小组作业数据-2021.xlsx'
dataset = []
X = []
y = []

wb = load_workbook(pathTrain)
ws = wb['训练和测试样本集合']
for i in list(ws.rows):
    data = [j.value for j in i]
    del (data[0])
    X.append(data[1:])
    y.append(data[0])

labels = X[0]

X = X[1:len(X) - 2]
y = y[1:len(y) - 2]

X = MinMaxScaler().fit_transform(np.array(X))

print('数据分布：', Counter(y))

smo = SMOTE(random_state=11)
X, y = smo.fit_resample(X, y)

print('过采样数据分布：', Counter(y))


# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=5)


class SBS():
    def __init__(self,
                 estimator,  # 评估器 比如LR，KNN
                 k_features,  # 想要的特征个数
                 scoring=accuracy_score,  # 对模型的预测评分
                 test_size=0.25,
                 random_state=1):

        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        # seperete train data and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        # dimension
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]  # 特征子集
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []  # 分数
            subsets = []  # 子集

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train,
                                         y_train,
                                         X_test,
                                         y_test,
                                         p)

                print(p, score)

                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


clf = RandomForestClassifier(n_estimators=130, max_features=None, max_depth=15, min_samples_split=20,
                             min_samples_leaf=4, random_state=0)

# selecting features
sbs = SBS(clf, k_features=1)
sbs.fit(X, y)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()
