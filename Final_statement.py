
# coding: utf-8

# In[5]:

import pandas


# In[6]:

# Импорт файла с обучающей выборкой

df = pandas.read_csv('C:/features.csv', index_col='match_id')


# In[7]:

# Признаки с пропущенными значениями

counts = pandas.DataFrame(df.count())
counts[counts[0] < 97230]


# In[8]:

# Удаляем признаки, связанные с итогами матча  

X = df.drop(["duration",
             "radiant_win",
             "tower_status_radiant",
             "tower_status_dire",
             "barracks_status_radiant",
             "barracks_status_dire"], axis=1)


# In[9]:

# Заполняем пропущенные значения нулем

X.fillna(0, inplace=True)


# In[10]:

# Проверяем что пропущенных значений нет

counts = pandas.DataFrame(X.count())
counts[counts[0] < 97230]


# In[11]:

# DataFrame с целевой переменной

Y = df["radiant_win"]


# In[12]:

# Импорт необходимых библиотек

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import numpy
import time
import datetime


# In[13]:

# Создаем генератор разбиений и массив с количеством деревьев

kfold = KFold(len(Y.index), n_folds=5, shuffle=True, random_state=241)
n_trees = [5, 10, 15, 20, 25, 30, 35]


# In[14]:

# Получаем значение метрики, а также время обучения для различного числа деревьев

for i in n_trees:
    start_time = datetime.datetime.now()
    print numpy.mean(cross_val_score(GradientBoostingClassifier(n_estimators=i), X, y=Y, scoring = 'roc_auc', cv=kfold))
    print 'Time elapsed with n_trees = ', i , ' - ', datetime.datetime.now() - start_time


# In[15]:

# Класс для проверки адекватности метрики AUC-ROC, встроенной в cross_val_score

class proba_gradient(GradientBoostingClassifier):
    def predict(self, X):
        return GradientBoostingClassifier.predict_proba(self, X)[:, 1]


# In[16]:

# Класс для проверки адекватности метрики AUC-ROC, встроенной в cross_val_score

class proba_logistic(LogisticRegression):
    def predict(self, X):
        return LogisticRegression.predict_proba(self, X)[:, 1]


# In[17]:

# Значение метрики для 150 деревьев 

roc_auc_score (Y, cross_val_predict(proba_gradient(n_estimators=150), X, y=Y, cv=kfold))


# In[18]:

# Масштабируем признаки для логистической регрессии

scaler = StandardScaler()
scaler.fit(X)
X_log = scaler.transform(X)


# In[19]:

X_log


# In[20]:

# Создаем объект GridSearchCV для подбора параметра регуляризации

Grid = GridSearchCV(LogisticRegression(penalty='l2'), [{'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}], cv=kfold,
                       scoring='roc_auc')


# In[21]:

# Подбираем оптимальный параметр регуляризации для выборки с категориальными признаками

Grid.fit(X_log,Y)
print Grid.best_params_
print Grid.best_score_


# In[22]:

# Проводим дополнительно кросс-валидацию и фиксируем время обучения

start_time = datetime.datetime.now()
numpy.mean(cross_val_score(LogisticRegression(penalty='l2', C=0.01), X_log, y=Y, scoring = 'roc_auc', cv=kfold))
print 'Time elapsed: ', datetime.datetime.now() - start_time


# In[23]:

# Убираем категориальные признаки и масштабируем

X_log_nocat = X.drop(['lobby_type',
                             'r1_hero',
                             'r2_hero',
                             'r3_hero','r4_hero','r5_hero',"d1_hero","d2_hero","d3_hero","d4_hero","d5_hero"], axis=1)

scaler.fit(X_log_nocat)
X_log_nocat = scaler.transform(X_log_nocat)


# In[24]:

# Проводим кросс-валидацию

numpy.mean(cross_val_score(LogisticRegression(penalty='l2', C=0.01), X_log_nocat, y=Y, scoring = 'roc_auc', cv=kfold))


# In[25]:

# Находим максимальное значение идентификатора героя

N = pandas.concat([X["r1_hero"],
               X["r2_hero"],
               X["r2_hero"],
               X["r4_hero"],
               X["r5_hero"],
               X["d1_hero"],
               X["d2_hero"],
               X["d3_hero"],
               X["d4_hero"],
               X["d5_hero"]]).unique().max()+1


# In[26]:

# Создаем мешок слов

X_pick = numpy.zeros((X.shape[0], N))
for i, match_id in enumerate(X.index):
    for p in xrange(5):
        X_pick[i, X.ix[match_id, 'r%d_hero' % (p+1)]] = 1
        X_pick[i, X.ix[match_id, 'd%d_hero' % (p+1)]] = -1


# In[27]:

# Соединяем мешок слов и остальные признаки

X_merged = numpy.concatenate((X_log_nocat, X_pick), axis = 1)


# In[28]:

# Подбираем оптимальный параметр регуляризации для выборки с мешком слов

Grid.fit(X_merged,Y)
print Grid.best_params_
print Grid.best_score_


# In[29]:

# Проводим кросс-валидацию

numpy.mean(cross_val_score(LogisticRegression(penalty='l2', C=0.1), X_merged, y=Y, scoring = 'roc_auc', cv=kfold))


# In[30]:

# Загружаем тестовую выборку

df_test = pandas.read_csv('C:/features_test.csv', index_col='match_id')
df_test.fillna(0, inplace=True)


# In[31]:

# Убираем категориальные признаки и масштабируем

X_log_nocat_test = df_test.drop(['lobby_type',
                             'r1_hero',
                             'r2_hero',
                             'r3_hero','r4_hero','r5_hero',"d1_hero","d2_hero","d3_hero","d4_hero","d5_hero"], axis=1)
X_log_nocat_test = scaler.transform(X_log_nocat_test)


# In[32]:

# Создаем мешок слов

X_pick_test = numpy.zeros((df_test.shape[0], N))
for i, match_id in enumerate(df_test.index):
    for p in xrange(5):
        X_pick[i, df_test.ix[match_id, 'r%d_hero' % (p+1)]] = 1
        X_pick[i, df_test.ix[match_id, 'd%d_hero' % (p+1)]] = -1


# In[33]:

# Соединяем мешок слов и остальные признаки

X_merged_test = numpy.concatenate((X_log_nocat_test, X_pick_test), axis = 1)


# In[34]:

# Объявляем классификатор

clf_log = LogisticRegression(penalty='l2', C=0.1)


# In[35]:

# Обучаем классификатор

clf_log.fit(X_merged, Y)


# In[36]:

# Создаем массив предсказанных значений

Y_pred = clf_log.predict_proba(X_merged_test)[:, 1]
print Y_pred.max()
print Y_pred.min()


# In[38]:

# Пишем в файл

Results = pandas.DataFrame()
Results["match_id"] = df_test.index
Results["radiant_win"] = Y_pred
Results.to_csv(path_or_buf = 'submission2.csv', index=False)

