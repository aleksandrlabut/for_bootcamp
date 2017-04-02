#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time,datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler



def check_if_None(df):
    print nr.format(" Колонки с пропусками ")
    holey_cols = []    
    for col in df.columns:
        if df[col].count() < len(df.index):
            print col
            holey_cols.append(col)
            
    return holey_cols

def run_prob_cv(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_prob[test_index] = clf.predict_proba(X_test)
        
    return y_prob, clf

def draw_roc(data):
    pl.clf()
    plt.figure(figsize=(8,6))
    for row in data:
        fpr, tpr, thresholds = roc_curve(row[0],row[1])
        roc_auc  = auc(fpr, tpr)
        pl.plot(fpr, tpr, label='%s ROC (area = %0.5f)' % (row[2],roc_auc))
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.legend(loc=0, fontsize='small')
    pl.show()
    
    
def draw_test_deviance(test_deviance,color='blue',label=''):
    plt.figure()
    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::2], test_deviance[::2],
            '-', color=color,label=label)

    plt.legend(loc='upper left')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Test Set Deviance')
    plt.show()        

if __name__ == '__main__':
    
    tab = "{:<30}"
    nr = "\n{:*^60}\n"    
    
    """
        
        Считайте таблицу с признаками из файла features.csv с помощью кода,
        приведенного выше. Удалите признаки, связанные с итогами матча 
        (они помечены в описании данных как отсутствующие в тестовой 
        выборке).
        
    """
    features = pd.read_csv('data/features.csv', index_col='match_id')
    t_features = pd.read_csv('data/features_test.csv', index_col='match_id')
    
    exclude = ['duration','tower_status_radiant','tower_status_dire','barracks_status_radiant','barracks_status_dire']
    features = features.filter(items=[col for col in features.columns if col not in exclude])
    """
    
    Проверьте выборку на наличие пропусков с помощью функции count(), 
    которая для каждого столбца показывает число заполненных значений. 
    Много ли пропусков в данных? Запишите названия признаков, имеющих 
    пропуски, и попробуйте для любых двух из них дать обоснование, 
    почему их значения могут быть пропущены.
    
    """
    holey_cols = check_if_None(features)
    """
    Замените пропуски на нули с помощью функции fillna(). 
    На самом деле этот способ является предпочтительным для 
    логистической регрессии, поскольку он позволит пропущенному 
    значению не вносить никакого вклада в предсказание. 
    Для деревьев часто лучшим вариантом оказывается замена 
    пропуска на очень большое или очень маленькое значение — 
    в этом случае при построении разбиения вершины можно будет 
    отправить объекты с пропусками в отдельную ветвь дерева. 
    Также есть и другие подходы — например, замена пропуска на 
    среднее значение признака. Мы не требуем этого в задании, 
    но при желании попробуйте разные подходы к обработке пропусков и 
    сравните их между собой.
    """
    features = features.fillna(0)
    t_features = t_features.fillna(0)
    """
    Какой столбец содержит целевую переменную? Запишите его название.
    """
    print nr.format(" Целевая переменная ")
    print "radiant_win"
    
    """
    Забудем, что в выборке есть категориальные признаки, и попробуем 
    обучить градиентный бустинг над деревьями на имеющейся матрице 
    "объекты-признаки". Зафиксируйте генератор разбиений для 
    кросс-валидации по 5 блокам (KFold), не забудьте перемешать 
    при этом выборку (shuffle=True), поскольку данные в таблице 
    отсортированы по времени, и без перемешивания можно столкнуться 
    с нежелательными эффектами при оценивании качества. Оцените качество 
    градиентного бустинга (GradientBoostingClassifier) с помощью данной 
    кросс-валидации, попробуйте при этом разное количество деревьев 
    (как минимум протестируйте следующие значения для количества 
    деревьев: 10, 20, 30). Долго ли настраивались классификаторы? 
    Достигнут ли оптимум на испытанных значениях параметра n_estimators, 
    или же качество, скорее всего, продолжит расти при дальнейшем его 
    увеличении?
    """
    X = np.array(features.filter(items = [col for col in features.columns if not col == 'radiant_win']).values)
    y = np.array(features['radiant_win'].values)
    
  
    n_estimators = 30
    start_time = datetime.datetime.now()
    pred_prob,clf = run_prob_cv(X,y,GradientBoostingClassifier,
        n_estimators=n_estimators)
        
    print nr.format(' Time elapsed, %s estimators' % n_estimators)
    print datetime.datetime.now() - start_time
        
    test_deviance = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X)):
        test_deviance[i] = clf.loss_(y, y_pred)
            
    draw_test_deviance(test_deviance)
    print "N_estimators %s ROC_AUC:" % n_estimators,roc_auc_score(y,pred_prob[:,1])
    draw_roc([(y,pred_prob[:,1],'Estimators %s' % n_estimators)])
  
    
    """
    Оцените качество логистической регрессии 
    (sklearn.linear_model.LogisticRegression с L2-регуляризацией) с 
    помощью кросс-валидации по той же схеме, которая использовалась 
    для градиентного бустинга. Подберите при этом лучший параметр 
    регуляризации (C). Какое наилучшее качество у вас получилось? 
    Как оно соотносится с качеством градиентного бустинга? 
    Чем вы можете объяснить эту разницу? Быстрее ли работает 
    логистическая регрессия по сравнению с градиентным бустингом?
    
    """
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)    
    
    curves = []
    scores = []
    for C in np.power(10.0, np.arange(-5, 5)):
        #start_time = datetime.datetime.now()
        pred_prob,clf = run_prob_cv(X_scaled,y,LogisticRegression,
                                    penalty='l2',C=C)
        #print nr.format(' Time elapsed, C=%s ' % C)
        #print datetime.datetime.now() - start_time
        scores.append((roc_auc_score(y,pred_prob[:,1]),C))                                    
        curves.append((y,pred_prob[:,1],'LR: L2,C=%s' % C))
        
    print nr.format(" COEF_ ")
    for i in range(len(X[0])):
        print features.columns[i],clf.coef_[0][i]
        
    scores.sort()
    scores.reverse()
 
    """
    Среди признаков в выборке есть категориальные, которые мы использовали как 
    числовые, что вряд ли является хорошей идеей. Категориальных признаков в 
    этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero, 
    d1_hero, d2_hero, ..., d5_hero. Уберите их из выборки, и проведите 
    кросс-валидацию для логистической регрессии на новой выборке с подбором 
    лучшего параметра регуляризации. Изменилось ли качество? 
    Чем вы можете это объяснить?
    """
  
    exclude = ['radiant_win','lobby_type','r1_hero','r2_hero','r3_hero','r4_hero',
               'r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']
    
    X = np.array(features.filter(items = [col for col in features.columns if col not in exclude]).values)
    
    scaler = StandardScaler()    
    X_scaled = scaler.fit_transform(X)
    
    pred_prob,clf = run_prob_cv(X_scaled,y,LogisticRegression,
                                penalty='l2',C=scores[0][1])
    curves.append((y,pred_prob[:,1],'LR filtered: L2,C=%s' % scores[0][1]))
    
    draw_roc(curves)                           
  
    """
    На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, 
    которые показывают, какие именно герои играли за каждую команду. 
    Это важные признаки — герои имеют разные характеристики, и некоторые из 
    них выигрывают чаще, чем другие. Выясните из данных, сколько различных 
    идентификаторов героев существует в данной игре (вам может пригодиться 
    фукнция unique или value_counts).
    """
    hero_cols = ['r1_hero','r2_hero','r3_hero','r4_hero',
               'r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']
    
    heroes = pd.Series(features.filter(hero_cols).values.ravel()).unique().tolist()
    print tab.format("Uniques "),len(heroes) 
    
    """
    Воспользуемся подходом "мешок слов" для кодирования информации о героях. 
    Пусть всего в игре имеет N различных героев. Сформируем N признаков, при 
    этом i-й будет равен нулю, если i-й герой не участвовал в матче; единице, 
    если i-й герой играл за команду Radiant; минус единице, если i-й герой играл 
    за команду Dire. Ниже вы можете найти код, который выполняет данной 
    преобразование. Добавьте полученные признаки к числовым, которые вы 
    использовали во втором пункте данного этапа.
    """
    exclude = ['radiant_win','lobby_type','r1_hero','r2_hero','r3_hero','r4_hero',
               'r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']
    
    X = np.array(features.filter(items = [col for col in features.columns if col not in exclude]).values)    
    
    # N — количество различных героев в выборке
    N = len(heroes)
    X_pick = np.zeros((features.shape[0], N))
    
    for i, match_id in enumerate(features.index):
        for p in xrange(5):
            X_pick[i, heroes.index(features.ix[match_id, 'r%d_hero' % (p+1)])] = 1
            X_pick[i, heroes.index(features.ix[match_id, 'd%d_hero' % (p+1)])] = -1    
    
    X = np.append(X,X_pick,axis=1)
    """
    Проведите кросс-валидацию для логистической регрессии на новой выборке с 
    подбором лучшего параметра регуляризации. Какое получилось качество? 
    Улучшилось ли оно? Чем вы можете это объяснить?
    """
    curves = []
    scores = []
    scaler = StandardScaler()    
    X_scaled = scaler.fit_transform(X)
    
    for C in np.power(10.0, np.arange(-5, 5)):
        pred_prob,clf = run_prob_cv(X_scaled,y,LogisticRegression,
                                penalty='l2',C=C)
        curves.append((y,pred_prob[:,1],'LR extended: L2,C=%s' % C))
        scores.append((roc_auc_score(y,pred_prob[:,1]),C)) 
    draw_roc(curves)

    """
    Постройте предсказания вероятностей победы команды Radiant для тестовой 
    выборки с помощью лучшей из изученных моделей (лучшей с точки зрения AUC-ROC 
    на кросс-валидации). Убедитесь, что предсказанные вероятности адекватные — 
    находятся на отрезке [0, 1], не совпадают между собой (т.е. что модель не 
    получилась константной).
    """
    
    scores.sort()
    scores.reverse()
    
    exclude = ['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero',
               'r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']
    
    X_test = np.array(t_features.filter(items = [col for col in t_features.columns if col not in exclude]).values)        
    X_pick = np.zeros((t_features.shape[0], N))
    for i, match_id in enumerate(t_features.index):
        for p in xrange(5):
            X_pick[i, heroes.index(t_features.ix[match_id, 'r%d_hero' % (p+1)])] = 1
            X_pick[i, heroes.index(t_features.ix[match_id, 'd%d_hero' % (p+1)])] = -1    
    
    X_test = np.append(X_test,X_pick,axis=1)
    X_test_scaled = scaler.fit_transform(X_test)
    
    pred_prob,clf = run_prob_cv(X_scaled,y,LogisticRegression,
                                penalty='l2',C=scores[0][1])
    pred_proba = clf.predict_proba(X_test_scaled)
 
    print tab.format('Минимальное значение'),np.min(pred_proba[:,1])
    print tab.format('Максимальное значение'),np.max(pred_proba[:,1])
    
    with open('data/predict.csv', 'w') as f:
        f.write("%s,%s\n" % ('match_id','radiant_win'))
        for i, match_id in enumerate(t_features.index):
            f.write("%d,%0.4f\n" % (match_id,pred_proba[:,1][i]))
        f.close()
    