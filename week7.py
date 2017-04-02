#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import time
import datetime
from sklearn import cross_validation
import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
#Подход 1 - п.1
features = pandas.read_csv('./features.csv', index_col='match_id')
X_pred = features[features.columns[0:102]]
t = features['radiant_win']

#Подход 1 - п.2
print 'Podhod 1 - p.2'
fsize = len(t)
for i in range(0,102):
    tmp = X_pred[X_pred.columns[i]]
    if tmp.count() < fsize:
        print tmp.name
        name = X_pred.columns[i]

# Признаки с пропусками:
#first_blood_time
#first_blood_team
#first_blood_player1
#first_blood_player2
#radiant_bottle_time
#radiant_courier_time
#radiant_flying_courier_time
#radiant_first_ward_time
#dire_bottle_time
#dire_courier_time
#dire_flying_courier_time
#dire_first_ward_time

#Следующие признаки:
#first_blood_time
#first_blood_team
#first_blood_player1
#first_blood_player2
#                   могут быть пустыми, если за первые 5 мин не произошло событие "первая кровь"

#Следующие признаки:
#radiant_bottle_time, dire_bottle_time: может быть пустыми, если не было приобретения командой предмета "bottle"
#radiant_courier_time, dire_courier_time: может быть пустыми, если не было приобретения командой предмета "courier"
#radiant_flying_courier_time, dire_flying_courier_time: может быть пустыми, если не было приобретения командой предмета "flying_courier"


#Подход 1 - п.3
X_tmp = X_pred.fillna(0)
X = X_tmp

#Подход 1 - п.4
y = features['radiant_win']
# целевая переменная radiant_win

#Подход 1 - п.5
print 'Podhod 1 - p.5'

for i in [10, 20, 30, 40, 50]:
    print 'Tree count:', i
    start_time = datetime.datetime.now()
    clf = GradientBoostingClassifier(n_estimators=i)
    clf.fit(X, y)
    kfold = cross_validation.KFold(y.size, n_folds=5, shuffle=True)
    scores = cross_validation.cross_val_score(clf, X, y, cv=kfold, scoring='roc_auc').mean()
    print 'Time elapsed:', datetime.datetime.now() - start_time
    print 'AUC Score:', scores


#Tree count: 10
#Time elapsed: 0:00:47.782000
#AUC Score: 0.664316763143
#Tree count: 20
#Time elapsed: 0:01:29.439000
#AUC Score: 0.682878011989
#Tree count: 30
#Time elapsed: 0:02:10.498000
#AUC Score: 0.688860168075
#Tree count: 40
#Time elapsed: 0:02:50.896000
#AUC Score: 0.694572871967
#Tree count: 50
#Time elapsed: 0:03:33.427000
#AUC Score: 0.697938009376

#Подход 2 - п.1
print 'Podhod 2 - p.1'
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
bestVal=0.0
bestC=0.0
for i in [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]:
    print 'C:', i
    start_time2 = datetime.datetime.now()
    clf2 = LogisticRegression(penalty='l2',C=i)
    clf2.fit(X_scaled, y)
    kfold2 = cross_validation.KFold(y.size, n_folds=5, shuffle=True)
    scores2 = cross_validation.cross_val_score(clf2, X_scaled, y, cv=kfold2, scoring='roc_auc')
    print 'Time elapsed:', datetime.datetime.now() - start_time2
    roc = scores2.mean()
    if roc > bestVal:
        bestVal = roc
        bestC = i

    print 'AUC Score:', roc


print 'Best C:', bestC
print 'Best ROC_AUC', bestVal


#C: 0.0001
#Time elapsed: 0:00:09.600000
#AUC Score: 0.711301287283
#C: 0.001
#Time elapsed: 0:00:17.490000
#AUC Score: 0.716372126648
#C: 0.01
#Time elapsed: 0:00:23.205000
#AUC Score: 0.716509920134
#C: 0.1
#Time elapsed: 0:00:24.605000
#AUC Score: 0.71641703315
#C: 1.0
#Time elapsed: 0:00:24.999000
#AUC Score: 0.716497261098
#C: 10
#Time elapsed: 0:00:24.713000
#AUC Score: 0.716488677578
#C: 100
#Time elapsed: 0:00:25.039000
#AUC Score: 0.716346092187
#C: 1000
#Time elapsed: 0:00:25.682000
#AUC Score: 0.716574203876
#C: 10000
#Time elapsed: 0:00:27.666000
#AUC Score: 0.716453608385

#Best C: 1000
#Best ROC_AUC 0.716574203876


#Подход 2 - п.2
print 'Podhod 2 - p.2'

X_Mod = X.copy()
del X_Mod['lobby_type']
del X_Mod['r1_hero']
del X_Mod['r2_hero']
del X_Mod['r3_hero']
del X_Mod['r4_hero']
del X_Mod['r5_hero']
del X_Mod['d1_hero']
del X_Mod['d2_hero']
del X_Mod['d3_hero']
del X_Mod['d4_hero']
del X_Mod['d5_hero']


X_scaled_filtered = scaler.fit_transform(X_Mod)
bestVal2=0.0
bestC2=0.0
for i in [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]:
    print 'C:', i
    start_time3 = datetime.datetime.now()
    clf3 = LogisticRegression(penalty='l2',C=i)
    clf3.fit(X_scaled_filtered, y)
    kfold3 = cross_validation.KFold(y.size, n_folds=5, shuffle=True)
    scores3 = cross_validation.cross_val_score(clf3, X_scaled_filtered, y, cv=kfold3, scoring='roc_auc')
    print 'Time elapsed:', datetime.datetime.now() - start_time3
    roc2 = scores3.mean()
    if roc2 > bestVal2:
        bestVal2 = roc2
        bestC2 = i

    print 'AUC Score:', roc2


print 'Best C:', bestC2
print 'Best ROC_AUC', bestVal2


#C: 0.0001
#Time elapsed: 0:00:09.832000
#AUC Score: 0.711275732188
#C: 0.001
#Time elapsed: 0:00:19.515000
#AUC Score: 0.716375241821
#C: 0.01
#Time elapsed: 0:00:23.318000
#AUC Score: 0.716588074087
#C: 0.1
#Time elapsed: 0:00:24.624000
#AUC Score: 0.716318817038
#C: 1.0
#Time elapsed: 0:00:21.813000
#AUC Score: 0.716397579033
#C: 10
#Time elapsed: 0:00:22.632000
#AUC Score: 0.716460186952
#C: 100
#Time elapsed: 0:00:22.350000
#AUC Score: 0.716429564073
#C: 1000
#Time elapsed: 0:00:22.688000
#AUC Score: 0.716533874603
#C: 10000
#Time elapsed: 0:00:22.288000
#AUC Score: 0.716426911293

#Best C: 0.01
#Best ROC_AUC 0.716588074087

#Подход 2 - п.3
print 'Podhod 2 - p.3'
un = np.unique(X[['r1_hero', 'r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']])
print 'Unique hero count:',un.size
#Unique hero count: 108

#Подход 2 - п.4
N = 112
X_pick = np.zeros((X.shape[0], N))

for i, match_id in enumerate(X.index):
    for p in xrange(5):
        X_pick[i, X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_changed = np.concatenate([X_Mod, X_pick], axis=1)

#Подход 2 - п.5
print 'Podhod 2 - p.5'

X_scaled_changed = scaler.fit_transform(X_changed)

bestVal3=0.0
bestC3=0.0
for i in [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]:
    print 'C:', i
    start_time4 = datetime.datetime.now()
    clf4 = LogisticRegression(penalty='l2',C=i)
    clf4.fit(X_scaled_changed, y)
    kfold4 = cross_validation.KFold(y.size, n_folds=5, shuffle=True)
    scores4 = cross_validation.cross_val_score(clf4, X_scaled_changed, y, cv=kfold4, scoring='roc_auc')
    print 'Time elapsed:', datetime.datetime.now() - start_time4
    roc3 = scores4.mean()
    if roc3 > bestVal3:
        bestVal3 = roc3
        bestC3 = i

    print 'AUC Score:', roc3


print 'Best C:', bestC3
print 'Best ROC_AUC', bestVal3


#C: 0.0001
#Time elapsed: 0:00:15.159000
#AUC Score: 0.742656038172
#C: 0.001
#Time elapsed: 0:00:29.285000
#AUC Score: 0.75170769036
#C: 0.01
#Time elapsed: 0:00:41.263000
#AUC Score: 0.751664381293
#C: 0.1
#Time elapsed: 0:00:43.268000
#AUC Score: 0.751845971152
#C: 1.0
#Time elapsed: 0:00:44.346000
#AUC Score: 0.751906998593
#C: 10
#Time elapsed: 0:00:44.203000
#AUC Score: 0.751910856251
#C: 100
#Time elapsed: 0:00:44.825000
#AUC Score: 0.751666243748
#C: 1000
#Time elapsed: 0:00:43.844000
#AUC Score: 0.751899581344
#C: 10000
#Time elapsed: 0:00:43.115000
#AUC Score: 0.75187639412

#Best C: 10
#Best ROC_AUC 0.751910856251

#Подход 2 - п.6
print 'Podhod 2 - p.6'
features_test = pandas.read_csv('./features_test.csv', index_col='match_id')
X_test_tmp = features_test.fillna(0)
X_test = X_test_tmp

X_test_pick = np.zeros((X_test.shape[0], N))

for i, match_id in enumerate(X_test.index):
    for p in xrange(5):
        X_test_pick[i, X_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_test_pick[i, X_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

del X_test['lobby_type']
del X_test['r1_hero']
del X_test['r2_hero']
del X_test['r3_hero']
del X_test['r4_hero']
del X_test['r5_hero']
del X_test['d1_hero']
del X_test['d2_hero']
del X_test['d3_hero']
del X_test['d4_hero']
del X_test['d5_hero']

X_new_changed = np.concatenate([X_test, X_test_pick], axis=1)

X_new_scaled_changed = scaler.fit_transform(X_new_changed)

clf5 = LogisticRegression(penalty='l2', C=10.0)
clf5.fit(X_scaled_changed, y)
result = clf5.predict_proba(X_new_scaled_changed)[:, 1]
#print result

print 'Max Val:', np.amax(result)
print 'Min Val:', np.amin(result)

#Max Val: 0.996485840045
#Min Val: 0.00868980489475