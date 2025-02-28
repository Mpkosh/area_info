# -*- coding: utf-8 -*-

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

file_path = 'app_package/src/for_mig_forecast/'


#Нормирование цен согласно инфляции
def normbyinf(inputdata):
    allrubfeatures = ['avgsalary', 'retailturnover', 'foodservturnover', 'agrprod', 'invest', 'budincome',
                      'funds', 'naturesecure', 'factoriescap']

    thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod']
    
    infdata = pd.read_csv(file_path+"inflation14.csv")
    
    # получить инфляцию за необходимый год
    year = inputdata['year'].values[0]
    inflation = infdata[infdata['year'] == year]['inf'].values[0]
    
    infnorm = 1 - inflation / 100
    inputdata.loc[:,thisrubfeatures] = inputdata[thisrubfeatures] * infnorm

    return inputdata


# Нормирование данных для модели (от 0 до 1)
def normformodel(inputdata):
    norm = pd.read_csv(file_path+"fornorm 24 all (IQR).csv")
    final = []
    tmp = []
    for k in range(len(inputdata)):
        for col in norm:
            if col != 'saldo':
                tmp.append(inputdata.iloc[k][col] / norm.iloc[0][col])

        final.append(tmp)
        tmp = []

    final = np.array(final)
    features = list(norm.columns[1:])
    final = pd.DataFrame(final, columns=features)
    inputdata = final
    return inputdata


# нормирование факторов на душу населения
def normpersoul(tonorm):
    # факторы для нормирования
    normfeat = ['avgemployers', 'shoparea', 'foodseats', 'retailturnover', 'sportsvenue', 'servicesnum',
                'livestock', 'harvest', 'agrprod', 'beforeschool']

    for k in range(len(tonorm)):
        for col in normfeat:
            index = tonorm.columns.get_loc(col)
            tonorm.iloc[k, index] = float(tonorm.iloc[k][col] / tonorm.iloc[k]['popsize'])

    return tonorm


# определить кластер для входных данных
def whatcluster(inputdata):
    # нормализация входных данных
    #inputdata = inputproc(request)
    inputdata = normbyinf(inputdata)
    inputdata = normformodel(inputdata)

    # загрузка модели
    kmeans_model = joblib.load(file_path+'kmeans_model (24-all-iqr) 01.joblib')

    pred_cluster = kmeans_model.predict(inputdata)

    # получение данных о профиле кластера
    medians = pd.read_csv(file_path+'medians 01.csv')
    clust = -1
    for i in range(len(medians)):
        if medians.iloc[i]['clust'] == pred_cluster[0]:
            clust = i

    return "Муниципальное образование входит в кластер: №" + str(pred_cluster[0]) +" - " + medians.iloc[clust]['profile']


# поиск наиболее близки поселений на основе социально-экономических индикаторов
def siblingsfinder(inputdata):

    # нормализация входных данных
    inputdata = normbyinf(inputdata)
    inputdata = normformodel(inputdata)
    inputdata = normpersoul(inputdata)

    #загрузка датасета
    data = pd.read_csv(file_path+"superdataset-24 alltime-clust (oktmo+name+clust) 01-normbysoul.csv")

    # наиболее близкие среди всех кластеров
    # shape (14, 9575)
    all_data_feats = data.iloc[:,6:21].values.T
    input_feats = np.repeat(inputdata.iloc[0][1:].values[None,:], 
                                 len(data), axis=0).T
    dist1 = list(mean_squared_error(all_data_feats, input_feats,
                                    multioutput='raw_values'))

    # сортировка датафрейма согласно отклонению (dist1)
    data['dist1'] = dist1
    data = data.sort_values(by='dist1')

    # выделение топ-10 наиболее близких (похожих)
    return data.iloc[:10,:3].reset_index(drop=True)#.to_dict()


# разница от наиболее близкого из лучшего кластера
def headtohead(inputdata):
    # обработка входных данных
    inputdata = normbyinf(inputdata)
    inputdata = normformodel(inputdata)
    inputdata = normpersoul(inputdata)

    #загрузка датасета
    data = pd.read_csv(file_path+"superdataset-24 alltime-clust (oktmo+name+clust) 01-normbysoul.csv")

    # наиболее близкий из лучшего кластера
    migprop = 0.0
    bestcluster = 0
    # определение лучшего кластера
    for k in range(int(data['clust'].max()) + 1):
        tmpdata = data[data['clust'] == k]
        msaldo = tmpdata['saldo'].median()
        mpopsize = tmpdata['popsize'].median()
        if k == 0:
            migprop = float(msaldo / mpopsize)
        else:
            if migprop < float(msaldo / mpopsize):
                migprop = float(msaldo / mpopsize)
                bestcluster = k

    # вычисление наиболее близкого МО среди лучшего кластера
    dist = []
    tmpdata = data[data['clust'] == bestcluster]
    for i in range(len(tmpdata)):
        tmp = mean_squared_error(tmpdata.iloc[i][6:21], inputdata.iloc[0][1:])  # кроме popsize
        dist.append(tmp)

    # сортировка датафрейма согласно отклонению (dist)
    tmpdata['dist'] = dist
    tmpdata = tmpdata.sort_values(by='dist')

    #вычисление разницы между наиболее близким и заданным
    dif = []
    for col in inputdata:
        dif.append(float(tmpdata.iloc[0][col] / inputdata.iloc[0][col]))

    dif = np.array(dif)
    features = list(inputdata.columns)
    dif = pd.DataFrame(dif, index=features)

    #return tmpdata.iloc[0].to_dict(), 
    return dif


# вычисляется во сколько раз входные данные отличаются от центра лучших кластеров
# по каждому социально-экономическому индикатору
def reveal(inputdata):
    # обработка входных данных
    # выброр медиан кластеров согласно уровню МО
    if inputdata.iloc[0]['type'] == 'all':
        filename = 'medians 01.csv'
    else:
        filename = 'medians only mundist.csv'

    medians = pd.read_csv(file_path+filename)

    # сортировка от лучшего кластера к худшему (согласно критерию)
    medians = medians.sort_values(by=['migprop'], ascending=False)

    medians = normpersoul(medians)
    inputdata = normbyinf(inputdata)
    inputdata = normpersoul(inputdata)

    changes = []
    tmp = []
    # вычисление разницы входа от медиан лучшего кластера (согласно профилю)
    for i in range(len(medians)):
        if inputdata.iloc[0]['profile'] == medians.iloc[i]['profile']:
            for col in inputdata.iloc[:, 4:]:
                tmp.append(float(medians.iloc[i][col] / inputdata.iloc[0][col]))

            changes.append(tmp)
            tmp = []
            break

    features = list(inputdata.iloc[:, 4:].columns)
    changes = np.array(changes)
    changes = pd.DataFrame(changes, columns=features)

    return changes
