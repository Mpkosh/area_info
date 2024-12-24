# -*- coding: utf-8 -*-

import pandas as pd
import joblib
import numpy as np

file_path = 'app_package/src/for_mig_forecast/'


#Нормирование цен согласно инфляции
def normbyinf(inputdata):
    # признаки для ценового нормирования
    thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod']
    
    infdata = pd.read_csv(file_path+"inflation14.csv")
    for k in range(len(inputdata)):
        # получить инфляцию за необходимый год
        inflation = infdata[infdata['year'] == inputdata.iloc[k]['year']]   
        for col in thisrubfeatures:
            index = inputdata.columns.get_loc(col)
            inputdata.iloc[k, index] = inputdata.iloc[k][col] * (
                                            inflation.iloc[0]['inf'] / 100)

    return inputdata


# Нормирование данных для модели (от 0 до 1)
def normformodel(inputdata):
    
    norm = pd.read_csv(file_path+"fornorm-24.csv")
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
    return inputdata, norm.iloc[0]['saldo']


def model_outcome(inputdata):
    # загрузка модели; sklearn == 1.2.2!
    model = joblib.load(file_path+'migpred (24, tree).joblib')

    #нормализация входных данных
    inputdata = normbyinf(inputdata)
    # отрезать показатель year
    inputdata = inputdata.iloc[:, 1:]
    inputdata, maxsaldo = normformodel(inputdata)

    # выполнение прогноза
    prediction = model.predict(inputdata)
    prediction = prediction * maxsaldo
    inputdata['predsaldo'] = int(prediction)

    return inputdata['predsaldo']