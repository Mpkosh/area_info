# -*- coding: utf-8 -*-

import pandas as pd
import joblib
import numpy as np

file_path = 'app_package/src/for_mig_forecast/'


#Нормирование цен согласно инфляции
def normbyinf(inputdata, infdata, year):
    # признаки для ценового нормирования
    allrubfeatures = ['avgsalary', 'retailturnover', 'foodservturnover', 'agrprod', 'invest', 'budincome',
                      'funds', 'naturesecure', 'factoriescap']

    thisrubfeatures = ['avgsalary', 'retailturnover', 'agrprod']
    
    infdata = pd.read_csv(file_path+"inflation14.csv")
    
    # получить инфляцию за необходимый год
    inflation = infdata[infdata['year'] == year]['inf'].values[0]
    
    infnorm = 1 - inflation / 100
    inputdata.loc[:,thisrubfeatures] = inputdata[thisrubfeatures] * infnorm

    return inputdata.iloc[0]


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
    
    return inputdata.iloc[0], norm.iloc[0]['saldo']


def model_outcome(inputdata):
    # загрузка модели; sklearn == 1.2.2!
    model = joblib.load(file_path+'migpred (24, tree).joblib')

    startyear = 2024    # начальная точка прогноза, т.е. первый прогноз делается на 25-ый год
    endyear = int(inputdata.iloc[0]['year'])
    inputdata = inputdata.iloc[:, 1:]  # отрезать показатель year

    # нормализация согласно инфляции
    infdata = pd.read_csv(file_path+"inflation14.csv")
    dataforpred = []
    dataforpred.append(np.array(normbyinf(inputdata, infdata, startyear)))

    # список в датафрейм
    dataforpred = pd.DataFrame(dataforpred, columns=inputdata.columns)

    #нормализация под модель прогноза
    maxsaldo = 0
    for i in range(len(dataforpred)):
        dataforpred.iloc[i], maxsaldo = normformodel(dataforpred.iloc[[i]])

    # выполнение прогноза
    predsaldo = 0
    prediction = model.predict(dataforpred)
    prediction = prediction * maxsaldo
    predsaldo = int(np.sum(prediction))
    
    fin_saldo = predsaldo * (endyear - startyear)

    return [inputdata['popsize'].values[0]+fin_saldo, fin_saldo] 