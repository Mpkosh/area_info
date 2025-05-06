# -*- coding: utf-8 -*-
"""
by A.N.
"""
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from app_package.src import PreproDF
from app_package.src.AuxFilePrepro import get_mor_rate


file_dir = 'app_package/src/population_data/'


def get_predictions(df, forecast_until, given_year, for_mig=False):
    last_pop_year = df.columns.levels[0][-1]
    n_age_groups = 1
    print('last_pop_year', last_pop_year)
    
    
    # если нужен год больше текущего -- делаем прогноз
    if forecast_until>0:
        df.index = df.index.astype(int)
        if forecast_until > last_pop_year:
            folders={'popdir':file_dir,
                 'file_name':'Ленинградская область.xlsx'}
            forecast_df = MakeForecast(df, last_pop_year, 
                                        forecast_until - last_pop_year, folders)
        else:
            forecast_df = pd.DataFrame()
        # отрезаем от прогноза первый год (== поданному на вход последнему году)
        df = pd.concat([df, forecast_df.iloc[:, 2:]], axis=1)

    else:
        if given_year > last_pop_year:
            df.index = df.index.astype(int)
            folders={'popdir':file_dir,
                 'file_name':'Ленинградская область.xlsx'}
            df = MakeForecast(df, last_pop_year, 
                              given_year - last_pop_year, folders)
        if for_mig:
            df = df[[given_year-1,given_year]]
        else:
            df = df[[given_year]]
    df.columns = df.columns.remove_unused_levels()    
    return df


'''Расчет миграционного сальдо'''
def GetMigSaldo(df, folders):
    mr = get_mor_rate(folders['popdir']+folders['file_name'][:-5]+" morrate.xlsx")

    saldo = pd.DataFrame()
    saldo['группа'] = df.index
    years = list(set(i[0] for i in df.columns))
    years.sort()
    

    for i in range(1,len(years)):
        delta = df[years[i]] - (df[years[i-1]] * mr[['Женщины','Мужчины']].values).shift(1)  
        delta.iloc[0] = [0,0]
        saldo[[(years[i], 'Женщины'),(years[i], 'Мужчины')]] = delta.round().astype(int)
        

    saldo.set_index('группа', inplace=True, drop=True)
    saldo.columns=pd.MultiIndex.from_tuples(saldo.columns)
    
    return saldo


'''Расчет прогноза со сдвигами, рождениями, смертями, миграциями'''
def MakeForecast(df, year0, horizon, folders):
    #Коэффициентики рождаемости
    brates = pd.read_excel(folders['popdir']+folders['file_name'][:-5]+" birthrate.xlsx")
    br=pd.DataFrame()
    br['группа']=np.arange(15,50)
    br['рождаемость']=np.zeros(len(br))
    
    for i in brates.Cohort:
        ind=br[(br['группа']>=int(i[:2]))&(br['группа']<=int(i[-2:]))].index
        br.iloc[ind, 1]=(brates[brates.Cohort==i]['avg births on 1000 female (year)'
                                                  ]/1000).to_list()*len(ind)
        
    #Коэффициентики смертности
    mr = get_mor_rate(folders['popdir']+folders['file_name'][:-5]+" morrate.xlsx")

    forecast=df.loc[:, year0:year0]
    
    for i in range(1,horizon+1):
        #Сдвиг пирамиды
        future=forecast.loc[:,year0+i-1:year0+i-1
                            ].shift(1).rename(columns={year0+i-1:year0+i})

        
        #Рождения
        neonatal=(future.loc[br['группа'],(year0+i,'Женщины')]*br['рождаемость'].values
                  ).dropna().sum()
        future.loc[0, (year0+i,'Женщины')]=neonatal*0.49
        future.loc[0, (year0+i,'Мужчины')]=neonatal*0.51
        
        #Смерти
        future[[(year0+i,'Женщины'), (year0+i,'Мужчины')]]*=mr[['Женщины','Мужчины']].values
        
        #Миграции
        years=np.sort(list(set(i[0] for i in df.columns)))
        ind=np.where(years<=year0)
        saldo=GetMigSaldo(df[years[ind]], folders)
        saldo.stack().mean(axis=1).unstack()
        future[[(year0+i,'Женщины'), 
                (year0+i,'Мужчины')]] += saldo.stack().mean(axis=1).unstack().values
        
        # иначе уезжает больше людей, чем вообще имеется
        future[future<0]=0
        forecast=(pd.concat([forecast, future], axis=1).fillna(value=0)).astype(int)
        
    return forecast


'''Всё население из таблицы по годам'''
def total(df):
    tot=df.sum()
    years=list(set(i[0] for i in tot.index))
    years.sort()
    pop = [tot[i].sum()/1000 for i in years]
    
    return np.array(years).astype(int), np.array(pop)
