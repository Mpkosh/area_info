# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:41:59 2024

@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app_package.src import PreproDF
from app_package.src.AuxFilePrepro import get_mor_rate


'''Расчет миграционного сальдо'''
def GetMigSaldo(df, folders):
    mr = get_mor_rate(folders['popdir']+folders['file_name'][:-5]+" morrate.xlsx")
    saldo=pd.DataFrame()
    saldo['группа']=df.index
    years=list(set(i[0] for i in df.columns))
    years.sort()
    for i in range(1,len(years)):
        delta = df[years[i]] - (df[years[i-1]]*mr[['Женщины','Мужчины']].values).shift(1)  
        delta.iloc[0]=[0,0]
        saldo[[(years[i], 'Женщины'),(years[i], 'Мужчины')]]=delta.round().astype(int)
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
        br.iloc[ind, 1]=(brates[brates.Cohort==i]['avg births on 1000 female (year)']/1000).to_list()*len(ind)
    #Коэффициентики смертности
    mr = get_mor_rate(folders['popdir']+folders['file_name'][:-5]+" morrate.xlsx")
    forecast=pd.DataFrame()
    forecast=df.loc[:, year0:year0]
    for i in range(1,horizon+1):
        #Сдвиг пирамиды
        future=forecast.loc[:,year0+i-1:year0+i-1].shift(1).rename(columns={year0+i-1:year0+i})
        #Рождения
        neonatal=(future.loc[br['группа'],(year0+i,'Женщины')]*br['рождаемость'].values).dropna().sum()
        future.loc[0, (year0+i,'Женщины')]=neonatal*0.49
        future.loc[0, (year0+i,'Мужчины')]=neonatal*0.51
        #Смерти
        future[[(year0+i,'Женщины'), (year0+i,'Мужчины')]]*=mr[['Женщины','Мужчины']].values
        #Миграции
        years=np.sort(list(set(i[0] for i in df.columns)))
        ind=np.where(years<=year0)
        saldo=GetMigSaldo(df[years[ind]], folders)
        saldo.stack().mean(axis=1).unstack()
        future[[(year0+i,'Женщины'), (year0+i,'Мужчины')]]+=saldo.stack().mean(axis=1).unstack().values
        # иначе уезжает больше людей, чем вообще имеется
        future[future<0]=0
        forecast=(pd.concat([forecast, future], axis=1).fillna(value=0)).astype(int)
    return forecast


'''Всё население из таблицы по годам'''
def total(df):
    tot=df.sum()
    years=list(set(i[0] for i in tot.index))
    years.sort()
    pop=[tot[i].sum()/1000 for i in years]
    return np.array(years).astype(int), np.array(pop)


'''График всего населения'''
def plot_total(df, forecast, title=None):
    oldpop=total(df)
    newpop=total(forecast)
    plt.plot(*oldpop, marker='o', label='Реальные данные')
    plt.plot(*newpop, marker='o', label='Прогноз')
    plt.xlabel('Год')
    x=np.arange(oldpop[0].min(), newpop[0].max()+1)
    plt.xticks(x, x.astype(str))
    plt.ylabel('Население (тыс. чел.)')
    plt.title(title +' район')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return None


if __name__=='__main__':
    folders={'datadir':'../region_data/',
         'popdir':'../population_data/',
         'geodir':'../geo_data/',
          'file_name':'Ленинградская область.xlsx'}
    
    region='Лодейнопольский'
    # читаем
    df_ex = PreproDF.df_from_excel(file_name=folders['datadir']+folders['file_name'], area_name=region)
    # предобрабатываем датафрейм
    df = PreproDF.prepro_df(df_ex, morrate_file=folders['popdir']+'morrate.xlsx')
    horizon=5
    year0=2023
    forecast=MakeForecast(df, year0, horizon, folders)
    plot_total(df, forecast, region)