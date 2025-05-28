# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 17:30:35 2025

@author: user
"""

import requests
import pandas as pd
import numpy as np
from app_package.src import PreproDF, PopulationInfo, DemForecast, PopInfoForAPI

file_path='app_package/src/for_soc_profiles/'
file_dir = 'app_package/src/population_data/'


def get_predictions(df, forecast_until, given_year):
    last_pop_year = df.columns.levels[0][-1]
    df.index = df.index.astype(int)
    # если нужен год больше текущего -- делаем прогноз
    if forecast_until>0:
        if forecast_until > last_pop_year:
            folders={'popdir':file_dir,
                 'file_name':'Ленинградская область.xlsx'}
            forecast_df = DemForecast.MakeForecast(df, last_pop_year, 
                                                forecast_until - last_pop_year, folders)
        else:
            forecast_df = pd.DataFrame()
        # отрезаем от прогноза первый год (== поданному на вход последнему году)
        df = pd.concat([df, forecast_df.iloc[:, 2:]], axis=1)

    else:
        if given_year > last_pop_year:
            folders={'popdir':file_dir,
                 'file_name':'Ленинградская область.xlsx'}
            df = DemForecast.MakeForecast(df, last_pop_year, 
                                                given_year - last_pop_year, folders)
        df = df[[given_year]]
    # тк в мультииндексе, если выбирать колонки или удалять, они остаются    
    df.columns = df.columns.remove_unused_levels()    
    return df


def get_profiles(territory_id, forecast_until=0, given_year=2023):
    session = requests.Session()
    unpack_after_70 = False
    last_year = True
    
    # ____ Половозрастная структура территории
    pop_df = PopInfoForAPI.get_detailed_pop(session, territory_id, True, False)
    pop_df = PopInfoForAPI.get_detailed_pop(session, territory_id, 
                                        unpack_after_70=unpack_after_70, 
                                        last_year=last_year, 
                                        specific_year=given_year)
    # если в БД нет данных по пирамиде
    if pop_df.shape[0] == 0:
        pop_df = PopInfoForAPI.estimate_child_pyr(session, territory_id, 
                                                  unpack_after_70, last_year, given_year)
        
    pop_df = get_predictions(pop_df, forecast_until, given_year)
    # ____ Численность всех групп
    soc_groups=['0-13',
                '14-15',
                #'16-59',
                '16-17', 
                '18-59',
                #'60-100', 
                '60-75','75-100',
               '60-100']
    soc_pyramid = []

    for age_group in soc_groups:
        start = int(age_group.split('-')[0])
        finish = int(age_group.split('-')[1])
        soc_pyr_part = pop_df.iloc[start:finish+1]
        soc_pyramid.append(soc_pyr_part.astype(int))
    
    citizens_by_age_gr = soc_pyramid
    citizens_with_kids = get_preg_and_with_kids(pop_df)
    citizens_by_religion = get_religions(pop_df)    
    citizens_students = get_students(pop_df) 
    #citizens_disabled = get_disabled(pop_df) 
    citizens_with_pets = get_with_pets(pop_df) 
    
    res = [*citizens_by_age_gr, *citizens_with_kids,
           *citizens_by_religion, citizens_students, 
            citizens_with_pets]
    
    for i in res:
        if i.loc[:,(slice(None),'Мужчины')].sum().sum() > 0:
            i.loc[:,(slice(None),'Мужчины')] *= -1
            
    return res


def get_religions(pop_df):
    names = ['Христиане','Мусульмане','Иудеи','Буддисты']
    # https://www.findeasy.in/population-of-russia/
    percs = [0.474, 0.065, 0.01, 0.05]

    list_df_rels = []
    for i,rel in enumerate(names):
        pop_rel = pd.DataFrame(0, columns=pop_df.columns, index=pop_df.index)

        for year in pop_df.columns.levels[0]:
            for sex in ['Мужчины','Женщины']:

                pop = pop_df.loc[:, (year, sex)].values
                np.random.seed(27) 
                q = np.random.multinomial(int(pop.sum()* percs[i]), 
                                              pop[18:]/pop[18:].sum())

                pop_rel.loc[18:, (year,sex)] = q

        list_df_rels.append(pop_rel)
        
    return list_df_rels


def get_preg_and_with_kids(pop_df):
    # ____ Горожане с детьми
    # берем беременных в долях
    n_years = len(pop_df.columns.levels[0])
    profiles = pd.read_csv(file_path+'profiles_spb19.csv')
    prof_clms = profiles.columns[profiles.columns.str.endswith('Беременные')]
    # 1. беременные в числах
    pregnant_n = (pop_df*profiles[prof_clms].values).astype(int)
    
    # доли одинаковые, тк изначально просто умножали на года
    profile = pd.concat([profiles[prof_clms]]*n_years, axis=1)
    frac2prob = (profile / profile.sum()).iloc[:,0]

    # считаем общее кол-во детей по группам
    kids_groups=['0-0','1-6','7-13','14-18']
    children = []
    for age_group in kids_groups:
        start = int(age_group.split('-')[0])
        finish = int(age_group.split('-')[1])
        soc_pyr_part = pop_df.iloc[start:finish+1]
        res = pd.Series(soc_pyr_part.T.groupby(level=[0]).sum().sum(1), 
                        name=age_group)
        children.append(res)  
    
    # на каждую женщину по 1.41 ребенка
    children_df = pd.DataFrame(np.array(children)).T / 1.41
    children_df.index = pop_df.columns.levels[0]
    children_df.columns = [kids_groups]
    children_df
    
    women_fraqs = pd.DataFrame()
    
    # сколько в долях должно быть детей разных возрастов у женщин
    # (новорожденные у беременных женщин .shift(1))
    # (ages 1-3 у беременных женщин .shift(2)+.shift(3)+.shift(4) в доли)
    for age_group in kids_groups:
        start = int(age_group.split('-')[0])
        finish = int(age_group.split('-')[1])

        shifts = np.array([frac2prob.shift(i+1).values for 
                           i in range(start,finish+1)])
        shifts[np.isnan(shifts)] = 0

        women_fraqs_age = pd.DataFrame( shifts.sum(0) / shifts.sum(0).sum(),
                                      columns=[age_group])
        women_fraqs = pd.concat([women_fraqs,women_fraqs_age], axis = 1)
        
    # Женщины с младенцами, 1-3, 4-6, 7-11, 12-18
    citizens_with_kids_by_ages = []
    
    for age_group in kids_groups:
        women_children = pd.DataFrame()

        for year in children_df.index:
            n_kids = children_df.loc[year, age_group].values[0]

            # numpy seed нужно перед каждым вызовом ф-ии
            np.random.seed(27) 
            # раскидываем числа с учетом вероятностей
            women_with_0 = np.random.multinomial(n_kids, 
                                                 women_fraqs[age_group])
            women_with_0_df = pd.DataFrame(women_with_0, columns=[year])
            women_children = pd.concat([women_children, women_with_0_df], axis=1)
        
        # Мужчин ставим так же
        women_and_men = np.repeat(women_children.values, 2, axis=1)
        clms = pd.MultiIndex.from_product([pop_df.columns.levels[0],
                                           ['Мужчины', 'Женщины']],
                                          names=['', 'пол'])
        w_m_df = pd.DataFrame(women_and_men, columns=clms)
        w_m_df.index.name = 'группа'
        citizens_with_kids_by_ages.append(w_m_df)
        
    return [pregnant_n,*citizens_with_kids_by_ages]


def get_students(pop_df):
    fraq_by_age = pd.read_csv(file_path+'Р2_12_мж_fraq.csv', sep=';', 
                              index_col=0, decimal=",").iloc[:,:2]/100
    
    studs_pyramid = pop_df.copy()
    # раньше 13 и старше 56 считаем, что студентов нет 
    studs_pyramid.loc[:"13"] = 0
    studs_pyramid.loc["56":] = 0
    
    # от 14 до 29 есть доля студентов для каждого возраста
    studs_pyramid.loc['14':"29",
                      (slice(None),'Мужчины')
                     ] = pop_df.loc['14':"29",(slice(None),'Мужчины')
                                   ].mul(fraq_by_age.iloc[1:-3,0].values, 
                                         axis="index")
    studs_pyramid.loc['14':"29",
                      (slice(None),'Женщины')
                     ] = pop_df.loc['14':"29",(slice(None),'Женщины')
                                   ].mul(fraq_by_age.iloc[1:-3,0].values, 
                                         axis="index")

    # для остальных есть доля студентов только для интервала возрастов
    # находим кол-во студентов в интервале; размазываем линейно по возрастам
    age_gs = ['30-34','35-39','40-55']
    years = studs_pyramid.columns.levels[0]

    for age_br in age_gs:
        for sex in ['men','women']:
            if sex=='men':
                sex_str = 'Мужчины'
            else:
                sex_str = 'Женщины'

            start, finish = list(map(int, age_br.split('-')))
            # находим кол-во студентов в интервале
            age_group_vals = studs_pyramid.loc[f'{start}':f"{finish}",
                                               (slice(None),sex_str)].sum()
            age_group_studs = (age_group_vals * fraq_by_age.loc[age_br, sex]
                              ).values.astype(int)
            
            # размазываем линейно по возрастам
            probs = np.linspace(1, 0.1, num=finish-start+1)
            probs = probs/probs.sum()
            for idx, year in enumerate(years):
                np.random.seed(27) 
                # раскидываем числа с учетом вероятностей
                q = np.random.multinomial(age_group_studs[idx], probs)

                studs_pyramid.loc[f'{start}':f"{finish}", 
                                  (year, sex_str)] = q
        
    return studs_pyramid.astype(int)

    
def get_with_pets(pop_df):
    # https://www.ipsos.ru/sites/default/files/ct/news/documents/2024-04/Всероссийская_перепись_животных.pdf
    pets_age_fraq = pd.DataFrame(index=pop_df.index, columns=[0])
    pets_age_fraq.loc[0:15,0] = 0
    
    # среднее от кошек и собак
    age_gs = ['16-24','25-34','35-44','45-54','55-64','65-100']
    percs = [0.135,0.15,0.21,0.18,0.17,0.15]

    for i, age_br in enumerate(age_gs):
        start, finish = list(map(int, age_br.split('-')))
        pets_age_fraq.loc[start:finish,0] = [percs[i]]*(finish-start+1)
    
    pets_df = (pop_df * pets_age_fraq.values).astype(int)
    return pets_df
