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
    n_age_groups = 1
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
    
    # ____ Половозрастная структура территории
    pop_df = PopInfoForAPI.get_detailed_pop(session, 
                                            territory_id, True, False)
    
    pop_df = get_predictions(pop_df, forecast_until, given_year)
    # ____ Численность всех групп
    soc_groups=['0-2','3-6','7-13','14-15','16-17',
                '18-35','36-59','60-75','75-100']
    soc_pyramid = []

    for age_group in soc_groups:
        start = int(age_group.split('-')[0])
        finish = int(age_group.split('-')[1])
        soc_pyr_part = pop_df.iloc[start:finish+1]
        soc_pyramid.append(soc_pyr_part)
    
    citizens_by_age_gr = soc_pyramid
    citizens_with_kids = get_preg_and_with_kids(pop_df)
    citizens_by_religion = get_religions(pop_df)    
    citizens_students = get_students(pop_df) 
    citizens_disabled = get_disabled(pop_df) 
    
    return [*citizens_by_age_gr, *citizens_with_kids,
           *citizens_by_religion, citizens_students, 
            citizens_disabled]


def get_religions(pop_df):
    names = ['Христиане','Мусульмане','Буддисты',
                      'Иудеи/евреи', 'Атеисты','Другое']
    
    # процент верующих по годам и по религиям
    religions = pd.read_csv(file_path+'religions_year.csv',index_col=0
                           ).set_index('Вера')
    # процент верующих по полу и возрастам
    rel_by_sex_and_age = pd.read_csv(file_path+'rel_by_sex_and_age.csv', 
                                     index_col=0, header=[0,1])

    sum_pop = pop_df.groupby(level=0, axis=1).sum().sum()
    rel_year_n = pd.DataFrame([], columns = pop_df.columns.levels[0],
                              index = names)

    fin_df = pd.DataFrame([],columns = pd.MultiIndex.from_product(
                                            [pop_df.columns.levels[0],
                                             names,
                                             ['Мужчины','Женщины']]
                                        ),
                             index = np.arange(0,101))


    for year in pop_df.columns.levels[0]:
        rel_by_sex_and_age_n = rel_by_sex_and_age.copy()
        if f'{year}' in religions.columns:
            rel_perc_year = religions[f'{year}']/100
        else:
            # если заданного года нет, то берем последний
            rel_perc_year = religions.iloc[:,-1]/100
        # Перед каждым вызовом multinomial! иначе буянит
        np.random.seed(27) 
        # Христиане, ... , Другое за 1 год.
        rels_n = np.random.multinomial(sum_pop[year], rel_perc_year)
        rel_year_n.loc[:,year]  = rels_n
        
        # для каждой религии берем общую сумму и делим на возраста и полы
        for i, rel in enumerate(names):
            men_women_p = rel_by_sex_and_age[rel].values.flatten()
            np.random.seed(27) 
            men_women_n = np.random.multinomial(rels_n[i], men_women_p)
            rel_by_sex_and_age_n.loc[:,(rel,"Мужчины")] = men_women_n[:83]
            rel_by_sex_and_age_n.loc[:,(rel,"Женщины")] = men_women_n[83:]

            fin_df.loc['18':,(year,rel,"Мужчины")] = men_women_n[:83]
            fin_df.loc['18':,(year,rel,"Женщины")] = men_women_n[83:]
    
    fin_dfs_list = []
    for rel in names:
        fin_dfs_list.append( fin_df.loc[:,(slice(None),rel,slice(None))
                                       ].droplevel(1, axis=1).fillna(0)
                           )
    return fin_dfs_list


def get_preg_and_with_kids(pop_df):
    # ____ Горожане с детьми
    # берем беременных в долях
    n_years = len(pop_df.columns.levels[0])
    profiles = pd.read_csv(file_path+'profiles_spb19.csv')
    prof_clms = profiles.columns[profiles.columns.str.endswith('Беременные')]
    profile = pd.concat([profiles[prof_clms]]*n_years, axis=1)
    # 1. беременные в числах
    pregnant_n = profile * pop_df.iloc[:,1::2].values
    pregnant_n.columns = pd.MultiIndex.from_product([pop_df.columns.levels[0],
                                                ['Женщины']])
    
    
    # доли одинаковые, тк изначально просто умножали на года
    frac2prob = (profile / profile.sum()).iloc[:,0]

    # считаем общее кол-во детей по группам
    kids_groups=['0-0','1-3','4-6','7-11','12-18']
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
        
    return studs_pyramid

    
def get_disabled(pop_df):
    dis_df_fraq = pd.read_csv(file_path+'disabled_fraq.csv',
                              index_col=0, header=[0,1])
    years = pop_df.columns.levels[0]
    
    # дублируем год, если в реальной пирамиде больше данных
    last_y_df = int(dis_df_fraq.columns.levels[0][-1])
    diff_years = years[-1] - last_y_df
    # если нужно меньше данных -- отрезаем
    if diff_years<0:
        for i in range(abs(diff_years)):
            dis_df_fraq = dis_df_fraq.iloc[:,:-2]
            
    else:
        for i in range(diff_years):
            part = dis_df_fraq.iloc[:,-2:].copy()
            part.columns = pd.MultiIndex.from_product([[f'{last_y_df+1}'],
                                                       ['men','women']])
            dis_df_fraq = pd.concat([dis_df_fraq, part], axis=1)
            last_y_df+= 1
    
    # если в пирамиде позже начинаются данные
    first_y_df = int(dis_df_fraq.columns.levels[0][0])
    if years[0] > first_y_df:
        dis_df_fraq = dis_df_fraq.loc[:,f'{years[0]:}':]
    # если в пирамиде раньше начинаются данные, то дополянем
    else:
        diff_years = first_y_df - years[0]
        for i in range(diff_years):
            part = dis_df_fraq.iloc[:,:2].copy()
            part.columns = pd.MultiIndex.from_product([[f'{first_y_df-1}'],
                                                       ['men','women']])
            dis_df_fraq = pd.concat([part, dis_df_fraq], axis=1)
            first_y_df-= 1
    
    
    
    pop_dis = pd.DataFrame(0, columns=pop_df.columns, index=pop_df.index)
    for age_br in dis_df_fraq.index:

        for sex in ['men','women']:
            df_age_str=''
            if sex=='men':
                sex_str = 'Мужчины'
                if age_br=='able':
                    df_age_str = '18-59'
                elif age_br=='old':
                    df_age_str = '60-100'
            else:
                sex_str = 'Женщины'
                if age_br=='able':
                    df_age_str = '18-54'
                elif age_br=='old':
                    df_age_str = '55-100'

            if df_age_str!='':
                start, finish = list(map(int, df_age_str.split('-')))
            else: 
                start, finish = list(map(int, age_br.split('-')))

            # находим кол-во студентов в интервале
            age_group_vals = pop_df.loc[start:finish,
                                        (slice(None),sex_str)].sum()
            age_group_dis = (age_group_vals * (dis_df_fraq.loc[age_br, 
                                                               (slice(None),sex)]
                                              ).values).values.astype(int)

            if age_group_dis.sum()!=0:
                for idx, year in enumerate(years):
                    # размазываем линейно по возрастам
                    probs = np.linspace(0.1, 1, num=finish-start+1)
                    probs = probs/probs.sum()
                    for idx, year in enumerate(years):
                        np.random.seed(27) 
                        # раскидываем числа с учетом вероятностей
                        q = np.random.multinomial(age_group_dis[idx], probs)
                        pop_dis.loc[start:finish,(year,sex_str)] = q


            df_age_str=''
            
    return pop_dis