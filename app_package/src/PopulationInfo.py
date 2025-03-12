import pandas as pd
import numpy as np
from io import BytesIO
'''
import matplotlib.ticker as mticker
import matplotlib
matplotlib.use('AGG') # must be imported before pyplot
import matplotlib.pyplot as plt
'''
from app_package.src.AuxFilePrepro import get_mor_rate


def age_groups(df, n_in_age_group=5):
    '''
    Выбор нужных возрастных групп из датасета.
    Параметры:
        df -- датасет;
        n_in_age_group -- кол-во возрастов в интервале. 
                          (Ex: 5 -> 0-4, 5-9, 10-14, ...)
    Вывод:
        Датасет с нужными возрастными группами.
    '''

    if n_in_age_group == 1:
        chosen_age_groups = df
    
    # если интервалы не из стандартных   
    else:
        # если человек задал число возрастов в интервале
        if isinstance(n_in_age_group, int):
            groups = [f'{i}-{i+n_in_age_group-1}' for i in range(0,101-n_in_age_group, 
                                                                      n_in_age_group)]
            # добавляем возраста, чтобы было ровно до 100 лет
            last_age = int(groups[-1].split('-')[-1])

            if 100 - last_age > 1:
                groups += [f'{last_age+1}-100']
            else:
                groups += [100]
        # если человек перечислил новые интервалы   
        if isinstance(n_in_age_group, list): 
            groups = n_in_age_group

        result = []
        indexes = []
        df.index = df.index.astype(int)
        for age_group in groups:
            # если задан интервал    
            if '-' in str(age_group):
                # если интервала нет в изначальных данных
 
                if age_group not in df.index:
                    
                    # раскрываем интервал
                    age_brackets = [int(i) for i in age_group.split('-')]
                    # суммируем все значения по каждому возрасту
                    new = df[(df.index.isin([ i for i in range(age_brackets[0],
                                                                    age_brackets[1]+1) ]))]
                    
                    new = new.sum()
                else:
                    new = df[df.index==age_group].squeeze(axis=0)
            else:
                # если возраст есть в изначальных данных
                if int(age_group) in df.index:
                    new = df[df.index==int(age_group)].squeeze(axis=0)
                # else:
                    # print(f'Возраст {age_group} в данных нет.')

            indexes.append(age_group)
            result.append(new)

        new_df = pd.concat(result, axis=1).T
        new_df.index = indexes
        chosen_age_groups = new_df

    chosen_age_groups.index.name = 'группа' 
    
    return chosen_age_groups



def calc_mor_rate(df_ages_1, morrate_file='/population_data/morrateLO.xlsx'):
    '''
    Умножение на коэффициенты доживания.
    Параметры:
        df_ages_1 -- датасет с возрастами с интервалом 1 год; 
                     (0 лет, 1 год, ...)
        morrate_file -- файл с коэф-ми смертности.
    Вывод:
        Датасет.
    '''
    df = df_ages_1.copy()
    kmor = get_mor_rate(morrate_file)
    female_clm, male_clm = kmor.columns[1:]
    
    # умножаем на коэф-т доживания
    df.loc[:,df.columns.get_level_values('пол')=='Женщины'
          ] = df.loc[:,df.columns.get_level_values('пол')=='Женщины'
                        ].mul(kmor[female_clm], axis=0)
    df.loc[:,df.columns.get_level_values('пол')=='Мужчины'
              ] = df.loc[:,df.columns.get_level_values('пол')=='Мужчины'
                            ].mul(kmor[male_clm], axis=0)    
    return df


def expected_vs_real(df_ages_1, morrate_file='population_data/morrateLO.xlsx'):
    '''
    Вычисление разницы реальных значений с ожидаемыми (предыдущий год * коэф-т смертности).
    Параметры:
        df_ages_1 -- датасет с возрастами с интервалом 1 год; 
                     (0 лет, 1 год, ...) 
        morrate_file -- файл с коэф-ми смертности.
    Вывод:
        Датасет.
    '''
    # данные, умноженные на коэф-т доживания
    df_with_mr = calc_mor_rate(df_ages_1, morrate_file)
    
    # сдвигаем на год (теперь они находятся в колонке год+1 и под индексом возраст+1)
    to_be_expected = df_with_mr.shift(1).shift(2,axis=1)
    # не учитываем пустые данные: 0 лет и самый первый год (2014)
    to_be_expected = to_be_expected.iloc[1:,2:]

    # отнимаем реальные данные от вычисленных
    res = df_ages_1.iloc[1:,2:].sub(to_be_expected)
    return res


def group_by_age(difference_df, n_in_age_group=5):
    '''
    Группируем и суммируем по заданным возрастным интервалам.
    Параметры:
        difference_df -- датасет с примерной оценкой сальдо;
                        (разница реальных и ожидаемых значений);
        n_in_age_group -- кол-во возрастов в интервале. 
                          (Ex: 5 -> 0-4, 5-9, 10-14, ...)
    Вывод:
        Датасет
    '''
    # суммируем по возрастным интервалам
    df = difference_df.groupby(difference_df.index//n_in_age_group).sum()
    # составляем строки для новых возрастных интервалов
    groups = [f'{i}-{i+n_in_age_group-1}' for i in range(0,101-n_in_age_group, 
                                                         n_in_age_group)]
    # добавляем возраста, чтобы было ровно до 100 лет
    last_age = int(groups[-1].split('-')[-1])
    if 100 - last_age > 1:
        groups += [f'{last_age+1}-100']
    else:
        groups += ['100']
    df.index = groups
    
    return df.round(0)


   