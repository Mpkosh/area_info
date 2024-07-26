import pandas as pd
import numpy as np
import re


def clean_indexes(x):
    '''
    Удаление ненужных слов: 'от 3-5' -> '3-5'.
    '''
    # чтобы остались данные для трудостпособный возраст/старше 30/до 60
    stop_words = ['старше','до','возраста']
    if any(sw in x for sw in stop_words):
        return x
    else:
        return re.findall('\d*-?\d+', x)[0]
    
    
def get_mor_rate(morrate_file='morrate.xlsx', female_str_in_df='Женщины',
                 male_str_in_df='Мужчины'):
    '''
    Предобработка файла с коэффициентами смертности.
    Параметры:
        morrate_file -- файл, с которым работаем;
        female_str_in_df -- название строки для данных по женщинам;
        male_str_in_df -- название строки для данных по мужчинам.
    Вывод:
        Датасет с данными на каждый возраст.
    '''
    # получение коэф. смертности
    df = pd.read_excel(morrate_file, header=1, 
                         usecols=['Cohort','avg%','avg%.1'])
    df.columns=[df.columns[0],'avg% male','avg% female']
    age_clm = df.columns[0] 
    
    # уберем лишние знаки
    df[age_clm] = df[age_clm].apply(lambda x: x.replace('--','-'))

    # добавим новые когорты
    new_ages = ['75-79', '80-84', '85-89', '90-94', 
                            '95-99', '100']
    df.loc[df[age_clm]=='70-',age_clm] = '70-74'
    to_add = pd.DataFrame([new_ages,
                           [0]*6, [0]*6]).T
    to_add.columns=[age_clm, 'avg% male', 'avg% female']
    df = pd.concat([df, to_add], axis=0).reset_index(drop=True)
    
    # уберем лишние слова
    df[age_clm] = df[age_clm].apply(clean_indexes)
    
    # линейно распределим процент доживающих по когортам до заданных величин
    male_70, female_70 = df[df[age_clm]=='70-74'].values[0,1:]
    male_to_100 = np.linspace(male_70, 0.3, num=7)
    female_to_100 = np.linspace(female_70, 0.4, num=7)
    
    # обновим для новых когорт коэф-т смертности
    df.loc[df[age_clm].isin(new_ages), 'avg% male'] = male_to_100[1:]
    df.loc[df[age_clm].isin(new_ages), 'avg% female'] = female_to_100[1:]
    
    # запишем величину для каждого возраста
    df_ages = pd.DataFrame([i for i in range(0,101)], columns=['группа'])
    df_ages.loc[:,female_str_in_df] = 1
    df_ages.loc[:,male_str_in_df] = 1
        
    # если были данные для 100 и более лет
    if '100' in df[age_clm].values:
        # сразу запишем соответстующие данные
        df_ages.loc[df_ages['группа']==100,male_str_in_df
                   ] = df[df[age_clm]=='100'].values[0,1]
        df_ages.loc[df_ages['группа']==100,female_str_in_df
                   ] = df[df[age_clm]=='100'].values[0,2]
        # уберем из интервалов "100"
        range_list = df[age_clm].values[:-1]
    else:
        range_list = df[age_clm].values

    # раскроем интервалы: коэф-т смертности для каждого возраста
    for i in range_list:
        arr = np.arange(int(i.split('-')[0]), 
                        int(i.split('-')[1])+1)
        for num in arr:
            df_ages.loc[df_ages['группа']==num, male_str_in_df] = df[df[age_clm]==i
                                                 ].iloc[:,1].values[0]
            df_ages.loc[df_ages['группа']==num, female_str_in_df] = df[df[age_clm]==i
                                                 ].iloc[:,2].values[0]      
    
    return df_ages


def two_clm_df(filename='migbyage.xlsx', new_clm_name='доля'):
    '''
    Предобработка данных с двумя колонками и интервалами.
    Параметры:
        filename -- файл, с которым работаем;
        new_clm_name -- название новой колонки с данными.
    Вывод:
        Датасет с данными на каждый возраст.
    '''
    if filename.split('.')[-1] == 'xlsx':
        df = pd.read_excel(filename)
    else:
        df = pd.read_csv(filename)
    
    age_clm = df.columns[0]
    data_col = df.columns[1]
    
    # уберем лишние слова
    df[age_clm] = df[age_clm].apply(clean_indexes)
    
    # возьмем начальный и конечный возраста
    first_range = df[age_clm].values[0]
    last_range = df[age_clm].values[-1]
    first_age = int(first_range.split('-')[0])
    last_age = int(last_range.split('-')[-1])

    # запишем величину для каждого возраста
    df_ages = pd.DataFrame([i for i in range(first_age, 
                                             last_age+1)], columns=['группа'])
    df_ages.loc[:,new_clm_name] = 0
    
    # если были данные для 100 и более лет
    if '100' in df[age_clm].values:
        # сразу запишем соответстующие данные
        df_ages.loc[df_ages['группа']==100,new_clm_name
                   ] = df[df[age_clm]=='100'][data_col].values[0]
        # уберем из интервалов "100"
        range_list = df[age_clm].values[:-1]
    else:
         range_list = df[age_clm].values
            
    # раскроем интервалы        
    for range_str in range_list:
        arr = np.arange(int(range_str.split('-')[0]), 
                        int(range_str.split('-')[1])+1)
        for num in arr:
            df_ages.loc[df_ages['группа']==num,new_clm_name
                       ] = df[df[age_clm]==range_str][data_col].values[0]
            
    return df_ages