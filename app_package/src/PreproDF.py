import pandas as pd
import numpy as np
import re
from app_package.src.AuxFilePrepro import clean_indexes


def df_from_db_path(path): 
    df = pd.read_csv(path, header=[1]).rename(columns = {"Unnamed: 0":"пол"})
    # year columns to int
    df_year_cols = [int(i[:4]) for i in df.columns[1:]]
    df.columns = [df.columns[0]] + df_year_cols
    return df


def df_from_excel(file_name='pandask.xlsx', area_name='Кингисеппский'):
    '''
    Получить датасет из excel-файла.
    Параметры:
        file_name -- excel-файл, откуда берем данные по населению;
        area_name -- район, для которого рисуем графики;
    Вывод:
        Датасет.
    '''
    file = pd.ExcelFile(file_name)
    # тк название страницы может быть с пробелом
    found_sheet_name = list(set([area_name, area_name+' ']
                               ) & set(file.sheet_names))
    if found_sheet_name:
        # читаем нужную страницу
        df = pd.read_excel(file, sheet_name=found_sheet_name[0], 
                           header=[1]).rename(columns = {"Unnamed: 0":"пол"})
        return df
    #else:
       # print(f"Страница с районом '{area_name}' не найдена")

        
def df_from_csv(df, year=2023):
    '''
    Преобразовать датасет.
    Параметры:
        df -- датасет, откуда берем данные по населению;
        year -- год прогноза.
    Вывод:
        Датасет.
    '''
    df.set_index('Cohort', inplace=True)
    new_df = df.stack().reset_index()
    
    #year = file.name.split('.')[0][-4:]
    new_df.columns = ['группа', 'пол', year]
    new_df.set_index('группа', inplace=True)
    
    di = {'Female':'Женщины', 'Male':'Мужчины'}
    new_df['пол'].replace(di, inplace=True)
    
    # уберем лишние слова в индексах
    new_df.index = new_df.index.map(clean_indexes)
    return new_df

#______________________

        
def prepro_df(df, from_api=False, morrate_file='population_data/morrate.xlsx',
             aux_clm_file = 'population_data/correct_clm_sex.csv'):
    '''
    Предобработка датасета.
    Параметры:
        df -- файл, с которым будем работать;
        morrate_file -- excel-файл с коэф-ми смертности по возрастам;
        aux_file -- excel-файл, из которого можно взять правильную колонку пол.
    Вывод:
        Датасет
    '''
    if not from_api:
        # оставляем нужные строки
        df = select_essential_info(df, aux_clm_file)
        # заполняем пропущенные данные
        df = fill_missing_data(df, morrate_file)
        # уберем лишние слова в индексах
        df.index = df.index.map(clean_indexes)
        
    # добавим возраста с интервалом в 1 год для 70-100-летних
    df = add_ages_70_to_100(df)

    # уберем возрастные интервалы
    df = df[df.index.isin([str(i) for i in range(0,100)]+['100'])]
    df.index = df.index.astype(int)
    # изменим форму
    pivoted = pd.pivot_table(df, index='группа', columns='пол')

    return pivoted
        
        
def select_essential_info(df, aux_clm_file): 
    '''
    Выделение строк с женщинами и мужчинами; изменение индекса.
    Параметры:
        df -- датасет;
        aux_clm_file -- колонка с правильным порядком возрастов.
    Вывод:
        Датасет.
    '''
    # на странице Волосовского района есть лишние строки 
    if 'Муниципальный район' in df['пол'].values:
        # берем образцовую колонку Пол
        aux_df = pd.read_csv(aux_clm_file, index_col=0)
        df['пол'] = aux_df['пол']

    # не берем информацию по Всем
    df_sex = df[df['пол'].isin(['Женщины','Мужчины'])].copy()
    # ставим возрастную группу как индекс в таблице
    age_groups = np.repeat(['все возраста',*df.iloc[3::4,0].values],2)
    df_sex.loc[:,'группа'] = age_groups
    df = df_sex.set_index(['группа'])
    
    return df


def get_mor_rate_70(morrate_file):
    '''
    Получение коэффициентов смертности из файла для людей старше 70 лет.
    Параметры:
        morrate_file -- excel-файл с коэф-ми смертности по возрастам;
    Вывод:
        Коэффициенты для женщин и мужчин.
    '''
    
    #kmor = pd.read_csv(morrate_file, index_col=0)
    df = pd.read_excel(morrate_file, header=1, 
                         usecols=['Cohort','avg%','avg%.1'])
    df.columns=[df.columns[0],'avg% male','avg% female']
    
    return df[df.Cohort=='70-'].values[0,1:][::-1]


def fill_missing_data(df, morrate_file):
    '''
    Заполнение пропущенных значений (без 'до 30' и 'старше 65').
    Параметры:
        df -- файл, с которым будем работать;
        morrate_file -- excel-файл с коэф-ми смертности по возрастам;
    Вывод:
        Датасет.
    '''
    # берем группы, где пропущены значения
    nan_info = df[df.isna().any(axis=1)].index.unique()

    # если пропущены 17-летние -- их заполним в первую очередь
    if '17' in nan_info:
        q = df.loc['17'].copy()
        # года с пропущенными данными
        missing_info_years = q.loc[:, q.isna().any()].columns
        all_clmns = ['пол',*missing_info_years]
        r = (df.loc['16-17', all_clmns].set_index('пол') - \
            df.loc['16', all_clmns].set_index('пол').values)
        df.loc['17',missing_info_years] = r.values

    # берем только интервалы
    age_ranges = nan_info[np.where(nan_info.str.contains('-'))[0]]

    for age_group in age_ranges:
        # раскрываем строку интервала на 2 числа: начало и конец
        age_brackets = [int(i) for i in re.findall(r'\d+', age_group)]

        # тк после 70 лет нет данных для 71,72,... есть только интервалы
        if age_brackets[0]<70:
            q = df[(df.index==age_group)]
            # года с пропущенными данными
            missing_info_years = q.loc[:, q.isna().any()].columns
            # если такой группы вообще нет в списке
            if len(missing_info_years)<1:
                missing_info_years = df.columns[1:]

            all_clmns = ['пол',*missing_info_years]
            # суммируем все значения по каждому возрасту
            new = df[(df.index.isin([ str(i) for i in range(age_brackets[0], 
                                                                      age_brackets[1]+1) ]))][all_clmns]
            new = new.groupby('пол').sum()

            if age_group in df.index:
                df.loc[age_group, missing_info_years] = new.values
            # если такой группы вообще нет в списке -- добавляем 
            else:
                # сначала каждый пол со своим индексом
                women = new.iloc[0]
                men = new.iloc[1]
                df.loc[age_group+'f',missing_info_years] = women
                df.loc[age_group,missing_info_years] = men
                # меняем индексы на одинаковые
                df.rename(index={age_group+'f': age_group},inplace =True)
                df.loc[age_group, 'пол'] = ['Женщины','Мужчины']

        elif age_brackets[0]>=70:
            q = df.loc[age_group, ~df.columns.isin(['пол'])]
            # года с пропущенными данными
            missing_info_years = q.loc[:, q.isna().any()].columns

            # коэфф-ты смертности
            female_morrate, male_morrate = get_mor_rate_70(morrate_file)
            # умножаем данные прошлого года на коэфф-т смертности
            # int(...) -- округляем вниз
            for year in missing_info_years:
                # женщины на нулевой строке, мужчины на первой
                df.loc[age_group, year] = (df.loc[age_group, year-1] * 
                                                [female_morrate, male_morrate]).astype(int)
    return df


def add_ages_70_to_100(df):
    '''
    Добавление возрастов с интервалом в 1 для 70-100 лет.
    '''
    # берем интервалы возрастов после 70
    ages_brackets = ['70-74','75-79','80-84','85-89','90-94','95-99',100]
    years = df.columns.get_level_values(0).unique()#df.columns[1:].values
    
    for i in range(len(ages_brackets)-1):
        res_both_sex = []
        for sex in ['Женщины','Мужчины']:
            #print(ages_brackets[i], sex)
            #first_cohort = df[(df.index == ages_brackets[i]) & (df['пол']==sex)].values[0]
            first_cohort = df.loc[:,df.columns.get_level_values('пол')==sex
                  ][df.index == ages_brackets[i]].values[0]
            #next_cohort = df[(df.index == ages_brackets[i+1]) & (df['пол']==sex)].values[0]
            next_cohort = df.loc[:,df.columns.get_level_values('пол')==sex
                  ][df.index == ages_brackets[i+1]].values[0]
            # чем больше разница, тем сильнее перекошены наши вероятности
            probs = np.linspace(first_cohort, next_cohort, num=5)
            probs = probs/probs.sum(0)

            res = []
            for prob, pop_size in zip(range(probs.shape[1]),
                                      first_cohort):
                # раскидываем числа с учетом вероятностей
                q = np.random.multinomial(pop_size, probs[:,prob], size=1)[0]
                res.append(q)

            # составляем датафрейм и меняем названия колонок и индексов    
            res = pd.DataFrame(np.array(res).T, columns=years)    
            #res['пол'] = sex
            new_index = np.arange(int(ages_brackets[i][:2]), int(ages_brackets[i][3:5])+1)
            res.index = new_index.astype('str')
            
            res.columns= pd.MultiIndex.from_product([years, [sex]])
            res_both_sex.append(res)
            
        age_bracket_df = pd.concat(res_both_sex, axis=1)     
        df = pd.concat([df, age_bracket_df], axis=0)
        df.columns.set_names(['', "пол"], level=[0,1], inplace=True)
            # объединяем с изначальными данными
            #df = pd.concat([df, res], ignore_index=False, sort=True)
    
    
    # поставим название колонки индексов
    df.index.names = ['группа']
    # перенесем колонку "пол" в начало
    #cols = df.columns.tolist()
    #cols = cols[-1:] + cols[:-1]
    #df = df[cols]
    
    return df.round()