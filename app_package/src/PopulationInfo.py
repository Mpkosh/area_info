import pandas as pd
import numpy as np
from io import BytesIO

import matplotlib.ticker as mticker
import matplotlib
matplotlib.use('AGG') # must be imported before pyplot
import matplotlib.pyplot as plt

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
                groups += ['100']
        # если человек перечислил новые интервалы   
        if isinstance(n_in_age_group, list): 
            groups = n_in_age_group

        result = []
        indexes = []
        for age_group in groups:
            # если задан интервал    
            if '-' in age_group:
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
                else:
                    print(f'Возраст {age_group} в данных нет.')

            indexes.append(age_group)
            result.append(new)

        new_df = pd.concat(result, axis=1).T
        new_df.index = indexes
        chosen_age_groups = new_df

    chosen_age_groups.index.name = 'группа' 
    
    return chosen_age_groups

    
def plot_population_info(age_groups_df, chosen_years='all', 
                         area_name='Кингисеппский', figsize=(10,13)):
    '''
    График возрастно-половой пирамиды.
    Параметры:
        age_groups_df -- датасет;
        chosen_years -- данные за какой год отображать;
        area_name -- район;
        figsize -- размер графика (x,y).
    Вывод:
        График.
    '''
    plt.figure(figsize=figsize)
    # выводим значения на осях на первый план
    plt.rcParams["axes.axisbelow"] = False
    
    female_str_in_df='Женщины'
    male_str_in_df='Мужчины'
    
    label_size = 12
    title_size = 14
    tick_size = 12
    color_list = ['red','orange','green','blue','cyan','magenta',
                  'gray','orchid','lime','teal','black']
    
    if chosen_years == 'all':
        # не берем колонку Пол
        chosen_years = age_groups_df.columns.levels[0]
        
    for year in chosen_years:
        k_temp = age_groups_df[year].copy()
        # чтобы мужчины были слева графика
        k_temp[male_str_in_df] *= -1

        alpha=0.4
        alpha_fill=0.15
        color = color_list.pop(0)
        
        plt.plot(k_temp[female_str_in_df], k_temp.index, ls='-', marker='.', 
                 color=color, alpha=alpha,lw=2, label=str(year)+' г.')
        plt.fill_betweenx(y=k_temp.index, x1=k_temp[female_str_in_df], 
                          facecolor=color, alpha=alpha_fill)
        
        plt.plot(k_temp[male_str_in_df], k_temp.index, ls='-', marker='.', 
                 color=color, alpha=alpha,lw=2)
        plt.fill_betweenx(y=k_temp.index, x1=k_temp[male_str_in_df], 
                          facecolor=color, alpha=alpha_fill)
    
    plt.legend(loc='best', prop={'size': 12})
    plt.ylabel('Возраст (лет)', fontsize=label_size)
    plt.xlabel('Количество (чел.)', fontsize=label_size)
    plt.title(f'Возрастно-половая структура населения района {area_name}',
             fontsize=title_size)
    # перенесем yaxis в середину графика
    plt.gca().spines['left'].set_position(('data', 0.0))
    plt.gca().spines['bottom'].set_position(('data', 0.0))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # меняем разделитель тысяч у xticks
    tick_locs, tick_labels = plt.xticks()
    plt.xticks(tick_locs, 
               [f'{i:,.0f}' for i in abs(tick_locs).astype(int)],  
               rotation=0, size=tick_size)
    # два названия на xlabel
    plt.gca().xaxis.set_minor_locator(mticker.FixedLocator((k_temp[male_str_in_df].mean(), 
                                                            k_temp[female_str_in_df].mean())))
    plt.gca().xaxis.set_minor_formatter(mticker.FixedFormatter((male_str_in_df, 
                                                                female_str_in_df)))
    plt.setp(plt.gca().xaxis.get_minorticklabels(), rotation=0, size=label_size, va="center")
    # bottom=False -- отключаем minor xticks с названием пола
    plt.gca().tick_params("x", which="minor", pad=35, left=False, bottom=False)
    
    bbox = dict(boxstyle="round", ec="gray", fc="white", alpha=0.7)
    plt.setp(plt.gca().get_yticklabels(), bbox=bbox)
    
    plt.grid()
    plt.ylim(0)
    plt.margins(0.05, 0.05)
    plt.tight_layout()
    # create a buffer to store image data
    bytes_image = BytesIO() # BytesIO stream containing the data
    plt.savefig(bytes_image, format='png')
    #plt.close(fig)
    bytes_image.seek(0)
    return bytes_image


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


def plot_difference_info(df, area_name = 'Кингисеппский',
                         chosen_year=2023, figsize=(16,8)):
    '''
    График разности ожидаемых и реальных значений числа людей.
    Параметры:
        df -- датасет с сальдо, сгруппированный по возрастам;
        area_name -- район;
        chosen_year -- данные за какой год отображать; 
                        (будет сравниваться с предыдущим)
        figsize -- размер графика (x,y).
    Вывод:
        График.
    '''

    required_part = df.loc[:,chosen_year]

    plt.figure(figsize = figsize)
    female_str_in_df = 'Женщины'
    male_str_in_df = 'Мужчины'
    label_size = 15
    title_size = 16
    tick_size = 14
    
    year = chosen_year
    alpha = 0.9
    width = 0.35 
    # количество столбиков
    N = required_part.index.nunique()
    # координаты столбиков на оси Х
    ind = np.arange(N)

    women = required_part.loc[:,female_str_in_df]
    men = required_part.loc[:,male_str_in_df]
    bars = plt.bar(height=women, x=ind, width=width, 
                   label=female_str_in_df, alpha=alpha, color='salmon')

    for bars in plt.gca().containers:
        plt.gca().bar_label(bars, color='brown', size=12)

    bars = plt.bar(height=men, x=ind+width, width=width, 
                   label=male_str_in_df, alpha=alpha, color='cornflowerblue')

    for bars in plt.gca().containers[1::2]:
        plt.gca().bar_label(bars, color='navy', size=12)

    plt.xticks(ind + width / 2, required_part.index.unique(),
               rotation=25, ha='center')
    plt.grid()
    plt.legend(loc='best', prop={'size': 14})
    plt.xlabel('Возрастной интервал', fontsize=label_size)
    plt.ylabel('Количество (чел.)', fontsize=label_size)
    plt.tight_layout()
    plt.title(f'Разница ожидаемого (от {year-1} г.) и реального ({year} г.) населения в районе {area_name}', 
              fontsize=title_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.axhline(0, color='red')
    plt.margins(0.02, 0.1)
    plt.tight_layout()
    # create a buffer to store image data
    bytes_image = BytesIO() # BytesIO stream containing the data
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image
   