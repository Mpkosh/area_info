import requests
import pandas as pd
import geopandas as gpd
from app_package.src import PreproDF, DemForecast
import numpy as np
#from tqdm.notebook import tqdm
import os


social_api = os.environ.get('SOCIAL_API')
territories_api = os.environ.get('TERRITORY_API') 


def to_interval(x):
    x1, x2 = x.iloc[0], x.iloc[1]
    
    if x1 == 100:
        return 100
    elif x1 != x2:
        return f'{int(x1)}-{int(x2)}'
    else:
        return int(x1)

    
def prepro_from_api(df_from_json, given_years=[2019,2020], unpack_after_70=False):
    df_all = df_from_json[df_from_json.year.isin(given_years)]

    df_list = []
    for year in given_years:
        df = pd.json_normalize(df_all[df_all['year']==year]['data'].explode())

        # всякий случай задаем возраста: 
        # 1) все с интервалом 1 год; 2) 100+; 3) с интервалом 4 года для старших
        df = df[(df['age_start']==df['age_end'])|(
                df['age_start']==100)|(
                (df['age_end']-df['age_start']==4)&(
                    df['age_end'].isin([74,79,84,89,94,99])))]
        df['группа'] = df.iloc[:,:2].apply(to_interval, 1)
        df = df.set_index('группа').iloc[:,2:]

        # ставим года
        df.columns = pd.MultiIndex.from_product([[year], ['Мужчины', 'Женщины']])
        df.columns.set_names(['', "пол"], level=[0,1], inplace=True)
        df.bfill(inplace=True)
        df_list.append(df)

    df = pd.concat(df_list, axis='columns')
    
    if unpack_after_70:
        df = PreproDF.add_ages_70_to_100(df)
        df.index = df.index.astype(str)
        # уберем возрастные интервалы
        df = df[df.index.isin([str(i) for i in range(0,101)])]
        df.index = df.index.astype(int)
        df.sort_index(inplace=True)    
        #df.index = df.index.astype(str)
    else:    
        df.index = df.index.astype(str)

    return df


# --
def get_S_and_dnst_children(session, territory_id=34):
    # здесь для ЛО нет площади
    # по этому запросу есть нужная площадь без воды.
    # + есть плотность за посл. год, но посчитана по площади C водой, поэтому считаем сами
    url= territories_api + f'api/v2/territories_without_geometry?parent_id={territory_id}&get_all_levels=false&ordering=asc&size=1000'
    r = session.get(url)
    res = pd.DataFrame(r.json())
    res = pd.json_normalize(res['results'], max_level=1)
    
    res = res[['territory_id','properties.Площадь территории, кв. км.',
               'properties.Численность населения']]
    res = res.rename(columns={'properties.Площадь территории, кв. км.':'S_km2',
                             'properties.Численность населения':'population'})
    # считаем плотность населения
    res['population_dnst'] = res['population'].div(res['S_km2'], axis=0).round(2)
    return res


def children_pop_dnst(session, territory_id, pop_and_dnst=True):
    # здесь для ЛО нет площади
    # здесь для НП нет инф-ии о численности и площади
    # для ЛО и territory_type_id=3 показывает только Сосновоборский
    # для ЛО и territory_type_id=4 показывает []
    url=territories_api + f'api/v2/territories?parent_id={territory_id}&get_all_levels=false&ordering=asc&size=1000'
    r = session.get(url).json()
    if r['results']:
        res = pd.DataFrame(r)
        res = pd.json_normalize(res['results'], max_level=0)
        fin = res[['territory_id','name']].copy()
        with_geom = gpd.GeoDataFrame.from_features(r['results'])
        fin['geometry'] = with_geom['geometry']

        if pop_and_dnst:
            pop_clm = 'Численность населения'
            s_clm = 'Площадь территории, кв. км.'
            if (pop_clm in with_geom.columns)&(s_clm in with_geom.columns):
                pop_and_S = with_geom[['Численность населения',
                                       'Площадь территории, кв. км.']]
                fin['pop_all'] = pop_and_S['Численность населения']
                fin['density'] = round(pop_and_S['Численность населения']/pop_and_S['Площадь территории, кв. км.'], 2)
            else:
                fin['pop_all'] = 0
                fin['density'] = 0

        return fin


def children_pop_dnst_LO(session, territory_id):
    url= territories_api + f'api/v1/territory/indicator_values?parent_id={territory_id}&indicator_ids=1%2C4&last_only=True'
    r = session.get(url).json()
    with_geom = gpd.GeoDataFrame.from_features(r['features']).set_crs(epsg=4326)
    fin = with_geom[['territory_id','name','geometry']]

    qq = pd.json_normalize(with_geom['indicators'])
    qq.columns = ['pop','S']

    pop_info = pd.json_normalize(qq['pop'])['value']
    pop_info.name = 'pop_all'
    S_info = pd.json_normalize(qq['S']).fillna(0)['value']
    # если где-то не указана площадь -- считаем сами
    zero_s_idx = S_info[S_info==0].index
    S_info.loc[zero_s_idx] = fin.loc[zero_s_idx, 
                                     'geometry'].to_crs(epsg=6933).area/10**6
    
    density = round(pop_info/S_info,2)
    density.name = 'density'
    ff = pd.concat([fin,pop_info,density], axis=1)

    return ff


def last_pop_and_dnst(session, territory_id, dnst=False, both=False):
    # площадь без воды и численность
    url= territories_api + f'api/v1/territory/{territory_id}/indicator_values?indicator_ids=4%2C%201&last_only=True'
    r = session.get(url)
    r = pd.DataFrame(r.json())
    indicators = pd.json_normalize(r['indicator'])

    #pop_ind = indicators[indicators['name_full']=='Численность населения'].index
    #pop_value = r.iloc[pop_ind].sort_values('date_value', ascending=False).iloc[0]['value']
    pop_ind = indicators[indicators['name_full']=='Численность населения'].index[0]
    pop_value = r.iloc[pop_ind]['value']
    
    if dnst:
        S_ind = indicators[indicators['name_full']=='Площадь территории'].index[0]
        S_value = r.iloc[S_ind]['value']
        dnst = round(pop_value/S_value, 2) 
        if both:
            return pop_value, dnst 
        else:
            return dnst
    else:
        return pop_value
    

def get_detailed_pop(session, territory_id, unpack_after_70, last_year=True):
    url = social_api + f'indicators/2/{territory_id}/detailed'
    r = session.get(url)
    if r.status_code == 200:
        df_from_json = pd.DataFrame(r.json())
        if last_year:
            given_years = [df_from_json['year'].max()]
        else:
            given_years = df_from_json.year.sort_values().values
        df = prepro_from_api(df_from_json, 
                             given_years = given_years, 
                             unpack_after_70=unpack_after_70)
    else:
        df = pd.DataFrame()
        
    return df


def groups_3(x):
    pop_all = x.sum().sum()
    pop_younger = x.iloc[:16].sum().sum()
    pop_can_work= x.iloc[16:61].sum().sum()
    pop_older = x.iloc[61:].sum().sum()
    
    return [pop_all, pop_younger, pop_can_work, pop_older]


def get_second_children(session, first_children, child_level):
    all_second_children = gpd.GeoDataFrame()
    # для каждого ребенка берем вторых детей
    for main_id in first_children.territory_id.values:
        if child_level < 4:
            second_children = children_pop_dnst(session, main_id, pop_and_dnst=True)
        else:
            second_children = children_pop_dnst(session, main_id, pop_and_dnst=False)
        #second_children['main_id'] = main_id
        all_second_children = pd.concat([all_second_children,
                                         gpd.GeoDataFrame(second_children)])
    
    return all_second_children


def fill_in_by_34(session, territory_id, geom):
    if geom['pop_all'].sum == 0:
        geom['pop_all'] = last_pop_and_dnst(session=session,
                                            territory_id=territory_id,
                                            dnst=False)
    # раскидываем числа с учетом вероятностей Всеволожского
    np.random.seed(27) 
    geom[['pop_younger','pop_can_work','pop_older']
        ] = geom['pop_all'].apply(
        lambda x: np.random.multinomial(x,[0.15,0.7,0.15])).to_list()
    
    return geom


def barebones_for_info(territory_id, session, show_level):
    # ____ Узнаем уровень территории
    url= territories_api + f'api/v1/territory/{territory_id}'
    r_main = session.get(url).json()
    territory_type = r_main['territory_type']['territory_type_id']

    # численность будем брать не здесь (r_main), тк в .../api/v1/territory/{territory_id} 
    # нет привязки численности к дате, лучше уверенно взять самую новую инф-ию
    # ____ Делаем костяк нашего geojson'а
    # если нужен уровень детальнее
    if show_level > territory_type:    
        # собираем данные первых детей
        # для ЛО у детей нет площади у 193
        if territory_type==1:
            geom = children_pop_dnst_LO(session, territory_id)
        elif (show_level-territory_type==1)&(show_level==4):
            # для НП (show_level=4) нет инф-ии о численности и площади
            geom = children_pop_dnst(session, territory_id, pop_and_dnst=False)
        else:
            # отдаем геометрию детей с плотностью и численностью
            geom = children_pop_dnst(session, territory_id)

        # если разница в два, то дополнительно берем вторых детей
        if show_level - territory_type >= 2:
            geom = get_second_children(session, geom,  territory_type+2)
            # если разница в три, то дополнительно берем третих детей
            if show_level - territory_type == 3:
                geom = get_second_children(session, geom,  territory_type+3)
            
        # для НП (show_level=4) нет инф-ии о численности и площади
        if show_level==4:
            # поэтому восполняем тем, что есть; на всякий случай сортируем
            # print('Заполнение колонки pop_all с файла towns.geojson')
            towns = gpd.read_file('app_package/src/towns.geojson').sort_values(by='territory_id')
            geom = geom.sort_values(by='territory_id')
            geom['pop_all'] = towns[towns.territory_id.isin(geom.territory_id)]['population'].values
            geom['pop_all'] = geom['pop_all'].fillna(0)
    
    # если НЕ нужен уровень детальнее
    elif show_level == territory_type:
        # отсюда берем только геометрию
        geom = gpd.GeoDataFrame.from_features([r_main])[['geometry']]
        geom['territory_id'] = territory_id
        geom['name'] = r_main['name']
        
        if show_level<=2:
            # (38 'Рахьинское ГП') здесь НЕ указана площадь; в ф-ии dnst=True, both=True
            # для territory_type=2 площадь отсюда:
            geom[['pop_all','density']] = last_pop_and_dnst(session=session, territory_id=territory_id,
                                                            dnst=True, both=True)
        else:
            geom['pop_all'] = last_pop_and_dnst(session=session, territory_id=territory_id,
                                                dnst=False)
            # (34 'Всев.район') геометрия с водой, т.е. площадь не посчитать; площади и плотности нет
            # (38 'Рахьинское ГП') здесь указана площадь, плотность неправильная
            # для territory_type=3 площадь отсюда:
            S_km2 = r_main['properties']['Площадь территории, кв. км.']
            geom['density'] = round(geom['pop_all']/S_km2, 2)
              
    # если просят показать уровень выше заданной тер-рии -- ошибка
    # (Ex: передали Романовское СП, а просят показать уровень Всев.района)
    else:
        return '!Stop the madness! Show_level should be less or equal to territory level'
       
    return r_main, territory_type, geom


def main_pop_info(territory_id=34, show_level=3):
    '''
    Основные характеристики населения:

    Параметры:
     - territory_id: id территории
     - show_level: в каком делении выводить данные (territory_type_id в API)
         (1 - регион, 2 - МО (Всев.район), 3 - Поселение (ГП и СП), 4 - населенные пункты)
    
    Вывод: geojson
    - geometry: геометрия территории
    - pop_all: численность общая
      pop_younger: 0-15 лет включительно
      pop_can_work: 16-60 лет включительно
      pop_older: 61-100 лет
    - демографические показатели (коэфф-ты смертности, рожд-ти, миграции)
    - density: плотность населения.
    
    '''
    session = requests.Session()
    r_main, territory_type, geom = barebones_for_info(territory_id, session, show_level)
    
    # ____ Наполняем костяк данными
    # Если уровень позволяет, берем площадь и половозр.пирамиду
    if (show_level <= 2) & (territory_type == show_level):
        # численность тоже отсюда, тк в .../api/v1/territory/{territory_id} 
        # нет привязки численности к дате, лучше уверенно взять самую новую инф-ию
        pop_df = get_detailed_pop(session, territory_id, False)
        if pop_df.shape[0]:
            geom[['pop_all','pop_younger','pop_can_work','pop_older']] = groups_3(pop_df)

        # Если половозрастные данные ожидались, но не получились -- наполняем так    
        else:
            # print('Половозрастные данные не найдены; берем вероятности Всеволожского района')
            geom = fill_in_by_34(session, territory_id, geom)


    #elif show_level-1 <= 2:
       
    # НО! если заданный родитель с возрастами ИЛИ у заданной территории есть родитель с возрастами,
    # то заполним инфу для детей на основе родителя

    # если уровень территории имеет родителя (поселение или город)
    elif territory_type-1 <= 2:
         # если уровень родителя позволяет (поселение или город)
        if territory_type<=2:
            id_of_interest = territory_id
        #elif territory_type
        # если уровень территории имеет родителя (поселение или город)
        else:
            id_of_interest = r_main['parent']['id']
        
        # print(f'Разделение по возрастам от родителя с id {id_of_interest}')
        pop_df = get_detailed_pop(session, id_of_interest, False)
        
        if pop_df.shape[0]:
            all_pop ,young, can_work, old = groups_3(pop_df)
            # доля возрастных групп относительно всей популяции
            young_fr, can_work_fr, old_fr = np.array([young, can_work, old])/all_pop
            # раскидываем числа с учетом вероятностей
            np.random.seed(27) 
            geom[['pop_younger','pop_can_work','pop_older']
                ] = geom['pop_all'].apply(
                lambda x: np.random.multinomial(x,
                                                [young_fr,can_work_fr,old_fr])).to_list()

        # Если половозрастные данные ожидались, но не получились -- наполняем так    
        else:
            # print('Половозрастные данные не найдены; берем вероятности Всеволожского района')
            geom = fill_in_by_34(session, territory_id, geom)
     
    # иначе довольствуемся общей численностью и площадью с основного запроса  
    # а мы и так взяли эту инф-ию с children_pop_dnst
    
    # ____ Если данных колонок нет, то добавляем и ставим нули
    cols = ['density','pop_all','pop_younger','pop_can_work','pop_older',
            'koeff_death','koeff_birth','koeff_migration']
    geom = geom.reindex(geom.columns.union(cols, sort=False), axis=1, fill_value=0)
    geom[['pop_all','pop_younger','pop_can_work','pop_older']] = \
        geom[['pop_all','pop_younger','pop_can_work','pop_older']].astype(int)
    cols_order = ['territory_id','name','geometry']+cols
    return geom[cols_order]


def detailed_pop_info(territory_id=34):
    session = requests.Session()
    
    # ____ Половозрастная структура
    pop_df = get_detailed_pop(session, territory_id, True, False)
    if pop_df.shape[0]:
        pass
    
    # ____ Численность всех групп
    by_work_groups = ['0-16','16-61','61-101']
    soc_groups=['0-6','6-11','11-15','15-18','18-30',
                '30-40','40-60','60-75','75-101']
    r = []
    soc_pyramid = []
    for age_group, name in zip(by_work_groups,
                               ['pop_younger','pop_can_work','pop_older']):
        start = int(age_group.split('-')[0])
        finish = int(age_group.split('-')[1])
        res = pd.Series(pop_df.iloc[start:finish].T.groupby(level=[0]
                                                           ).sum().sum(1), 
                        name=name)
        r.append(res)
        
    for age_group in soc_groups:
        start = int(age_group.split('-')[0])
        finish = int(age_group.split('-')[1])
        soc_pyr_part = pop_df.iloc[start:finish]
        soc_pyramid.append(soc_pyr_part) # для пункта 4
        res = pd.Series(soc_pyr_part.T.groupby(level=[0]).sum().sum(1), 
                        name=age_group)
        r.append(res)  
        
    groups_df = pd.concat(r, axis=1).T
    
    # ____ Динамика и прогноз
    folders={'popdir':'app_package/src/population_data/',
             'file_name':'Ленинградская область.xlsx'}
    last_pop_year = pop_df.columns.levels[0][-1]
    forecast = DemForecast.MakeForecast(pop_df, last_pop_year, 
                                        1, folders)
    dynamic_pop = pd.concat([pop_df,forecast.loc[:, (2024, slice(None))]], axis=1)
    dynamic_pop_df = pd.DataFrame([dynamic_pop.T.groupby(level=[0]).sum().sum(1)])

    # ____ Половозрастная структура соц.групп (то же, что и пункт 1?)
    soc_pyramid_df = pd.concat(dict(zip(soc_groups, soc_pyramid)), names=['Соц_группа']) 
    
    # ____ Ценности
    pop_years = pop_df.columns.levels[0]
    values_df = pd.DataFrame(np.zeros_like(0, shape=(9, pop_years.shape[0])))
    values_df.columns=pop_years
    values_df
    
    return pop_df, groups_df, dynamic_pop_df, soc_pyramid_df, values_df