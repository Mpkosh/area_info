import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import os
from app_package.src import PreproDF, DemForecast


social_api = os.environ.get('SOCIAL_API')
terr_api = os.environ.get('TERRITORY_API')
file_path = 'app_package/src/'


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


class Territory():
    
    def __init__(self, territory_id=34):
        self.territory_id = territory_id
        self.df = pd.DataFrame([])
        self.df['territory_id'] = [self.territory_id]
        self.children = []
        self.parent = 0
        

def create_point(x):
    return Point(x['coordinates']) 

        
def info(territory_id=34, show_level=3):
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
    current_territory = Territory(territory_id)
    main_info(session, current_territory, show_level)
    
    if show_level < current_territory.territory_type:
        raise ValueError(f'Show level (given: {show_level}) must be >= territory type (given: {current_territory.territory_type})')
        
    n_children = show_level - current_territory.territory_type
    terr_classes = [current_territory]
    terr_ids = current_territory.df['territory_id'].values
    # если нужен уровень детальнее
    for i in range(n_children):
        for ter_id, ter_class in zip(terr_ids, terr_classes):
            get_children(session, ter_id, ter_class)
        # новый набор для итерации    
        terr_classes = [child for one_class in terr_classes for child in one_class.children]
        # новые id тер-рий
        terr_ids = [one_class.territory_id for one_class in terr_classes]    
    
    # all final children in <terr_classes>    
    fin_df = pd.DataFrame([])
    fin_df['territory_id'] = [cl.territory_id for cl in terr_classes]
    fin_df['oktmo'] = [cl.oktmo for cl in terr_classes]
    fin_df['name'] = [cl.name for cl in terr_classes]
    fin_df['geometry'] = [cl.geometry for cl in terr_classes]
    
    fin_df[['pop_all','density']] = [0,0]
    fin_df['pop_all'] = [cl.pop_all for cl in terr_classes]
    fin_df['density'] = [cl.density for cl in terr_classes]
    
    change = True
    if show_level==4:
        # восполняем тем, что есть; на всякий случай сортируем
        # print('Заполнение колонки pop_all с файла towns.geojson')
        towns = gpd.read_file(file_path+'towns.geojson')
        towns = towns.set_index('territory_id').loc[fin_df.territory_id].reset_index()
        fin_df['pop_all'] = towns[towns.territory_id.isin(fin_df.territory_id)]['population'].values
        fin_df['pop_all'] = fin_df['pop_all'].fillna(0)
        
        for child in terr_classes:
            child.pop_all = fin_df[fin_df.territory_id==child.territory_id]['pop_all'].values[0]
            change = False
            #print(child.territory_id, child.name, pop_for_children.loc[child.territory_id])
    
    pyramid_info(session, terr_classes)
    if change:
        fin_df['pop_all'] = [cl.pop_all for cl in terr_classes]
        
    fin_df['pop_younger'] = [cl.pop_younger for cl in terr_classes]
    fin_df['pop_can_work'] = [cl.pop_can_work for cl in terr_classes]
    fin_df['pop_older'] = [cl.pop_older for cl in terr_classes]
    
    # у ЛО нет октмо в БД
    with pd.option_context("future.no_silent_downcasting", True):
        fin_df['oktmo'] = fin_df['oktmo'].fillna(0)
    
    # ____ Если данных колонок нет, то добавляем и ставим нули
    cols = ['density','pop_all','pop_younger','pop_can_work','pop_older',
            'koeff_death','koeff_birth','koeff_migration']
    fin_df = fin_df.reindex(fin_df.columns.union(cols, sort=False), axis=1, fill_value=0)
    fin_df[['pop_all','pop_younger','pop_can_work','pop_older']] = \
        fin_df[['pop_all','pop_younger','pop_can_work','pop_older']].astype(int)
    cols_order = ['territory_id','name','geometry']+cols
    return fin_df[cols_order].sort_values('territory_id')
    
    
def main_info(session, current_territory, show_level):
    # ____ Узнаем уровень территории
    try:
        url = terr_api + f'api/v1/territory/{current_territory.territory_id}'
        r_main = session.get(url).json()
        current_territory.territory_type = r_main['territory_type']['territory_type_id']
        current_territory.name = r_main['name']
        current_territory.oktmo = r_main['oktmo_code']
        geom_data = gpd.GeoDataFrame.from_features([r_main])[['geometry']]
    except:
        raise requests.exceptions.RequestException(f'Problem with {url}')

    current_territory.geometry = geom_data.values[0][0]
    current_territory.parent = Territory(r_main['parent']['id'])
    
    if show_level == current_territory.territory_type:
        current_territory.df['oktmo'] = current_territory.oktmo
        current_territory.df['geometry'] = geom_data
        current_territory.df['name'] = current_territory.name
        
        last_pop_and_dnst(session, current_territory, dnst=True, both=True)
        

def pyramid_info(session, terr_classes):
    for child in terr_classes:
        chosen_class = child
        # если у ребенка не может быть пирамиды, то постепенно проверяем его родителя
        if child.territory_type <= 2:
            ter_id_for_pyramid = child.territory_id
        else:
            for i in range(child.territory_type-2):
                ter_id_for_pyramid = chosen_class.parent.territory_id
                chosen_class = child.parent
        
        # заглушка для ЛО        
        if child.territory_type == 1:
            ter_id_for_pyramid = 34
            
        #print(f'pyramid from {chosen_class.name}')        
        pop_df = get_detailed_pop(session, ter_id_for_pyramid, False)
        
        if pop_df.shape[0]:
            p_all, p_y, p_w, p_o = groups_3(pop_df)
            # если брали пирамиду самой территории
            if ter_id_for_pyramid == child.territory_id:
                child.pop_all, child.pop_younger,\
                    child.pop_can_work,child.pop_older = p_all, p_y, p_w, p_o

            # если это пирамида родителя, то раскидываем по вероятностям
            else:
                parent_data = np.array([p_all, p_y, p_w, p_o])
                probs = parent_data/parent_data.max()
                
                np.random.seed(27) 
                child.pop_younger, child.pop_can_work, \
                    child.pop_older = np.random.multinomial(child.pop_all, probs[1:])
        else:
            child.pop_younger,child.pop_can_work,child.pop_older = [0,0,0]
            
        
def child_to_class(x, parent_class):
    child = Territory(x['territory_id'])
    child.name = x['name']
    child.oktmo = x['oktmo_code']
    child.df['name'] = child.name
    child.geometry = x['geometry']
    child.territory_type = parent_class.territory_type+1
    
    if 'pop_all' in x.index:
        child.pop_all = x['pop_all']
    else:
        child.pop_all = 0
        
    if 'density' in x.index:
        child.density = x['density']
    else:
        child.density = 0
        
    child.parent = parent_class
    parent_class.children.append(child)        

    
def children_pop_dnst(session, parent_class, pop_and_dnst=True):
    # здесь для ЛО нет площади
    # здесь для НП нет инф-ии о численности и площади
    # для ЛО и territory_type_id=3 показывает только Сосновоборский
    # для ЛО и territory_type_id=4 показывает []
    try:
        territory_id = parent_class.territory_id
        url = terr_api + 'api/v2/territories'
        params = {'parent_id':territory_id,'get_all_levels':'false',
                  'cities_only':'false','page_size':'1000'}
        r = session.get(url, params=params)
        res = pd.DataFrame(r.json()) 
        res = pd.json_normalize(res['results'], max_level=0)
        fin = res[['territory_id','name','oktmo_code']].copy()
        
        children_type = parent_class.territory_type+1
        if children_type <= 3:
            with_geom = gpd.GeoDataFrame.from_features(r['results'])
            fin['geometry'] = with_geom['geometry']
        else:
            fin.loc[:,'geometry'] = res['centre_point'].apply(lambda x: create_point(x))
            fin = fin.set_geometry('geometry').set_crs(epsg=4326)
            
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
        
    if pop_and_dnst:
        pop_clm = 'Численность населения'
        s_clm = 'Площадь территории, кв. км.'
        if (pop_clm in with_geom.columns)&(s_clm in with_geom.columns):
            pop_and_S = with_geom[['Численность населения',
                                   'Площадь территории, кв. км.']]
            fin['pop_all'] = pop_and_S['Численность населения']
            fin['density'] = round(pop_and_S['Численность населения'
                                             ]/pop_and_S['Площадь территории, кв. км.'], 2)
    else:
        fin['pop_all'] = 0
        fin['density'] = 0

    return fin
    
    return pd.DataFrame([])


def children_pop_dnst_LO(session, parent_class):
    # но здесь нет октмо 
    try:
        territory_id = parent_class.territory_id
        url= terr_api + 'api/v1/territory/indicator_values'
        params = {'parent_id':territory_id,'indicator_ids':'1,4','last_only':'true'}
        r = session.get(url, params=params)
        with_geom = gpd.GeoDataFrame.from_features(r.json()['features']).set_crs(epsg=4326)
        fin = with_geom[['territory_id','name','geometry']]
        fin['oktmo_code'] = 0
    
        qq = pd.json_normalize(with_geom['indicators'])
        qq.columns = ['pop','S']
    
        pop_info = pd.json_normalize(qq['pop'])['value']
        pop_info.name = 'pop_all'
        
        S_info = pd.json_normalize(qq['S']).fillna(0)['value']
        # если где-то не указана площадь -- считаем сами
        zero_s_idx = S_info[S_info==0].index
        S_info.loc[zero_s_idx] = fin.loc[zero_s_idx, 
                                         'geometry'].to_crs(epsg=6933).area/10**6
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
        
    density = round(pop_info/S_info,2)
    density.name = 'density'
    ff = pd.concat([fin,pop_info,density], axis=1)

    return ff    
    
    
def get_children(session, parent_id, parent_class):

    if parent_class.territory_type == 1:
        # для ЛО у детей нет площади у 193
        fin = children_pop_dnst_LO(session, parent_class)
    elif parent_class.territory_type == 3:
        # для НП (show_level=4) нет инф-ии о численности и площади
        fin = children_pop_dnst(session, parent_class, pop_and_dnst=False)
    else:
        fin = children_pop_dnst(session, parent_class, pop_and_dnst=True)
    if fin.shape[0]:       
        fin.apply(child_to_class, parent_class=parent_class, axis=1)
    
        
        
def last_pop_and_dnst(session, current_territory, dnst=False, both=False):
    # площадь без воды и численность
    try:
        url= terr_api + f'api/v1/territory/{current_territory.territory_id}/indicator_values'
        params = {'indicator_ids':'1,4','last_only':'true'}
        r = session.get(url, params=params)
        res = pd.DataFrame(r.json())
        indicators = pd.json_normalize(res['indicator'])
        
        pop_ind = indicators[indicators['name_full']=='Численность населения'].index[0]
        pop_value = res.iloc[pop_ind]['value']
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
    
    if dnst:
        try:
            S_ind = indicators[indicators['name_full']=='Площадь территории'].index[0]
            S_value = res.iloc[S_ind]['value']
        except IndexError:
            geom = gpd.GeoSeries(current_territory.geometry).set_crs(epsg=4326)
            S_value = (geom.to_crs(epsg=6933).area/10**6).values[0]
        
        dnst = round(pop_value/S_value, 2) 
        if both:
            current_territory.pop_all = pop_value
            current_territory.density = dnst
        else:
            current_territory.density = dnst
    else:
        current_territory.pop_all = pop_value
        
        
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
    folders={'popdir':file_path+'population_data/',
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