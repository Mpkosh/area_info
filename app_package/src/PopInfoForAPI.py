import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, shape
import os
from app_package.src import PreproDF, DemForecast, AreaOnMapFile


social_api = os.environ.get('SOCIAL_API')
terr_api = os.environ.get('TERRITORY_API')
file_path = 'app_package/src/'


def estimate_child_pyr(session, territory_id, unpack_after_70,
                       last_year, given_year):
    # получаем численность нашей территории по годам
    try:
        url = terr_api + f'api/v1/territory/{territory_id}/indicator_values'
        params = {'indicator_ids':'1','last_only':'false'}
        r = session.get(url, params=params)
        child_pops = pd.DataFrame(r.json())[['date_value','value']]
        
        child_pops['date_value'] = child_pops['date_value'].str[:4].astype(int)
        child_pops = child_pops.set_index('date_value')
    except:
        raise requests.exceptions.RequestException(f'Problem with {url}')
        
    # получаем id родителя
    try:
        url = terr_api + f'api/v1/territory/{territory_id}'
        r_main = session.get(url).json()
        parent_id = r_main['parent']['id']
        level = r_main['level']
        print('LEVEL:', level, parent_id)
        
        # по идее level-3, тк у уровня 3 уже есть пирамида
        # но мы уже сделали один поиск родителя выше
        #to_available = level-4
        #print(to_available)
        df = get_detailed_pop(session, parent_id, 
                              unpack_after_70=unpack_after_70, 
                              last_year=last_year, 
                              specific_year=given_year)
        
        uu = df.columns.get_level_values(0).nunique()

        # если в БД нет данных по пирамиде за минимум 2 года
        while uu < 2:
            url = terr_api + f'api/v1/territory/{parent_id}'
            r_main = session.get(url).json()
            parent_id = r_main['parent']['id']
            
            df = get_detailed_pop(session, parent_id, 
                                  unpack_after_70=unpack_after_70, 
                                  last_year=last_year, 
                                  specific_year=given_year)
            # если в БД нет данных по пирамиде за минимум 2 года
            uu = df.columns.get_level_values(0).nunique()
            print(f'for parent {parent_id} available years = {uu}')
            
            
           
            
    except:
        raise requests.exceptions.RequestException(f'Problem with {url}')
    
    print('FINAL PARENT', parent_id)
    # получаем половозрастную пирамиду родителя
    df = get_detailed_pop(session, parent_id, unpack_after_70=unpack_after_70, 
                          last_year=last_year, specific_year=given_year)
    # данные за какие года?
    df_years = df.columns.get_level_values(0).unique()
    # суммарное население по годам
    df_all_pop = df.abs().T.groupby(level=0).sum(0).sum(1)
    
    for year in df_years:
        # вычисляем долю
        parent_pyr_fracs = df[year] / df_all_pop[year]
        
        # если у ребенка данные начинаются позже; копируем первый год
        if year < child_pops.index[0]:
            child_pop = child_pops.iloc[0]
        # если у ребенка данные заканчиваются раньше; копируем посл.год
        elif year > child_pops.index[-1]:
            child_pop = child_pops.iloc[-1]    
        else:
            child_pop = child_pops.loc[year]
        
        # распределяем население ребенка по долям родительской пирамиды
        np.random.seed(27) 
        child_pyr =np.random.multinomial(child_pop, 
                                         parent_pyr_fracs.values.flatten())
        # меняем размер: мужчины -- [:,0], женщины [:,1] 
        # переприсваем в род.пирамиду
        df[year] = child_pyr.reshape(-1,2)
    
        print(year, df[year].sum().sum(),child_pop)
        
    return df


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
        # отсеиваем случайные данные
        if (year != 0):# and (year!=2024):
            df = pd.json_normalize(df_all[df_all['year']==year]['data'].explode())
            # 101 если раскрыты возраста
            if df.shape[0] > 99:
                
                unpack_after_70 = False
                df = df[(df['age_start']==df['age_end'])|(
                        (df['age_start']==100))]
            else:
                unpack_after_70 = True
                # всякий случай задаем возраста: 
                # 1) все с интервалом 1 год; 2) 100+; 3) с интервалом 4 года для старших
                '''
                df = df[(df['age_start']==df['age_end'])|(
                        df['age_start']==100)|(
                        (df['age_end']-df['age_start']==4)&(
                            df['age_end'].isin([74,79,84,89,94,99])))]
                '''            
  
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

        
def info(territory_id=34, show_level=3, down_by=0, specific_year=2022):
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
    fill_in_4 = True
    session = requests.Session()
    current_territory = Territory(territory_id)
    main_info(session, current_territory, down_by)

    if down_by < 0 :
        raise ValueError(f'Show level must be positive; given down_by={down_by}')
        
    n_children = down_by
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
    #fin_df['oktmo'] = [cl.oktmo for cl in terr_classes]
    fin_df['name'] = [cl.name for cl in terr_classes]
    fin_df['geometry'] = [cl.geometry for cl in terr_classes]
    
    fin_df[['pop_all','density']] = [0,0]
    fin_df['pop_all'] = [cl.pop_all for cl in terr_classes]
    fin_df['density'] = [cl.density for cl in terr_classes]
    
    change = True
    if (current_territory.level==5)&(fill_in_4):
        # для НП восполняем тем, что есть; на всякий случай сортируем
        # print('Заполнение колонки pop_all с файла towns.geojson')
        try:
            towns = gpd.read_file(file_path+'towns.geojson')
            towns = towns.set_index('territory_id').loc[fin_df.territory_id].reset_index()
            fin_df['pop_all'] = towns[towns.territory_id.isin(fin_df.territory_id)]['population'].values
            fin_df['pop_all'] = fin_df['pop_all'].fillna(0)
            
            for child in terr_classes:
                child.pop_all = fin_df[fin_df.territory_id==child.territory_id]['pop_all'].values[0]
                change = False
            #print(child.territory_id, child.name, pop_for_children.loc[child.territory_id])
        except:
            pass
        
    pyramid_info(session, terr_classes, specific_year)
    
    if change:
        fin_df['pop_all'] = [cl.pop_all for cl in terr_classes]
        
    fin_df['pop_younger'] = [cl.pop_younger for cl in terr_classes]
    fin_df['pop_can_work'] = [cl.pop_can_work for cl in terr_classes]
    fin_df['pop_older'] = [cl.pop_older for cl in terr_classes]
    '''
    # у ЛО нет октмо в БД
    with pd.option_context("future.no_silent_downcasting", True):
        fin_df['oktmo'] = fin_df['oktmo'].fillna(0)
    '''
    # ____ Если данных колонок нет, то добавляем и ставим нули
    cols = ['density','pop_all','pop_younger','pop_can_work','pop_older',
            'coeff_death','coeff_birth','coeff_migration']
    fin_df = fin_df.reindex(fin_df.columns.union(cols, sort=False), axis=1, fill_value=0)
    fin_df[['pop_all','pop_younger','pop_can_work','pop_older']] = \
        fin_df[['pop_all','pop_younger','pop_can_work','pop_older']].astype(int)
    fin_df[['coeff_death','coeff_birth','coeff_migration']] = [0.01, 0.871, 0.2]
    cols_order = ['territory_id','name','geometry']+cols
    
    return fin_df[cols_order].sort_values('territory_id')
    
    
def main_info(session, current_territory, down_by):
    # ____ Узнаем уровень территории
    try:
        url = terr_api + f'api/v1/territory/{current_territory.territory_id}'
        r_main = session.get(url).json()
        
        
        current_territory.territory_type = r_main['territory_type']['id']
        current_territory.level = r_main['level']
        print('CURRENT TERR LEVEL', current_territory.level)
        current_territory.name = r_main['name']
        #current_territory.oktmo = r_main['oktmo_code']
        
        geom_data = gpd.GeoDataFrame.from_features([r_main])[['geometry']]
        current_territory.geometry = geom_data.values[0][0]
        current_territory.centre_point = create_point(r_main['centre_point'])
        
        current_territory.parent = Territory(r_main['parent']['id'])
        current_territory.parent.name = r_main['parent']['name']
        current_territory.parent.level = current_territory.level-1
        
    except:
        raise requests.exceptions.RequestException(f'Problem with {url}')

    
    try:
        current_territory.pop_all = r_main['properties']['Численность населения']
    except KeyError:
        current_territory.pop_all = 0
        pass
    
    # если будем работать с этой же терр-й
    if down_by == 0:
        #current_territory.df['oktmo'] = current_territory.oktmo
        current_territory.df['geometry'] = geom_data
        current_territory.df['name'] = current_territory.name
        current_territory.df['centre_point'] = current_territory.centre_point
        
        last_pop_and_dnst(session, current_territory, dnst=True, both=True)
        

def pyramid_info(session, terr_classes, specific_year):
    for child in terr_classes:
        #chosen_class = child
        
        pop_df = get_detailed_pop(session, child.territory_id, 
                                    unpack_after_70=False, 
                                    last_year=False, 
                                    specific_year=specific_year)
        
        # если в БД нет данных по пирамиде за 2 года
        if pop_df.columns.get_level_values(0).nunique() < 2:
            pop_df = estimate_child_pyr(session, child.territory_id, 
                                        unpack_after_70=False, 
                                        last_year=False, 
                                        given_year=specific_year)
        

        if pop_df.shape[0]:
            p_all, p_y, p_w, p_o = groups_3(pop_df)
            
            child.pop_all, child.pop_younger,\
                child.pop_can_work,child.pop_older = p_all, p_y, p_w, p_o

        else:
            child.pop_younger,child.pop_can_work,child.pop_older = [0,0,0]
            
        
def child_to_class(x, parent_class):
    child = Territory(x['territory_id'])
    child.name = x['name']
    #child.oktmo = x['oktmo_code']
    child.df['name'] = child.name
    child.geometry = x['geometry']
    child.level = parent_class.level+1
    print()
    print('CHILD TO CLASS', x.index)
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
                  'cities_only':'false','page_size':'5000'}
        r = session.get(url, params=params)
        res = pd.DataFrame(r.json()) 
        res = pd.json_normalize(res['results'], max_level=0)
        fin = res[['territory_id','name','oktmo_code']].copy()
        
        children_level = parent_class.level+1
        if children_level <= 4:
            with_geom = gpd.GeoDataFrame.from_features(r.json()['results'])
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
    

def get_first_children_data(session, territory_id=34):
    try:
        # 34 -- Всеволожский муниципальный район
        url= terr_api + 'api/v1/territory/indicator_values'
        params = {'parent_id':territory_id,'indicator_ids':'1','last_only':'true'}
        r = session.get(url, params=params)
        first_children = pd.DataFrame(r.json())
        
        # df: type, geometry, properties(territory_id,name,...)
        vills_with_geom = pd.json_normalize(first_children['features'], 
                                            max_level=0)
        # df: territory_id, name, indicators
        vills_with_pop = pd.json_normalize(vills_with_geom['properties'], 
                                           max_level=0)
    
        if vills_with_pop['indicators'].str.len().min() > 0:
            # раскрываем json с данными о населении
            # на выходе pd Series [[years, pop_values],...]
            pop_vals = vills_with_pop['indicators'
                                      ].apply(lambda x: pd.json_normalize(x
                                                 )[['date_value','value']].values)
            # берем года для будущих названий колонок
            clms = pop_vals[0][:,0]
            clms = [c[:4] for c in clms] # ['2019', '2020', '2021', '2022', '2023']
            # собираем все значения в один массив
            # 19 x 5 x -1
            pop_vals_one_arr = np.concatenate(pop_vals).reshape(vills_with_geom.shape[0], 
                                                                len(clms), -1) 
            
            vills_with_pop[clms] = pop_vals_one_arr[:,:,1]
            vills_with_pop.rename(columns={clms[0]:'pop_all'}, inplace=True)
            
        vills_with_pop = vills_with_pop.drop('indicators', axis='columns')
        
        for_use = vills_with_geom.copy()
        first_children_f = gpd.GeoDataFrame(vills_with_pop, 
                                           geometry=[shape(d) for d in for_use.pop("geometry")])
        first_children_f['geometry'].crs = 'EPSG:4326'
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
    
    
    return first_children_f

    
def get_children(session, parent_id, parent_class):
    print()
    print('INSIDE GET CHILDREN', parent_class.level) 
    if parent_class.level == 2:
        # для ЛО у детей нет площади у 193
        first_children_f = children_pop_dnst_LO(session, parent_class)
    elif parent_class.level == 4:
        # для НП (show_level=4) нет инф-ии о численности и площади
        first_children_f = children_pop_dnst(session, parent_class, pop_and_dnst=False)
    else:
        first_children_f = get_first_children_data(session, parent_class.territory_id)
        years = first_children_f.filter(regex='\\d{4}').columns
        first_children_f.rename(columns=dict(zip(years,
                                                 years.astype(int))
                                            ),
                                inplace=True)
        #    Плотность населения о ГП/СП
        first_children_f = AreaOnMapFile.calculate_density(first_children_f, 
                                                      pop_clm='pop_all', dnst_clm='density')

        
    print(first_children_f)
    # если есть данные по детям, то раскидываем их по классам детей
    if first_children_f.shape[0]:       
        first_children_f.apply(child_to_class, parent_class=parent_class, axis=1)
    
        
        
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
        
        
def get_detailed_pop(session, territory_id, unpack_after_70=True, 
                     last_year=True, specific_year=0):
    url = social_api + f'indicators/2/{territory_id}/detailed'
    print('get_detailed_pop', unpack_after_70)
    r = session.get(url)
    if r.status_code == 200:
        df_from_json = pd.DataFrame(r.json())
        given_years = df_from_json.year.sort_values().values
        
        '''
        print(given_years)
        if specific_year!=0:
            given_years = [specific_year]
        elif last_year:
            given_years = [given_years.max()]
            
        '''
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


def detailed_pop_info(territory_id=34, forecast_until=2030):
    session = requests.Session()
    # ____ Половозрастная структура
    
    pop_df = get_detailed_pop(session, territory_id, 
                                unpack_after_70=False, 
                                last_year=False, 
                                specific_year=0)
    
    # если в БД нет данных по пирамиде за минимум 2 года
    if pop_df.columns.get_level_values(0).nunique() < 2:
        print('getting parent')
    #if pop_df.shape[0] == 0:
        pop_df = estimate_child_pyr(session, territory_id, 
                                    unpack_after_70=False, 
                                    last_year=False, 
                                    given_year=0)
    print(pop_df)
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
    # прогноз до 2030 года включительно
    
    given_year = 0
    print('FORECAST UNTIL', forecast_until)
    pop_df = DemForecast.get_predictions(pop_df, forecast_until, given_year)
    print(pop_df)
    dynamic_pop_df = pd.DataFrame([pop_df.T.groupby(level=[0]).sum().sum(1)])

    # ____ Половозрастная структура соц.групп (то же, что и пункт 1?)
    soc_pyramid_df = pd.concat(dict(zip(soc_groups, soc_pyramid)), names=['Соц_группа']) 
    
    # ____ Ценности
    pop_years = pop_df.columns.levels[0]
    values_df = pd.DataFrame(np.zeros_like(0, shape=(9, pop_years.shape[0])))
    values_df.columns=pop_years
    values_df
    
    return pop_df, groups_df, dynamic_pop_df, soc_pyramid_df, values_df