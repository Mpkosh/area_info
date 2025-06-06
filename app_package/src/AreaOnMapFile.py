import pandas as pd
import geopandas as gpd

# в geopandas.clip() используется старая версия pandas?  
import warnings
warnings.filterwarnings("ignore",  
                        category=FutureWarning,
                        message=".*will attempt to set the values inplace")


def get_area_from_file(geojson_filename='Границы ЛО (только ГП и СП).geojson', 
             area_name='Гатчинский',  
             choose_from_files=False, area_files_path='area_files/old/', 
             temp_no_water_file='area_files/Границы_только_МР_Границы_ЛО_Без_воды.geojson'):

    p_df = gpd.read_file(area_files_path+geojson_filename)
    
    # оставим только название без "городское поселение"/... 
    clm_area_name = 'name'
    p_df[clm_area_name] = p_df[clm_area_name].apply(lambda x: x.split(' ')[0])
    
    year_clms = p_df.filter(regex='\\d{4}').columns
    p_df[year_clms] = p_df[year_clms].astype(float)
    return p_df


def get_center_point(p_df, clm_area_name='layer', area_name='Гатчинский'):
     # возьмем полигон самого района 
    the_area = p_df[p_df[clm_area_name] == area_name].copy()
    # чтобы подавить warning: считаем центр на метрах, потом переводим
    coord = the_area['geometry'].to_crs(epsg=6933
                                       ).centroid.to_crs(epsg=4326).values[0]
    center_ll = [coord.y, coord.x]
    return center_ll
  
    
def calculate_density(p_df, pop_clm='', dnst_clm=''):
    # сначала придаем crs (в котором !текущие! значения),
    # потом меняем CRS, чтобы мера длины была метр^2
    gpd_df = p_df.copy().set_crs(epsg=4326).to_crs(epsg=6933)
    # мера длины -- км^2
    gpd_df["S"] = gpd_df['geometry'].area/ 10**6
    print(gpd_df)
    
    if not pop_clm:
        # берем колонки с годами в названии (там лежат данные по населению)
        pop_clm = gpd_df.filter(regex='\\d{4}').columns.values

    print(gpd_df[pop_clm])    
    gpd_df[pop_clm]= gpd_df[pop_clm].astype(int)
        
    # считаем плотность населения
    to_add = gpd_df[pop_clm].div(gpd_df["S"], axis=0).round(2)
    # меняем названия колонок и добавляем
    if not dnst_clm:
        to_add.columns = [str(i)+'_dnst' for i in pop_clm]
    else:
        to_add.columns = [dnst_clm]
        
    df = p_df.copy()
    df = pd.concat([df, to_add], axis=1)
    
    return df
