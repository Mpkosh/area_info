import pandas as pd
from io import BytesIO
#import glob
#import plotly.express as px
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

    # ищем среди файлов
    if choose_from_files:    
        # новые границы ЛО
        without_water = gpd.read_file(temp_no_water_file)
        all_files = glob.glob(area_files_path + "/*.geojson")
        for filename in all_files:
            if area_name in filename:
                p_df = gpd.read_file(filename)
                # обрезаем границы ГП района по новым границам ЛО
                p_df = gpd.clip(p_df, without_water, keep_geom_type=False)
    # или читаем заданный
    else:
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
  
    
def calculate_density(p_df):
    
    # меняем CRS, чтобы мера длины была метр^2
    gpd_df = p_df.copy().to_crs(epsg=6933)
    # мера длины -- км^2
    gpd_df["S"] = gpd_df['geometry'].area/ 10**6
    # берем колонки с годами в названии (там лежат данные по населению)
    year_clms = gpd_df.filter(regex='\\d{4}').columns
    gpd_df[year_clms]= gpd_df[year_clms].astype(int)
    # считаем плотность населения
    to_add = gpd_df[year_clms].div(gpd_df["S"], axis=0).round(2)
    # меняем названия колонок и добавляем
    to_add.columns = [str(i)+'_dnst' for i in year_clms]# list(year_clms + '_dnst')
    df = p_df.copy()
    df = pd.concat([df, to_add], axis=1)
    
    return df
