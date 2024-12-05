#для работы с таблицами
import numpy as np
import pandas as pd

import os

#для полного копирования данных
import copy

#для создания запросов к api
import requests

#для работы с json-файлами
import json

#terr_api = os.environ.get('TERRITORY_API')

def get_oktmo(territory_id):
	URL = f"http://10.32.1.107:5300/api/v1/territory/{territory_id}"
	r = requests.get(url = URL)
	return int(r.json()['oktmo_code'])

def loc_counts(loc_data, grid_coeffs):
	n_coeffs = copy.deepcopy(grid_coeffs)
	for col in grid_coeffs.keys():
		for row in grid_coeffs[col].keys():
			cell_sum = 0
			if type(grid_coeffs[col][row]) == dict:
				for indicator in grid_coeffs[col][row].keys():
					cell_sum += loc_data[indicator] * grid_coeffs[col][row][indicator]
				n_coeffs[col][row] = cell_sum
	return pd.DataFrame(n_coeffs).reindex(['dev', 'soc', 'bas'])

def tab_to_ser(df, loc = None):
	sr = pd.Series(df.values.flatten(), index = ['comm_dev', 'soc_workers_dev', 'soc_old_dev', 'soc_parents_dev', 'loc_dev',
												 'comm_soc', 'soc_workers_soc', 'soc_old_soc', 'soc_parents_soc', 'loc_soc',
												 'comm_bas', 'soc_workers_bas', 'soc_old_bas', 'soc_parents_bas', 'loc_bas'])
	if loc:
		sr.name = loc
	return sr

def reg_df_to_tab(reg_df, grid_coeffs):
	muni1 = reg_df.index.min()
	reg_tab = pd.DataFrame(tab_to_ser(loc_counts(reg_df.loc[muni1], grid_coeffs), muni1))
	for loc, row in reg_df.iterrows():
		if loc == muni1:
			continue
		reg_tab[loc] = tab_to_ser(loc_counts(row, grid_coeffs))
	return reg_tab.T

def ser_to_tab(sr, grid_coeffs):
	n_coeffs = copy.deepcopy(grid_coeffs)
	for ix in sr.index:
		n_coeffs.loc[ix[-3:], ix[:-4]] = sr.loc[ix]
	return n_coeffs

def recount_data_for_reg(reg_df):
	return reg_df.apply(lambda s: (s - s.min()) / (s.max() - s.min()), axis = 0)

#Final function
def muni_tab(territory_id = 34):
	#34 - это Всеволожский район

	#получаем oktmo данного муниципального образования
	oktmo = get_oktmo(territory_id)
	#получаем oktmo региона, в котором данное мун. образование находится
	reg_oktmo = oktmo - (oktmo % 1000000)

	#загружаем общую таблицу
	full_df = pd.read_csv('app_package/src/full_df2.csv', sep = ';', index_col = 0)

	#получаем таблицу региона
	reg_df = full_df[(full_df.index >= reg_oktmo) & (full_df.index < reg_oktmo + 1000000)]

	##переводим таблицу индикаторов региона в таблицу значений "клеточек" для региона
	#для этого загрузим таблицу коэффициентов
	grid_coeffs = pd.read_json('app_package/src/grid_coeffs.json').reindex(['dev', 'soc', 'bas'])
	reg_tab = reg_df_to_tab(reg_df, grid_coeffs)

	#нормализуем полученные значения по региону
	reg_tab = recount_data_for_reg(reg_tab)

	#получим обратно значения клеточек для нашего мун. образования
	tab = ser_to_tab(reg_tab.loc[oktmo], grid_coeffs)
	
	#теперь выдаём это, как json, и всё
	return tab.to_json()
