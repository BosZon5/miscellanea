#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction of ISTAT data about municipalities, provinces and regions of Italy.
The extracted data have been analysed with Tableau software to produce several
visualizations. Data come from two ISTAT sources:
    
- Borders of municipalities, provinces and regions:   https://www.istat.it/it/archivio/222527
- Populations, areas and altitudes of municipalities: https://www.istat.it/it/archivio/156224

In the folder in which it is executed, this script produces two subfolders: the
first contains ISTAT data as downloaded from their sources; the second contains
the data preprocessed for the analysis in Tableau.

@author: Andrea Boselli
"""

#%% Relevant libraries

import geopandas as geopd
import os
import pandas    as pd
import requests

from io       import BytesIO
from datetime import datetime
from zipfile  import ZipFile


#%% Settings

settings = {}
settings['shape_data']         = "https://www.istat.it/storage/cartografia/confini_amministrativi/generalizzati/2023/Limiti01012023_g.zip" # location of polygons data
settings['pop_area_alti_data'] = "https://www.istat.it/it/files//2015/04/Classificazioni-statistiche-Anni-2022-2023.zip"                   # location of population, area and altitude data


#%% Introductory steps

# Get current time and data folders names
curr_time   = f'{datetime.now():%Y-%m-%d_%H-%M-%S%z}'
folder_temp = 'data_temp_'+curr_time
folder_out  = 'data_out_' +curr_time
shape_date  = settings['shape_data'].split('/')[-1].replace("Limiti","").replace("_g.zip","")
pop_area_alti_years = settings['pop_area_alti_data'].split("-Anni-")[-1].replace(".zip","")

# Create temporary and output data folder names
if not os.path.exists(folder_temp):
    os.mkdir(folder_temp)
if not os.path.exists(folder_out):
    os.mkdir(folder_out)


#%% Shapes data

# Save the shapes data in the temporary folder
r = requests.get(settings['shape_data'])
z = ZipFile(BytesIO(r.content))
z.extractall(folder_temp)

# Open, process and save the regions data
region_path = os.path.join(folder_temp, "Limiti"+shape_date+"_g", "Reg"+shape_date+"_g") # set data path
region_shpfile = geopd.read_file(region_path, encoding='utf-8')[['COD_REG',   'DEN_REG',
                                                                 'Shape_Area','geometry']] # open data and sort columns
region_shpfile.rename(columns={'COD_REG':   'REG_ID'  , 
                               'DEN_REG':   'Name', 
                               'Shape_Area':'Area (km2)'}, inplace=True)                       # rename columns
region_shpfile.to_file(os.path.join(folder_out, 'shapes_per_region.geojson'), driver='GeoJSON')# save file

# Open, process and save the provinces data
province_path = os.path.join(folder_temp, "Limiti"+shape_date+"_g", "ProvCM"+shape_date+"_g") # set data path
province_shpfile = geopd.read_file(province_path, encoding='utf-8')[['COD_PROV','COD_REG', 
                                                                     'DEN_UTS', 'SIGLA',
                                                                     'TIPO_UTS','Shape_Area',
                                                                     'geometry'            ]] # open data and sort columns
province_shpfile.rename(columns={'COD_PROV':  'PROV_ID',
                                 'COD_REG':   'REG_ID',
                                 'DEN_UTS':   'Name',
                                 'SIGLA':     'Tag',
                                 'TIPO_UTS':  'Type',
                                 'Shape_Area':'Area (km2)'}, inplace=True)                         # rename columns
province_shpfile.to_file(os.path.join(folder_out, 'shapes_per_province.geojson'), driver='GeoJSON')# save file

# Open, process and save the municipalities data
municip_path = os.path.join(folder_temp, "Limiti"+shape_date+"_g", "Com"+shape_date+"_g") # set data path
municip_shpfile = geopd.read_file(municip_path, row = 20, encoding='utf-8')[['PRO_COM','COD_PROV',
                                                                             'COD_REG','COMUNE',
                                                                             'geometry'          ]] # open data and sort columns
municip_shpfile.rename(columns={'PRO_COM': 'MUNI_ID', 
                                'COD_PROV':'PROV_ID', 
                                'COD_REG': 'REG_ID', 
                                'COMUNE':  'Name'}, inplace=True)                                     # rename columns
municip_shpfile.to_file(os.path.join(folder_out, 'shapes_per_municipality.geojson'), driver='GeoJSON')# save file


#%% Population, area, altitude data

# Save the data in the temporary folder
r = requests.get(settings['pop_area_alti_data'])
z = ZipFile(BytesIO(r.content))
z.extractall(folder_temp)

# Find the most recent data filename
pop_area_alti_folder = os.path.join(folder_temp, "Classificazioni-statistiche-Anni_"+pop_area_alti_years)
pop_area_alti_files  = [x for x in os.listdir(pop_area_alti_folder) if '.xls' in x]
pop_area_alti_dates  = [''.join(x.replace('.xls','').strip().split('_')[-1:-4:-1]) for x in pop_area_alti_files]
most_recent_file = pop_area_alti_files[pop_area_alti_dates.index(max(pop_area_alti_dates))]

# Open the most recent file
pop_area_alti_frame = pd.read_excel(os.path.join(pop_area_alti_folder,most_recent_file))

# Retrieve the labels for area and population (that contain also dates)
legal_pop_label = [x for x in pop_area_alti_frame.columns if 'Popolazione legale'      in x][0]
resid_pop_label = [x for x in pop_area_alti_frame.columns if 'Popolazione residente'   in x][0]
area_label      = [x for x in pop_area_alti_frame.columns if 'Superficie territoriale' in x][0]

# Sort the frame columns and change their name
pop_area_alti_frame = pop_area_alti_frame[['Codice Istat del Comune \n(alfanumerico)','Denominazione (Italiana e straniera)',legal_pop_label,
                                           resid_pop_label,area_label,'Altitudine del centro (metri)','Zona altimetrica','Comune litoraneo',
                                           'Comune isolano','Zone costiere','Grado di urbanizzazione']] # sort columns
pop_area_alti_frame.rename(columns={'Codice Istat del Comune \n(alfanumerico)':'MUNI_ID',
                                    'Denominazione (Italiana e straniera)':    'Name',
                                    legal_pop_label:                           'Population (legal)',
                                    resid_pop_label:                           'Population (resident)',
                                    area_label:                                'Area (km2)',
                                    'Altitudine del centro (metri)':           'Altitude (m MSL)',
                                    'Zona altimetrica':                        'Altimetric area',
                                    'Comune litoraneo':                        'Coastal municipality',
                                    'Comune isolano':                          'Island municipality',
                                    'Zone costiere':                           'Coastal zones',
                                    'Grado di urbanizzazione':                 'Urbanization level'}, inplace=True) # rename columns

# Save the data in the output folder
pop_area_alti_frame.to_csv(path_or_buf=os.path.join(folder_out,'pop_area_alti_per_municipality.csv'), index=False)