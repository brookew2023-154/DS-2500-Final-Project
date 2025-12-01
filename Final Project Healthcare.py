# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:36:36 2025

@author: Brooke Wassmann
NUID: 02607768
DS 2500
Description of program
"""
import pandas as pd 
import numpy as np 
from pathlib import Path
import re
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Scraping():
    
    def __init__(self, countyName, stateName, stateAbr, dataType):
        self.countyName = countyName
        self.stateName = stateName
        self.dataType = dataType
        self.stateAbr = stateAbr
        self.countyColumn = f"{self.countyName.replace('_', ' ')} County"
        
    def scrape(self):
        rural = ['Pend_Oreille', 'Garfield', 'Asotin', 'Talbot', 'Randolph', 'Clinch', 'Jasper', 'Gallatin', 'Vinton', 'Monroe', 'Morgan', 'Jackson', 'Sedgwick', 'Baca']
        suburban = ['Clark', 'Kitsap', 'Yakima', 'Whitfield', 'Walton', 'Bartow', 'McHenry', 'Winnebago', 'Madison', 'Butler', 'Stark', 'Lorain', 'Douglas', 'Larimer', 'Weld']
        urban = ['King', 'Spokane', 'Pierce', 'Fulton', 'Gwinnett', 'Cobb', 'Franklin', 'Cuyahoga', 'Cook', 'DuPage', 'Lake', 'El_Paso', 'Denver', 'Arapahoe']
        if self.stateAbr == 'IL':
            if 'Hamilton' not in rural: rural.append('Hamilton')
        elif self.stateAbr == 'OH':
            if 'Hamilton' not in urban: urban.append('Hamilton')
        base_directory = Path(__file__).parent
        filename = f"{self.stateName}/{self.dataType}_{self.countyName}_{self.stateAbr}.csv"
        filepath = base_directory/filename
        table = pd.read_csv(filepath, skiprows=1)
        table = table.dropna(axis = 0, how = 'any')

        if self.dataType == 'PC':
            extracted_values = (table[self.countyColumn]
                                .astype(str)
                                .str.extract(r'^(\d+)', expand=False) 
                                .astype(float) 
                               )
            cleaned_values = np.where(extracted_values > 0, 
                                      np.floor(100000 / extracted_values), 
                                      0) 
            table[self.countyColumn] = cleaned_values
            
        if self.dataType == 'VAX':
            valueVAX_cleaned = (table[self.countyColumn]
                                .astype(str)
                                .str.replace(r'\%$', '', regex=True) 
                                .astype(float) 
                               )
            table[self.countyColumn] = np.floor((valueVAX_cleaned / 100) * 100000)
        if self.countyName in rural:
            table['County Type'] = 'Rural'
        elif self.countyName in suburban:
            table['County Type'] = 'Suburban'  
        elif self.countyName in urban:
            table['County Type'] = 'Urban' 
        return table

class Hospitalizations():
    
    def __init__(self, stateName, stateAbr ):
        self.stateName = stateName
        self.stateAbr = stateAbr
        self.year_column = 'YEAR' 
        self.rate_column = 'WEEKLY RATE' 
        
    def hospitalizations (self):
        base_directory = Path(__file__).parent
        filename = f"{self.stateName}/Hospitalizations_{self.stateAbr}.csv"
        filepath = base_directory/filename
        table = pd.read_csv(filepath, skiprows=2)
        table[self.rate_column] = pd.to_numeric(
            table[self.rate_column], errors='coerce'
        ).fillna(0)
        annual_summary = (
            table.groupby(self.year_column)[self.rate_column]
                 .sum()
                 .reset_index(name='Total_Flu_Rate')
        )
        annual_summary = annual_summary[annual_summary[self.year_column].str.strip() != 'null']
        annual_summary['Year'] = annual_summary[self.year_column].apply(
            lambda x: int(x.split('-')[1]) if '-' in x else 0
        )
        return annual_summary[annual_summary['Year'] != 0][['Year', 'Total_Flu_Rate']]
        
def merge_and_graph_multi_county(tables_dict, data_type):
        first_table = tables_dict.values().__iter__().__next__()
        master_table = pd.DataFrame(first_table[['Year']])
        county_type_lookup = {}
        for key, table in tables_dict.items():
            county_name_file = key.replace(data_type, '')
            county_col = f"{county_name_file.replace('_', ' ')} County"
            county_type_lookup[county_col] = table['County Type'].iloc[0]
            data_to_merge = table[['Year', county_col]].copy()
            new_col_name = county_col.replace(' ', '_') 
            data_to_merge.rename(columns={county_col: new_col_name}, inplace=True)
            master_table = pd.merge(master_table, data_to_merge, on='Year', how='inner')
        plt.figure(figsize=(12, 8))
        if data_type == 'PC':
            title = "Primary Care Physicians per 100,000 People (All Counties)"
            ylabel = "PCPs per 100,000 People"
        elif data_type == 'VAX':
            title = "Citizens Vaccinated Against Flu per 100,000 People (All Counties)"
            ylabel = "Vaccinated Citizens per 100,000 People"
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("Year")
        color_map = {'Rural': 'b', 'Suburban': 'r', 'Urban': 'k'}
        for col in master_table.columns:
            if col != 'Year':
                county_name_key = col.replace('_', ' ') 
                county_type = county_type_lookup.get(county_name_key)
                color = color_map[county_type]
                plt.plot(master_table['Year'], master_table[col], marker='o', label=county_name_key, color = color)
        plt.xticks(master_table['Year'], rotation=45, ha='right')
        plt.legend(title="County", loc='best')
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        print('Rural counties are in blue, suburban counties are in red, urban counties are in black.')
        return master_table 
        
    

coloradoCounties = ['Arapahoe', 'Baca', 'Denver', 'Douglas', 'El_Paso', 'Jackson', 'Larimer', 'Sedgwick', 'Weld']
georgiaCounties = ['Bartow', 'Clinch', 'Cobb', 'Fulton', 'Gwinnett', 'Randolph', 'Talbot', 'Walton', 'Whitfield']
illinoisCounties = ['Cook', 'DuPage','Gallatin', 'Hamilton', 'Jasper', 'Lake', 'Madison', 'McHenry', 'Winnebago']
ohioCounties = ['Butler', 'Cuyahoga', 'Franklin', 'Hamilton', 'Lorain', 'Monroe', 'Morgan', 'Stark', 'Vinton']
washingtonCounties = ['Asotin', 'Clark', 'Garfield', 'King', 'Kitsap', 'Pend_Oreille', 'Pierce', 'Spokane', 'Yakima']
county_vax_scrapers = {}
county_PC_scrapers = {}
for county in coloradoCounties:
    object_name = f"{county}VAX"
    scraper_instance = Scraping(county, 'Colorado','CO', 'VAX')
    tableVax = scraper_instance.scrape()
    county_vax_scrapers[object_name] = tableVax
    object_name = f"{county}PC"
    scraper_instance = Scraping(county, 'Colorado', 'CO', 'PC')
    tablePC = scraper_instance.scrape()
    county_PC_scrapers[object_name] = tablePC
for county in georgiaCounties:
    object_name = f"{county}VAX"
    scraper_instance = Scraping(county, 'Georgia','GA', 'VAX')
    tableVax = scraper_instance.scrape()
    county_vax_scrapers[object_name] = tableVax
    object_name = f"{county}PC"
    scraper_instance = Scraping(county, 'Georgia', 'GA', 'PC')
    tablePC = scraper_instance.scrape()
    county_PC_scrapers[object_name] = tablePC
for county in illinoisCounties:
    object_name = f"{county}VAX"
    scraper_instance = Scraping(county, 'Illinois','IL', 'VAX')
    tableVax = scraper_instance.scrape()
    county_vax_scrapers[object_name] = tableVax
    object_name = f"{county}PC"
    scraper_instance = Scraping(county, 'Illinois', 'IL', 'PC')
    tablePC = scraper_instance.scrape()
    county_PC_scrapers[object_name] = tablePC
for county in ohioCounties:
    object_name = f"{county}VAX"
    scraper_instance = Scraping(county, 'Ohio','OH', 'VAX')
    tableVax = scraper_instance.scrape()
    county_vax_scrapers[object_name] = tableVax
    object_name = f"{county}PC"
    scraper_instance = Scraping(county, 'Ohio', 'OH', 'PC')
    tablePC = scraper_instance.scrape()
    county_PC_scrapers[object_name] = tablePC
for county in washingtonCounties:
    object_name = f"{county}VAX"
    scraper_instance = Scraping(county, 'Washington','WA', 'VAX')
    tableVax = scraper_instance.scrape()
    county_vax_scrapers[object_name] = tableVax
    object_name = f"{county}PC"
    scraper_instance = Scraping(county, 'Washington', 'WA', 'PC')
    tablePC = scraper_instance.scrape()
    county_PC_scrapers[object_name] = tablePC
merged_pc_table = merge_and_graph_multi_county(county_PC_scrapers, 'PC')
merged_vax_table = merge_and_graph_multi_county(county_vax_scrapers, 'VAX')
initialize_hospitalizations_CO = Hospitalizations("Colorado", "CO")
hospitalizations_CO = initialize_hospitalizations_CO.hospitalizations()
initialize_hospitalizations_GA = Hospitalizations("Georgia", "GA")
hospitalizations_GA = initialize_hospitalizations_GA.hospitalizations()
initialize_hospitalizations_OH = Hospitalizations("Ohio", "OH")
hospitalizations_OH = initialize_hospitalizations_OH.hospitalizations()
co_flu_rate = hospitalizations_CO.set_index('Year')['Total_Flu_Rate']
coloradoCounties_formatted = [f"{c.replace('_', ' ')} County".replace(' ', '_') for c in coloradoCounties]
pc_co_cols = [col for col in merged_pc_table.columns if col in coloradoCounties_formatted]
vax_co_cols = [col for col in merged_vax_table.columns if col in coloradoCounties_formatted]
X_independent_vars = merged_pc_table[['Year'] + pc_co_cols].merge(
    merged_vax_table[['Year'] + vax_co_cols], on='Year', how='inner'
).set_index('Year')
final_co_regression_data = X_independent_vars.merge(
    co_flu_rate, left_index=True, right_index=True, how='inner'
)
if final_co_regression_data.empty:
    print("FATAL ERROR: The final dataset is empty. Check Year columns for mismatch.")
    # Check what the indices look like if it fails
    print("\nX Independent Vars Index (first 5):", X_independent_vars.index.head())
    print("\nFlu Rate Index (first 5):", co_flu_rate.index.head())
else:
    Y = final_co_regression_data['Total_Flu_Rate']
    X = final_co_regression_data.drop(columns=['Total_Flu_Rate'])
    X = sm.add_constant(X)
    
    model = sm.OLS(Y, X, missing='drop').fit()
    
    print("\n--- Colorado Multiple Linear Regression Results (Rate vs. PC/VAX) ---\n")
    print(model.summary().as_text())