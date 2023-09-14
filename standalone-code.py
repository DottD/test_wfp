from io import StringIO

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import requests
from shapely.geometry import shape

# General parameters
adm_units_path = 'mdg_adm_bngrc_ocha_20181031_shp.zip'
file_ref = 'mdg_admbnda_adm2_BNGRC_OCHA_20181031'
pop_est_path = 'mdg_pd_2020_1km.tif'
cyclone_url = 'https://www.gdacs.org/gdacsapi/api/polygons/getgeometry?eventtype=TC&eventid=1000859&episodeid=13&sourceid=JTWC'


# Read the TIF with population estimates
img = rasterio.open(pop_est_path)
# Convert the raster image to vector of squares corresponding to pixels
shapes = rasterio.features.shapes(img.read(1, masked=True), transform=img.transform)
# Get two array with geojson-like shapes and values
gjs_shapes, values = zip(*shapes)
# Convert to GeoDataFrame by using only the centroid of the pixel.
# This avoid counting multiple times the same pixel if it intersects
# more than one administrative subdivision.
pop_gis = gpd.GeoDataFrame(data={'Total_population_by_adm2': list(values)}, geometry=[shape(s).centroid for s in gjs_shapes], crs=4326)

# Read the zip file with administrative subdivisions
adm = gpd.read_file(f'zip://{adm_units_path}!{file_ref}.shp')
# Remove useless columns to speed up the computation
adm.drop(columns=set(adm.columns)-set([
    'ADM0_EN',
    'ADM1_EN',
    'ADM2_PCODE',
    'ADM2_EN',
    'geometry',
]), inplace=True)

# Perform the spatial join between the administrative information
# and the population dataframe: the resulting dataframe will contain
# any matching line between the two initial dataframes
# with information coming from both.
adm_sjoin_pop = adm.sjoin(pop_gis, how='inner', predicate="intersects")

# Remove useless columns
adm_sjoin_pop.drop(columns=['index_right'], inplace=True)

# Turn the index of the left dataframe to a column
adm_sjoin_pop = adm_sjoin_pop.reset_index(names=['index_left'])

# Now we can group by that index to compute the population by adm.
# This answers question 1.
ops = {k: 'first' for k in adm_sjoin_pop.columns}
ops.update({'Total_population_by_adm2': 'sum'})
pop_by_adm = adm_sjoin_pop.groupby('index_left').aggregate(ops)
pop_by_adm.drop(columns=['index_left'], inplace=True)

# Download the cyclone information and convert to a GeoDataFrame
cyclone_response = requests.get(cyclone_url)
if not cyclone_response.ok:
    raise ValueError()
cyclone_data = cyclone_response.text
geojson_raw = StringIO(cyclone_data)
cyclone_gis = gpd.read_file(geojson_raw)
# Create a column with wind buffer speed as a numerical field (where possible)
# (this is done by stripping the speed unit and converting to float)
wind_speed_mask = cyclone_gis.polygonlabel.str.contains('km/h')
cyclone_gis = cyclone_gis[wind_speed_mask].copy()
cyclone_gis['wind_speed'] = cyclone_gis.polygonlabel.str.replace(' km/h', '').astype(float)
cyclone_gis.sort_values('wind_speed', inplace=True)
# Remove useless columns
cyclone_gis.drop(columns=set(cyclone_gis.columns)-set([
    'wind_speed',
    'geometry',
]), inplace=True)

# Compute the difference between the wind buffer so as not to consider
# each adm polygon more than once.
poly60 = cyclone_gis[cyclone_gis.wind_speed == 60.0].geometry.item()
poly90 = cyclone_gis[cyclone_gis.wind_speed == 90.0].geometry.item()
poly120 = cyclone_gis[cyclone_gis.wind_speed == 120.0].geometry.item()
cyclone_gis.loc[cyclone_gis.wind_speed == 90.0, 'geometry'] = [poly90 - poly120]
cyclone_gis.loc[cyclone_gis.wind_speed == 60.0, 'geometry'] = [poly60 - poly90]

# Perform the spatial join between the cyclone wind buffer polygons and
# the population data points.
pop_by_cyclone = cyclone_gis.sjoin(pop_gis, how='inner', predicate="contains")
pop_by_cyclone.reset_index(names='index_left', inplace=True)
# Grouping by wind_speed and summing over the population field, we get the answer to question 2.
pop_by_cyclone = pop_by_cyclone.groupby('wind_speed').aggregate({'Total_population_by_adm2': 'sum'}).round(2)
pop_by_cyclone = pop_by_cyclone.T.rename(columns={
    60.0: 'Total_people_at_60kmph',
    90.0: 'Total_people_at_90kmph',
    120.0: 'Total_people_at_120kmph',
})
print(pop_by_cyclone.to_string())

# Spatial join between administrative data and cyclone data.
final_data = gpd.GeoDataFrame(pop_by_adm, crs=4326).sjoin(cyclone_gis, how='inner', predicate='intersects')
# Convert the dataframe to a metric CRS and compute the area.
# The population pertaining to a wind buffer will be computed
# as the a fraction of the total population, which is given by
# the area of the overlapping wind buffer divided by the total area.
overlay_data = gpd.overlay(gpd.GeoDataFrame(pop_by_adm, crs=4326).reset_index(names='index_left'), cyclone_gis, how='intersection')
overlay_data['overlapping_area'] = overlay_data.to_crs(8441).area
overlay_data.set_index(['index_left', 'wind_speed'], inplace=True)
final_data = final_data.reset_index()
final_data.set_index(['index_left', 'wind_speed'], inplace=True)
final_data['fraction'] = overlay_data.overlapping_area / final_data.to_crs(8441).area
# Finally compute the population at risk
final_data['fraction'] *= final_data['Total_population_by_adm2']

# Bring this information into the "population by adm" dataframe.
final_data = final_data.drop(columns=['geometry'])
final_data.reset_index('wind_speed', inplace=True)
final_60kmph = final_data[final_data.wind_speed == 60.0].fraction
final_90kmph = final_data[final_data.wind_speed == 90.0].fraction
final_120kmph = final_data[final_data.wind_speed == 120.0].fraction
pop_by_adm['People_at_60kmph'] = np.nan
pop_by_adm.loc[final_60kmph.index, 'People_at_60kmph'] = final_60kmph.round(2)
pop_by_adm['People_at_90kmph'] = np.nan
pop_by_adm.loc[final_90kmph.index, 'People_at_90kmph'] = final_90kmph.round(2)
pop_by_adm['People_at_120kmph'] = np.nan
pop_by_adm.loc[final_120kmph.index, 'People_at_120kmph'] = final_120kmph.round(2)
# Compute the percentage of population living in each wind buffer.
pop_by_adm['%_people_at_60kmph'] = (pop_by_adm.People_at_60kmph / pop_by_adm.Total_population_by_adm2 * 100.0).round(2)
pop_by_adm['%_people_at_90kmph'] = (pop_by_adm.People_at_90kmph / pop_by_adm.Total_population_by_adm2 * 100.0).round(2)
pop_by_adm['%_people_at_120kmph'] = (pop_by_adm.People_at_120kmph / pop_by_adm.Total_population_by_adm2 * 100.0).round(2)
# Get rid of geometry information
pop_by_adm.drop(columns=['geometry'], inplace=True)
# Save the dataframe to an Excel spreadsheet.
pop_by_adm.to_excel('outuput.xlsx', index=False)
