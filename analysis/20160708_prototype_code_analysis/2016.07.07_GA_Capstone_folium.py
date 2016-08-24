#Python script file for producing marker-maps and heatmaps using the folium package.

import folium
from folium.plugins import HeatMap
import webbrowser
import pandas as pd

df = pd.read_csv('mapping_merged_cleaned_report_results.csv')

#Maps adjusted for 75/25% split of violation data
def marker_crit_viol_non_crit_viol(df):
    #Keep latitude and longitude seperate for markers
    df_crit_viol = df[df.crit_viol >= 6]
    lat_crit_viol = df_crit_viol['lat']
    lat_crit_viol = lat_crit_viol.tolist()
    lon_crit_viol = df_crit_viol['lon']
    lon_crit_viol = lon_crit_viol.tolist()

    df_less_crit_viol = df[df.crit_viol < 6]
    lat_less_crit_viol = df_less_crit_viol['lat']
    lat_less_crit_viol = lat_less_crit_viol.tolist()
    lon_less_crit_viol = df_less_crit_viol['lon']
    lon_less_crit_viol = lon_less_crit_viol.tolist()


    #for i in range(0,2):
    #    df_i = df[df.ViolationType == i]
    #    lats_i = df_i['Latitude']
    #    lats_i = lats_i.tolist()
    #    longs_i = df_i['Longitude']
    #    longs_i = longs_i.tolist()


    map_1 = folium.Map(location = [38.8951100, -77.0363700], zoom_start = 10)

    for i in range(len(df_crit_viol)):
        map_1.simple_marker([lat_crit_viol[i], lon_crit_viol[i]], marker_color = 'red')
    for i in range(len(df_less_crit_viol)):
        map_1.simple_marker([lat_less_crit_viol[i], lon_less_crit_viol[i]], marker_color = 'blue')
    
    map_1.create_map(path = 'DC_viol_marker_map.html')
    
    webbrowser.open_new("DC_viol_marker_map.html")

def marker_risk(df):
    #Keep latitude and longitude seperate for markers
#    df_0 = df[df.risk == 0]
#    lat_0 = df_0['lat']
#    lat_0 = lat_0.tolist()
#    lon_0 = df_0['lon']
#    lon_0 = lon_0.tolist()
#    
#    df_1 = df[df.risk == 1]
#    lat_1 = df_1['lat']
#    lat_1 = lat_1.tolist()
#    lon_1 = df_1['lon']
#    lon_1 = lon_1.tolist()
#    
#    df_2 = df[df.risk == 2]
#    lat_2 = df_2['lat']
#    lat_2 = lat_2.tolist()
#    lon_2 = df_2['lon']
#    lon_2 = lon_2.tolist()
    
    df_3 = df[df.risk == 3]
    lat_3 = df_3['lat']
    lat_3 = lat_3.tolist()
    lon_3 = df_3['lon']
    lon_3 = lon_3.tolist()
    
    df_4 = df[df.risk == 4]
    lat_4 = df_4['lat']
    lat_4 = lat_4.tolist()
    lon_4 = df_4['lon']
    lon_4 = lon_4.tolist()
    
    df_5 = df[df.risk == 5]
    lat_5 = df_5['lat']
    lat_5 = lat_5.tolist()
    lon_5 = df_5['lon']
    lon_5 = lon_5.tolist()
    
    map_1 = folium.Map(location = [38.8951100, -77.0363700], zoom_start = 10)

#    for i in range(len(df_0)):
#        map_1.simple_marker([lat_0[i], lon_0[i]], marker_color = 'purple')
#    for i in range(len(df_1)):    
#        map_1.simple_marker([lat_1[i], lon_1[i]], marker_color = 'green')
#    for i in range(len(df_2)):     
#        map_1.simple_marker([lat_2[i], lon_2[i]], marker_color = 'blue')
    for i in range(len(df_3)):     
        map_1.simple_marker([lat_3[i], lon_3[i]], marker_color = 'blue')
    for i in range(len(df_4)):    
        map_1.simple_marker([lat_4[i], lon_4[i]], marker_color = 'orange')
    for i in range(len(df_5)):    
        map_1.simple_marker([lat_5[i], lon_5[i]], marker_color = 'red')

    map_1.create_map(path = 'DC_risk_marker_map.html')
    
    webbrowser.open_new("DC_risk_marker_map.html")
    
def heatmap_crit_viol(df):
    #Zip latitude and longitude together for heatmaps
    df_crit_viol = df[df.crit_viol >= 6]
    lat_crit_viol = df_crit_viol['lat']
    lat_crit_viol = lat_crit_viol.tolist()
    lon_crit_viol = df_crit_viol['lon']
    lon_crit_viol = lon_crit_viol.tolist()    

    coordinates_crit_viol = zip(lat_crit_viol, lon_crit_viol)    
    
    heatmap_1 = folium.Map(location = [38.8951100, -77.0363700], zoom_start = 10)
    gradient = {0.2: 'blue', 0.2: 'green', 1: 'red'} 
    heatmap_1.add_children(HeatMap(coordinates_crit_viol, radius = 10, gradient = gradient))
       
    heatmap_1.create_map(path = 'DC_crit_viol_heatmap.html')
    
    webbrowser.open_new("DC_crit_viol_heatmap.html")

#For breaking apart by different health violations or clusters, create 
#different lats,longs lists for each group and create a different scatter
#and heatmap instance. Draw all of them at once.

def heatmap_less_crit_viol(df):
    #Zip latitude and longitude together for heatmaps
    df_crit_viol = df[df.crit_viol < 6]
    lat_crit_viol = df_crit_viol['lat']
    lat_crit_viol = lat_crit_viol.tolist()
    lon_crit_viol = df_crit_viol['lon']
    lon_crit_viol = lon_crit_viol.tolist()    

    coordinates_crit_viol = zip(lat_crit_viol, lon_crit_viol) 
    
    heatmap_1 = folium.Map(location = [38.8951100, -77.0363700], zoom_start = 10)
     
    heatmap_1.add_children(HeatMap(coordinates_crit_viol, radius = 10))
       
    heatmap_1.create_map(path = 'DC_less_crit_viol_heatmap.html')
    
    webbrowser.open_new("DC_less_crit_viol_heatmap.html")
    
def heatmap_risk3plus(df):
    #Zip latitude and longitude together for heatmaps
    df_3plus = df[df.risk >= 3]
    lat_3plus = df_3plus['lat']
    lat_3plus = lat_3plus.tolist()
    lon_3plus = df_3plus['lon']
    lon_3plus = lon_3plus.tolist()

    coordinates_3plus = zip(lat_3plus, lon_3plus)
    
    heatmap_1 = folium.Map(location = [38.8951100, -77.0363700], zoom_start = 10)
    gradient = {0.1: 'blue', 0.2: 'green', 1: 'red'} 
    heatmap_1.add_children(HeatMap(coordinates_3plus, radius = 10, gradient = gradient))
       
    heatmap_1.create_map(path = 'DC_risk_3_plus_heatmap.html')
    
    webbrowser.open_new("DC_risk_3_up_heatmap.html")


marker_crit_viol_non_crit_viol(df)
marker_risk(df)
heatmap_crit_viol(df)
heatmap_less_crit_viol(df)
heatmap_risk3plus(df)
