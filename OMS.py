import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import requests

place_name1 = "Mumbai, India"
place_name2 = "Delhi, India"
place_name3 = "Kolkata, India"
place_name4 = "Chennai, India"

# Road network capture
G = ox.graph_from_place(place_name, 
                        network_type='drive', 
                        simplify=True, 
                        retain_all=False, 
                        truncate_by_edge=False, 
                        which_result=None)
ox.plot_graph(G, node_size=2, 
              node_color='r',
              edge_color='w',
              edge_linewidth=0.2)

## POI (Might need to choose pilot first)

# Check boundaries
#zone = ox.geocode_to_gdf("Mumbai, India")
#print(zone)

gdf = ox.geocode_to_gdf ("Mumbai, India")
bounds = gdf.bounds

north = bounds.maxy[0]
south = bounds.miny[0]
east = bounds.maxx[0]
west = bounds.minx[0]

print(bounds)

# Filter freight type facilities (e.g. warehouses, logistics centres, etc.) by tags
tags = {'landuse': 'industrial', 'building': 'warehouse'}

pois = ox.features_from_bbox ([north, south, east, west], tags)

print (pois [['landuse', 'building', 'geometry']].head())

# Calculate POI density
pois = pois.to_crs(bounds.crs)
joined = gpd.sjoin(pois, bounds, how='inner', predicate='intersects')
print(f"POI count: {len(joined)}")

# Visulisation
pois.plot(figsize=(10, 10), color='orange')
plt.title("Industrial & Warehouse POIs in Mumbai")
plt.show()


## SV (need API key)
API_KEY = "YOUR_GOOGLE_API_KEY"

# Latitude and longitude coordinates
location = ","

# Parameters
params = {
    "size": "640x640",      # image size
    "location": location,   # latitude and longitude coordinates
    "heading": "90",        # Viewing angle (rotatable)
    "pitch": "0",           # vertical angle
    "fov": "90",            # field of view
    "key": API_KEY
}

# Construct the request URL
base_url = "https://maps.googleapis.com/maps/api/streetview"
response = requests.get(base_url, params=params)

# Save the image
if response.status_code == 200:
    with open("streetview_sample.jpg", "wb") as f:
        f.write(response.content)
    print("saved")
