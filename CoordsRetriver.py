import pandas as pd
from geopy.geocoders import Nominatim
from geopy.point import Point

EUROPE = [Point(34.35873, -24.96094), Point(71.52491, 39.02344)]  # Limiti geografici per tutta Europa

def get_coordinates(address):
    geolocator = Nominatim(user_agent="Geoloc", timeout=10)
    location = geolocator.geocode(address, bounded  = True, viewbox = EUROPE)
    if location: return location
    else: 
        geolocator = Nominatim(user_agent="Geoloc", timeout=10)
        location = geolocator.geocode(address)
        if location: return location
        else: return None

def main():
    nodedata = pd.read_csv("./Data/Euromap_labels.csv", usecols=["city_name"])

    nodedata["gcode"] = nodedata.city_name.apply(get_coordinates)
    nodedata['lat'] = [g.latitude if g != None else None for g in nodedata.gcode]
    nodedata['long'] = [g.longitude if g != None else None for g in nodedata.gcode]
    nodedata.drop("gcode", axis=1, inplace = True)

    nodedata.to_csv("./Data/Euromap_node_data.csv")

main()

