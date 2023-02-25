import pandas as pd
import networkx as nx

edges = pd.read_csv("./Data/Euromap_edges.csv")
names = pd.read_csv("./Data/Euromap_labels.csv", usecols=["city_name"])
merged = pd.DataFrame(columns=["Source", "Target"])

for index, row in edges.iterrows():
    f = row["from"]
    t = row["to"]
    newrow = {'Source': names['city_name'].iloc[f - 1], 'Target': names['city_name'].iloc[t - 1]}
    merged = merged.append(newrow, ignore_index=True)

merged.to_csv("./Data/Euromap_named_edges.csv", index=False)

coords = pd.read_csv("./Data/Euromap_node_data.csv", usecols=["city_name", "lat", "long"])
coords = coords.rename(columns={"city_name": "id"})
coords.to_csv("./Data/Euromap_data_for_gephy.csv", index=False)