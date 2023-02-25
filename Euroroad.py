from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import seaborn as sns
from collections import Counter
import numpy as np
import random
import community as community_louvain
import matplotlib.cm as cm
import itertools
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform
import networkx.algorithms.community as nx_comm

DIGIT_PRECISION = 3
TOP_NODES = 7
NUMBER_OF_PARTITIONS = 15
ALPHA = 0.9

def plot_stats(graph):
    stats = {}

    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    stats["Number of nodes"] = graph.number_of_nodes()
    stats["Number of edges"] = graph.number_of_edges()
    stats["Average degree"] = round(sum(degree_sequence) / len(degree_sequence),DIGIT_PRECISION)
    stats["Average distance"] = round(nx.average_shortest_path_length(graph),DIGIT_PRECISION)
    stats["Diameter"] = round(nx.diameter(graph),DIGIT_PRECISION)
    stats["Global clustering coefficient"] = round(nx.average_clustering(graph),DIGIT_PRECISION)
    stats["Density"] = round(nx.density(graph),DIGIT_PRECISION)
    stats["Efficiency"] = round(nx.global_efficiency(graph),DIGIT_PRECISION)
    
    data = [(k, v) for k, v in stats.items()]
    
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=data, loc='center')  

    for i in range(len(data)):
        for j in range(2):
            cell = table[i, j]
            text = cell.get_text()
            if(j == 0): text.set_ha('left')
            else: text.set_ha('center')
            text.set_va('center')

    for i in range(len(data)):
        table[i, 0].set_facecolor("#CCE5FF")

    ax.set_title('Graph stats', fontweight ="bold") 
    fig.tight_layout()
    plt.savefig("./Images/Europe_stats.png")
    plt.show() 

def plot_rankings(graph):
    top_degree = dict(Counter(dict(graph.degree())).most_common(TOP_NODES))

    top_eigenvector = dict(Counter(nx.eigenvector_centrality_numpy(graph)).most_common(TOP_NODES))

    top_closeness = dict(Counter(nx.closeness_centrality(graph)).most_common(TOP_NODES))

    top_betweenness = dict(Counter(nx.betweenness_centrality(graph)).most_common(TOP_NODES))

    hubs_centrality, authorities_centrality = nx.hits(graph, normalized=True, tol=1e-08)
    top_hubs = dict(Counter(hubs_centrality).most_common(TOP_NODES))
    top_authorities = dict(Counter(authorities_centrality).most_common(TOP_NODES))
    
    plt.title("Degree centrality")
    plt.bar(top_degree.keys(), top_degree.values())

    plt.savefig("./Images/Degree_top.png")
    plt.show()
    
    plt.title("Eigenvector centrality")
    plt.bar(top_eigenvector.keys(), top_eigenvector.values())

    plt.savefig("./Images/Eigenvector_top.png")
    plt.show()
    
    plt.title("Closeness centrality")
    plt.bar(top_closeness.keys(), top_closeness.values())

    plt.savefig("./Images/Closeness_top.png")
    plt.show()
    
    plt.title("Betweenness centrality")
    plt.bar([k.replace(" ", "\n") for k in top_betweenness.keys()], top_betweenness.values())

    plt.savefig("./Images/Betweenness_top.png")
    plt.show()

    plt.title("Hub score")
    plt.bar(top_hubs.keys(), top_hubs.values())

    plt.savefig("./Images/Hub_top.png")
    plt.show()
    
    plt.title("Authority score")
    plt.bar(top_authorities.keys(), top_authorities.values())

    plt.savefig("./Images/Authority_top.png")
    plt.show()
    
def plot_graph_on_map(graph, nodesdata):
    fig, ax = plt.subplots(figsize=(10,10))
    m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=70, llcrnrlon=-28, urcrnrlon=55, resolution='l')
    mx, my = m(nodesdata['long'].values, nodesdata['lat'].values)

    pos = {}
    for count, elem in enumerate (nodesdata['city_name']):
        pos[elem] = (mx[count], my[count])

    m.drawcoastlines()
    m.fillcontinents(color='#F0E0BB', lake_color='#BBDDF0')
    m.drawmapboundary(fill_color='#BBDDF0')
    m.drawcountries(linewidth=0.5)

    nx.draw_networkx_nodes(G = graph, pos = pos, nodelist = graph.nodes(), node_color = 'r', alpha = 1, node_size = 15)
    nx.draw_networkx_edges(G = graph, pos = pos, edge_color='b', alpha=0.6, arrows = True)

    plt.tight_layout()
    plt.savefig("./Images/Europe_plot.png")
    plt.show()

def plot_degree_distribution(graph):
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt = tuple(i/graph.number_of_nodes() for i in cnt)

    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.plot(deg, cnt)
    plt.savefig("./Images/Degree_distribution.png")
    plt.show()

    cs = np.cumsum(cnt)
    plt.xlabel("k")
    plt.ylabel(r'$\bar{P}$' + "(k)",fontsize=12)
    plt.plot(deg, cs)
    plt.savefig("./Images/Cumulated_degree_distribution.png")
    plt.show()

def plot_conn_components(graph):
    pos = nx.fruchterman_reingold_layout(graph)
    subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    for index, sg in enumerate(subgraphs):
        r = lambda: random.randint(0,255)
        color = '#{:02x}{:02x}{:02x}'.format(r(), r(), r())
        nx.draw_networkx(sg, pos = pos, edge_color = color, node_color = color, with_labels=False, node_size=10)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("./Images/Components_plot.png")
    plt.show()

def inverse_community_mapping(partition):
    partition_mapping = {}
    internal_degrees = {}
    for c in range(min(partition.values()),max(partition.values()) + 1):
        partition_mapping[c] = [k for k, v in partition.items() if v == c]
        internal_degrees[c] = 0

    return partition_mapping, internal_degrees

def compute_persistance_probabilities(graph, partition):
    total_degree = {}
    persistance_probabilities = {}

    #Inverse communities mapping
    partition_mapping, internal_degrees = inverse_community_mapping(partition)

    #Internal degree computation
    for community, nodes in partition_mapping.items():
        for edge in itertools.combinations(nodes, 2):
            if(graph.has_edge(*edge)):
                sourceFile = open('demo.txt', 'a', encoding='utf-8')
                print(edge, file = sourceFile)
                sourceFile.close()
                internal_degrees[community] += 2

    #Total degree for communities
    for community, nodes in partition_mapping.items():
        total_degree[community] = sum([val for (_, val) in graph.degree(nodes)])

    for c in range(min(partition.values()),max(partition.values()) + 1):
        persistance_probabilities[c] = internal_degrees[c] / total_degree[c]

    return persistance_probabilities

def hierarchial_divisive_clustering(graph, t_max):
    partitions = {}
    distance_matrix = nx.floyd_warshall_numpy(graph)

    # Appy divisive clustering
    Z = linkage(squareform(distance_matrix), 'ward')
    for i in range(1,t_max+1):
        clusters = fcluster(Z, i , criterion='maxclust')
        community = {}
        for j in range(len(clusters)):
            community[list(graph.nodes)[j]] = clusters[j]
        partitions[i] = community

    return partitions

def plot_alpha_communities(prob):
    for nop in range(2, NUMBER_OF_PARTITIONS + 1):
        plt.plot([nop]*len(prob[nop].keys()) , prob[nop].values(), '-o')
    
    plt.axhline(y = ALPHA, color = 'r', linestyle = 'dashed', label = "\u03B1")
    plt.xlabel('n° of communities', fontweight ='bold')
    plt.ylabel('Persistance prob.', fontweight ='bold')
    plt.legend(loc="lower right")
    plt.ylim([max(ALPHA-0.2,0),1])
    plt.tight_layout()
    plt.show()

def compute_k_core_decomposition(graph):
    core_numbers = nx.algorithms.core.core_number(graph)
    print(len(inverse_community_mapping(core_numbers)[0][2]))
    kcoredf = pd.DataFrame({"Id":core_numbers.keys(), "kShellId":core_numbers.values()})
    kcoredf.to_csv("./Data/Euromap_GCC_kcoreDecomposition.csv", index=False)

def main():
    # Load the data
    nodesdata = pd.read_csv("./Data/Euromap_node_data.csv", usecols=["city_name", "lat", "long"])
    edges = pd.read_csv("./Data/Euromap_named_edges.csv")
    graph = nx.from_pandas_edgelist(edges, source = 'Source', target = 'Target')

    sns.set()

    # Plot the graph
    #plot_graph_on_map(graph, nodesdata)
    #plot_conn_components(graph)

    # Retrieve only the giant component
    GCC = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
    df = nx.to_pandas_edgelist(GCC)
    df.to_csv("./Data/Euromap_GCC.csv", index=False)

    #plot_stats(GCC)
    #plot_rankings(GCC)
    #plot_degree_distribution(GCC)

    # Core-Periphery analysis
    compute_k_core_decomposition(GCC)
    
    # Alpha partition analysis
    persistance_probabilities = {}
    partitions = hierarchial_divisive_clustering(GCC, NUMBER_OF_PARTITIONS)
    for nop in range(2, NUMBER_OF_PARTITIONS + 1):
        persistance_probabilities[nop] = compute_persistance_probabilities(GCC,partitions[nop])

    plot_alpha_communities(persistance_probabilities)
    best_partition = {}
    for nop in range(2, NUMBER_OF_PARTITIONS + 1):
        if(all(prob > ALPHA for prob in persistance_probabilities[nop].values())):
            best_partition = partitions[nop]
        else: break

    AlphaDf = pd.DataFrame({"Id":best_partition.keys(), "AlphaCommId":best_partition.values()})
    AlphaDf.to_csv("./Data/Euromap_GCC_alphaComm.csv", index=False)
    best_partition, _ = inverse_community_mapping(best_partition)
    print("Best alpha-partition modularity: ", nx_comm.modularity(GCC, best_partition.values()))
    Louvain, _ = inverse_community_mapping(community_louvain.best_partition(GCC))
    print("Louvain modularity: ", nx_comm.modularity(GCC, Louvain.values()))

main()