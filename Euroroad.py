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
import cpnet
import infomap
import math

DIGIT_PRECISION = 3
TOP_NODES = 7
NUMBER_OF_PARTITIONS = 25
ALPHA = 0.85
REMOVALS_PER_ITER = 1
ROBUSTNESS_SIM_LIMIT = 5

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
    stats["Assortativity coefficient"] = round(nx.degree_assortativity_coefficient(graph),DIGIT_PRECISION)
    
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
    print(dict(zip(deg,cs)))
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

def mean_first_passage_time(adjacency):
    P = np.linalg.solve(np.diag(np.sum(adjacency, axis=1)), adjacency)

    n = len(P)
    D, V = np.linalg.eig(P.T)

    aux = np.abs(D - 1)
    index = np.where(aux == aux.min())[0]

    if aux[index] > 10e-3:
        raise ValueError("Cannot find eigenvalue of 1. Minimum eigenvalue " +
                         "value is {0}. Tolerance was ".format(aux[index]+1) +
                         "set at 10e-3.")

    w = V[:, index].T
    w = w / np.sum(w)

    W = np.real(np.repeat(w, n, 0))
    I = np.eye(n)

    Z = np.linalg.inv(I - P + W)

    mfpt = (np.repeat(np.atleast_2d(np.diag(Z)), n, 0) - Z) / W

    return mfpt

def hierarchial_divisive_clustering(graph, distance_matrix, t_max):
    partitions = {}

    # Appy divisive clustering
    Z = linkage(distance_matrix, 'ward')
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
    plt.ylabel('Persistence prob.', fontweight ='bold')
    plt.legend(loc="lower right")
    plt.ylim([min(ALPHA-0.1,max(min(prob[NUMBER_OF_PARTITIONS].values())-0.1,0)),1])
    plt.tight_layout()
    plt.savefig("./Images/AlphaCommPlot.png")
    plt.show()

def compute_k_core_decomposition(graph):
    core_numbers = nx.algorithms.core.core_number(graph)
    kcoredf = pd.DataFrame({"Id":core_numbers.keys(), "kShellId":core_numbers.values()})
    kcoredf.to_csv("./Data/Euromap_GCC_kcoreDecomposition.csv", index=False)

def simulate_attacks_failures(graph):
    G = graph.copy()
    N = G.number_of_nodes()
    size_ratio_att_b, size_ratio_att_d, size_ratio_fai  = {}, {}, {}
    efficiency_ratio_att_b, efficiency_ratio_att_d, efficiency_ratio_fai = {}, {}, {}

    #Failures simulation
    removed_nodes = 0
    while(G.number_of_nodes() > ROBUSTNESS_SIM_LIMIT):
        if(G.number_of_nodes() == 0): S = 0
        else: S = len(max(nx.connected_components(G), key=len))

        size_ratio_fai[removed_nodes/N] =  S / N
        efficiency_ratio_fai[removed_nodes/N] = nx.global_efficiency(G)

        for i in range(REMOVALS_PER_ITER):
            if(G.number_of_nodes() == 0): break
            G.remove_node(random.choice(list(G.nodes)))
            removed_nodes+=1
                    
    G = graph.copy() #restore graph

    #Attacks simulation degree
    removed_nodes = 0
    degrees = dict(G.degree)
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True) #ordered list by nodes degree
    while(G.number_of_nodes() > ROBUSTNESS_SIM_LIMIT):
        if(G.number_of_nodes() == 0): S = 0
        else: S = len(max(nx.connected_components(G), key=len))
        
        size_ratio_att_d[removed_nodes/N] =  S / N
        efficiency_ratio_att_d[removed_nodes/N] = nx.global_efficiency(G)
        
        for i in range(REMOVALS_PER_ITER):
            if(G.number_of_nodes() == 0): break
            G.remove_node(sorted_nodes.pop(0))
            removed_nodes+=1

    G = graph.copy() #restore graph
    removed_nodes = 0
    betwennesses = dict(nx.betweenness_centrality(G))
    sorted_nodes = sorted(betwennesses, key=betwennesses.get, reverse=True) #ordered list by nodes degree

    #Attacks simulation betweennes
    while(G.number_of_nodes() > ROBUSTNESS_SIM_LIMIT):
        if(G.number_of_nodes() == 0): S = 0
        else: S = len(max(nx.connected_components(G), key=len))
        
        size_ratio_att_b[removed_nodes/N] =  S / N
        efficiency_ratio_att_b[removed_nodes/N] = nx.global_efficiency(G)
        
        for i in range(REMOVALS_PER_ITER):
            if(G.number_of_nodes() == 0): break
            G.remove_node(sorted_nodes.pop(0))
            removed_nodes+=1

    fig, ax = plt.subplots()
    ax.plot(size_ratio_fai.keys(), size_ratio_fai.values(), color="blue", label="Failures")
    ax.plot(size_ratio_att_d.keys(), size_ratio_att_d.values(), color="red", label="Attacks - degree")
    ax.plot(size_ratio_att_b.keys(), size_ratio_att_b.values(), color="green", label="Attacks - betweennes")
    ax.set_xlabel("Removed nodes")
    ax.set_ylabel("S / N")
    ax.legend()
    plt.savefig("./Images/SN_ratio_robustness.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(efficiency_ratio_fai.keys(), efficiency_ratio_fai.values(), color='blue', label="Failures")
    ax.plot(efficiency_ratio_att_d.keys(), efficiency_ratio_att_d.values(), color='red', label="Attacks - degree")
    ax.plot(efficiency_ratio_att_b.keys(), efficiency_ratio_att_b.values(), color='green', label="Attacks - betweennes")
    ax.set_xlabel("Removed nodes")
    ax.set_ylabel("Efficiency")
    ax.legend()
    plt.savefig("./Images/Efficiency_ratio_robustness.png")
    plt.show()

def core_periphery_profile(graph):
    alg = cpnet.Rossa() # Load the Borgatti-Everett algorithm
    alg.detect(graph) # Feed the network as an input
    corness = alg.get_coreness()  # Get the coreness of nodes
    c = alg.get_pair_id()  # Get the group membership of nodes

    AlphaDf = pd.DataFrame({"Id":corness.keys(), "CornessCPP":corness.values()})
    AlphaDf.to_csv("./Data/Euromap_CorePeripheryProfile.csv", index=False)

    
    cmap = plt.cm.plasma

    vmin = min(corness.values())
    vmax = max(corness.values())
    norm = plt.Normalize(vmin, vmax)

    sortedx = sorted(corness, key=corness.get, reverse=True)

    n = len(graph.nodes())
    r_max = math.sqrt(n)

    # Definisce il parametro phi della spirale
    phi = math.pi / 24
    spiral = {}
    # Assegna a ogni nodo la sua posizione sulla spirale di Archimede
    for i, node in enumerate(sortedx):
        r = math.sqrt(i + 1)  # Aggiungi 1 per evitare la radice quadrata di 0
        theta = i * phi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        spiral[node]= (x, y)  # Salva la posizione come attributo del nodo

    # Disegna il grafo con le posizioni dei nodi
    nx.draw(graph, pos=spiral, with_labels=False, node_color=[cmap(norm(corness[node])) for node in graph.nodes()],
            cmap=cmap, node_size = 30, edge_color="#ccc")

    for node in dict(Counter(dict(graph.degree())).most_common(TOP_NODES)).keys():
        x, y = spiral[node]
        plt.text(x, y, node, ha="center", va="bottom", fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, shrink=0.7)
    plt.tight_layout()
    plt.savefig("./Images/Core_periphery_profile.png")
    plt.show()

def compute_infomap(graph):
    im = infomap.Infomap("--two-level")
    im.add_networkx_graph(graph)
    im.run()

    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    infomap_com = {}
    for node in im.nodes:
        infomap_com[node.node_id] = node.module_id

    names = pd.read_csv("./Data/Euromap_labels.csv", usecols=["city_name"])

    named = pd.DataFrame(columns=["Id", "ModuleId"])
    for index, row in names.iterrows():
        if((index + 1) in infomap_com):
            newrow = {'Id': names['city_name'].iloc[index], 'ModuleId': infomap_com[index + 1]}
            named = named.append(newrow, ignore_index=True)

    named.to_csv("./Data/Euromap_GCC_infomap.csv", index=False)
    print("Infomap modularity ", nx_comm.modularity(graph, inverse_community_mapping(infomap_com)[0].values()))
    print("Infomap avg. conductance:", average_conductance(graph, inverse_community_mapping(infomap_com)[0]))
    
def average_conductance(graph, partition):
    avg_conductance = 0
    for c in partition.values():
        print(avg_conductance)
        avg_conductance += nx.conductance(graph, c)
    
    avg_conductance = avg_conductance/len(partition.values())
    print(len(partition.values()))
    return avg_conductance

def main():
    # Load the data
    nodesdata = pd.read_csv("./Data/Euromap_node_data.csv", usecols=["city_name", "lat", "long"])
    edges = pd.read_csv("./Data/Euromap_named_edges.csv")
    unamed_edges = pd.read_csv("./Data/Euromap_edges.csv")
    graph = nx.from_pandas_edgelist(edges, source = 'Source', target = 'Target')
    unamed_graph = nx.from_pandas_edgelist(unamed_edges, source = 'from', target = 'to')

    sns.set()

    # Plot the graph
    #plot_graph_on_map(graph, nodesdata)
    #plot_conn_components(graph)

    # Retrieve only the giant component
    GCC = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
    df = nx.to_pandas_edgelist(GCC)
    df.to_csv("./Data/Euromap_GCC.csv", index=False)

    unamed_GCC = unamed_graph.subgraph(sorted(nx.connected_components(unamed_graph), key=len, reverse=True)[0])

    plot_stats(GCC)
    #plot_rankings(GCC)
    #plot_degree_distribution(GCC)

    # Core-Periphery analysis
    #compute_k_core_decomposition(GCC)
    #core_periphery_profile(GCC)

    # Failures attack analysis
    #simulate_attacks_failures(GCC)

    #Compute infomap
    #compute_infomap(unamed_GCC)

    # Alpha partition analysis
    persistance_probabilities = {}
    #distances = mean_first_passage_time(nx.to_numpy_array(GCC))
    distances = squareform(nx.floyd_warshall_numpy(GCC))
    partitions = hierarchial_divisive_clustering(GCC, distances, NUMBER_OF_PARTITIONS)
    for nop in range(2, NUMBER_OF_PARTITIONS + 1):
        persistance_probabilities[nop] = compute_persistance_probabilities(GCC,partitions[nop])

    plot_alpha_communities(persistance_probabilities)
    best_partition = {}
    for nop in range(2, NUMBER_OF_PARTITIONS + 1):
        if(all(prob > ALPHA for prob in persistance_probabilities[nop].values())):
            best_partition = partitions[nop]
        else: break
        print(nop," : ",min(persistance_probabilities[nop].values()))

    AlphaDf = pd.DataFrame({"Id":best_partition.keys(), "AlphaCommId":best_partition.values()})
    AlphaDf.to_csv("./Data/Euromap_GCC_alphaComm.csv", index=False)
    best_partition, _ = inverse_community_mapping(best_partition)
    print("Best alpha-partition modularity (q = ", len(best_partition) ,"): ", nx_comm.modularity(GCC, best_partition.values()))
    print("Best alpha-partition avg. conductance: ", average_conductance(GCC, best_partition))
    Louvain, _ = inverse_community_mapping(community_louvain.best_partition(GCC))
    print("Louvain modularity: ", nx_comm.modularity(GCC, Louvain.values()))
    print("Louvain avg. conductance: ", average_conductance(GCC, Louvain))

main()