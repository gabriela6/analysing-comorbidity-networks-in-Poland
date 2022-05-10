import pandas as pd
import igraph as ig
import textwrap
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import math
from sklearn.utils import resample
from scipy.special import zeta
from scipy.misc import derivative

# tworzenie grafu
def create_graph(plec, choroby_smiec):
    # wczytanie macierzy połączeń
    adj_matrix = pd.read_csv('adj_matrix/adj_matrix_' + plec + '_' + choroby_smiec + '.csv')
    # utworzenie obiektu grafu
    g = ig.Graph.Weighted_Adjacency(adj_matrix.values, mode="undirected")
    # nadanie atrybutów węzłów
    g.vs['label'] = adj_matrix.columns
    g.vs['letter_id'] = [ord(x[0]) - 65 for x in g.vs['label']]
    ill_counts_series = pd.read_csv('ill_counts/ill_counts_' + plec + '_' + choroby_smiec + '.csv', index_col=0,
                                    squeeze=True)
    g.vs['counts'] = ill_counts_series.tolist()
    # usunięcie węzłów o zerowym stopniu
    g = create_subgraph_vs(g, 1)
    return g

# utworzenie grafu o minimalnym stopniu równym min_degree
def create_subgraph_vs(g, min_degree):
    subg_vs = g.vs(_degree_ge=min_degree)
    return g.subgraph(subg_vs)

# utworzenie grafu o minimalnej wadze równej min_weight
def create_subgraph_es(g, min_weight):
    subg_es = g.es(weight_ge=min_weight)
    return g.subgraph_edges(subg_es)

# utworzenie pustych wykresów
def prepare_2_charts(xlabel, ylabel):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(12, 6)
    axis_font = {'size': '14'}

    ax1.set_ylabel(ylabel, **axis_font)

    ax1.set_xlabel(xlabel, **axis_font)
    ax2.set_xlabel(xlabel, **axis_font)

    ax1.set_title('A', **axis_font)
    ax2.set_title('B', **axis_font)
    return fig, ax1, ax2

# utworzenie legendy, zapis i zamknięcie wykresów
def complete_2_charts(ax1, ax2, path):
    plt.sca(ax1)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0))
    plt.sca(ax2)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0))

    plt.savefig(path, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()


# KS_ważone
def KS_weighted(fit):
    # empiryczne cdf
    X, Actual_CDF = fit.cdf()
    # teoretyczne cdf
    Theoretical_CDF = np.unique(fit.power_law.cdf())
    # obliczenie KS ważonego
    a = np.abs(Actual_CDF - Theoretical_CDF)
    b = np.sqrt(Theoretical_CDF * (1 - Theoretical_CDF))
    weight_CDF_diff = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return weight_CDF_diff.max()

# niepewność standardowa rozkład potęgowy dykretny
def sterr_discrete_powerlaw(fit):
    def zeta_xmin(x):
        return zeta(x, fit.xmin)

    return 1 / math.sqrt(len(fit.data) * (
            derivative(zeta_xmin, fit.power_law.alpha, n=2, dx=1e-6) / zeta_xmin(fit.power_law.alpha) - (
            derivative(zeta_xmin, fit.power_law.alpha, n=1, dx=1e-6) / zeta_xmin(fit.power_law.alpha)) ** 2))

# niepewność standardowa rozkład potęgowy z obcięciem dyskretny (zaimplementowany bootstrap)
def sterr_truncated_powerlaw(fit):
    N_samples = 100
    alphas = []
    # procedura bootstrapu
    for i in range(N_samples):
        sample = resample(fit.data, replace=True, n_samples=len(fit.data))
        fit_sample = powerlaw.Fit(sample, xmin=fit.xmin, discrete=True, estimate_discrete=False)
        alpha = fit_sample.truncated_power_law.alpha
        alphas.append(alpha)
    return np.std(alphas, ddof=1)

# rysowanie i zapis rozkładu dla podanej zmiennej wraz z dopasowaniem rozkładu potęgowego
def plt_dist_powerlaw(dg_list, var, title):
    np.seterr(divide='ignore', invalid='ignore')
    # utworzenie pustych wykresów
    fig, ax1, ax2 = prepare_2_charts(var, 'P(' + var + ')')
    # rysowanie wykresów
    def plt_chart(plec, ax, f):
        plt.sca(ax)

        if plec == 'm':
            dg_list_slice = dg_list[0:3]
        elif plec == 'k':
            dg_list_slice = dg_list[3:6]

        for dg in dg_list_slice:
            if var == 'k':
                data = dg.g.vs.degree()
                if (dg.choroby_smiec == "wszyst"):
                    xmin = 2
                else:
                    xmin = 1
            elif var == 'w':
                data = dg.g.es['weight']
                xmin=1
            elif var == 's':
                if (dg.choroby_smiec == "wszyst"):
                    xmin = 2
                else:
                    xmin = 1
                data = dg.g.strength(dg.g.vs.indices, weights=dg.g.es['weight'])
            # dopasowanie
            fit = powerlaw.Fit(data, discrete=True, estimate_discrete=False, xmin=xmin)
            # rysowanie
            powerlaw.plot_pdf(data, color=dg.choroby_color, marker=dg.marker, linestyle='None',
                              label=dg.choroby_smiec_pelna_nazwa)
            fit.power_law.plot_pdf(color=dg.choroby_color, linestyle='-')
            # zapis parametrów
            print(dg.nazwa, file=f)
            print(fit.power_law.alpha, file=f)
            print(sterr_discrete_powerlaw(fit), file=f)
            print(xmin, file=f)
            print(fit.powerlaw.KS(), file=f)
            print(KS_weighted(fit), file=f)

    # otworzenie pliku
    with open(dg_list[0].g_name + '/charts/' + title + '.txt', 'w') as f:
        print("discrete_powerlaw", file=f)
        print("alpha:", file=f)
        print("alpha_err:", file=f)
        print("xmin:", file=f)
        print("KS:", file=f)
        print("KS_weighted:", file=f)

        plt_chart('m', ax1, f)
        plt_chart('k', ax2, f)

    #zamknięcie i zapis wykresów
    complete_2_charts(ax1, ax2, dg_list[0].g_name + '/charts/' + title + '.png')

    np.seterr()


# rysowanie i zapis rozkładu dla podanej zmiennej wraz z dopasowaniem rozkładu potęgowego z obcięciem
def plt_dist_powerlaw_cutoff(dg_list, var, title):
    np.seterr(divide='ignore', invalid='ignore')
    # utworzenie pustych wykresów
    fig, ax1, ax2 = prepare_2_charts(var, 'P(' + var + ')')
    # rysowanie wykresów
    def plt_chart(plec, ax, f):
        plt.sca(ax)

        if plec == 'm':
            dg_list_slice = dg_list[0:3]
        elif plec == 'k':
            dg_list_slice = dg_list[3:6]

        for dg in dg_list_slice:
            if var == 'k':
                data = dg.g.vs.degree()
                if (dg.choroby_smiec == "wszyst"):
                    xmin = 2
                else:
                    xmin = 1
            elif var == 'w':
                data = dg.g.es['weight']
                xmin = 1
            elif var == 's':
                if (dg.choroby_smiec == "wszyst"):
                    xmin = 2
                else:
                    xmin = 1
                data = dg.g.strength(dg.g.vs.indices, weights=dg.g.es['weight'])
            # dopasowanie
            fit = powerlaw.Fit(data, discrete=True, estimate_discrete=False, xmin=xmin)
            # rysowanie
            powerlaw.plot_pdf(data, color=dg.choroby_color, marker=dg.marker, linestyle='None',
                              label=dg.choroby_smiec_pelna_nazwa)
            fit.truncated_power_law.plot_pdf(color=dg.choroby_color, linestyle='-')
            # zapis parametrów
            print(dg.nazwa, file=f)
            print(fit.truncated_power_law.alpha, file=f)
            print(sterr_truncated_powerlaw(fit), file=f)
            print(fit.truncated_power_law.Lambda, file=f)
            print(xmin, file=f)
            print(fit.powerlaw.KS(), file=f)
            print(KS_weighted(fit), file=f)

    # otworzenie pliku
    with open(dg_list[0].g_name + '/charts/' + title + '.txt', 'w') as f:
        print("truncated_powerlaw", file=f)
        print("alpha:", file=f)
        print("alpha_err:", file=f)
        print("Lambda:", file=f)
        print("xmin:", file=f)
        print("KS:", file=f)
        print("KS_weighted:", file=f)

        plt_chart('m', ax1, f)
        plt_chart('k', ax2, f)

    # zamknięcie i zapis wykresów
    complete_2_charts(ax1, ax2, dg_list[0].g_name + '/charts/' + title + '.png')

    np.seterr()


# rysowanie i zapis relacji zmiennej od k
def plt_y_k(dg_list,var,title):
    # utworzenie pustych wykresów
    if var == 'knn':
        ytitle =r'$\langle k \rangle _{nn}$'
    elif var == 'c':
        ytitle ='C(k)'
    elif var == 's':
        ytitle ='s(k)'
    fig, ax1, ax2 = prepare_2_charts('k', ytitle)
    # rysowanie wykresów
    def plt_chart(plec, ax):
        plt.sca(ax)
        # osie logarytmiczne
        plt.yscale('log')
        plt.xscale('log')

        if plec == 'm':
            dg_list_slice = dg_list[0:3]
        elif plec == 'k':
            dg_list_slice = dg_list[3:6]

        for dg in dg_list_slice:
            if var == 'knn':
                g = dg.g
                x = [ii for ii in range(g.maxdegree())]
                y = []
                # wyliczanie uśrednionego średniego stopnia najbliższego sąsiada dla każdej wartości stopnia
                for ii in x:
                    subg_ind = g.vs(_degree_eq=ii).indices
                    n = len(subg_ind)
                    if n > 0:
                        knn = mean([sum(g.degree(g.neighbors(i))) / g.degree(i) for i in subg_ind])
                        y.append(knn)
                    else:
                        y.append(np.nan)
                #rysowanie
                plt.plot(x, y, color=dg.choroby_color, marker='o', linestyle='None',
                         label=dg.choroby_smiec_pelna_nazwa)

            elif var == 'c':
                g = dg.g
                x = []
                y = []
                # wyliczanie średniego współczynnika gronowania dla każdej wartości stopnia > 1 (nie można obliczyć dla k=1)
                for ii in range(2, g.maxdegree()):
                    subg_ind = g.vs(_degree_eq=ii).indices
                    n = len(subg_ind)
                    if n > 0:
                        c = g.transitivity_local_undirected(subg_ind)
                        c = np.nanmean(c)
                        y.append(c)
                        x.append(ii)
                # rysowanie
                plt.plot(x, y, color=dg.choroby_color, marker=dg.marker, linestyle='None',
                         label=dg.choroby_smiec_pelna_nazwa, mfc='none')

            elif var == 's':
                g = dg.g
                iter = [ii for ii in range(g.maxdegree())]
                x = []
                y = []
                # wyliczanie średniej siły dla każdej wartości stopnia
                for ii in iter:
                    subg_ind = g.vs(_degree_eq=ii).indices
                    n = len(subg_ind)
                    if n > 0:
                        s = g.strength(subg_ind, weights='weight')
                        s = mean(s)
                        y.append(s)
                        x.append(ii)
                #średnia waga
                w_avg = mean(g.es['weight'])
                #rysowanie
                plt.plot(x, y, color=dg.choroby_color, marker=dg.marker, linestyle='None',
                         label=dg.choroby_smiec_pelna_nazwa)
                plt.plot(np.arange(1., 1001.), [k * w_avg for k in np.arange(1., 1001.)], linestyle='--',
                         color=dg.choroby_color)
        if var == 'c':
            # rysowanie k^-1
            def func_powerlaw(x, c, alpha):
                return c * (x ** (-alpha))
            plt.plot(np.arange(70., 1001.), [func_powerlaw(i, 50, 1) for i in np.arange(70., 1001.)], linestyle='--',
                     color='black', label='k^-1')

    plt_chart('m', ax1)
    plt_chart('k', ax2)
    # zamknięcie i zapis wykresów
    complete_2_charts(ax1, ax2, dg_list[0].g_name + '/charts/' + title + '.png')

# utworzenie i zapis wykresów podstawowych własności grafu
def analysis_chart(g_list, folder):
    def save_chart(title, y, type="bar"):
        labels = ["wszystkie kody", "bez kodów śmieciowych podstawowych", "bez kodów śmieciowych rozszerzonych"]
        x = np.arange(3)
        width = 0.4
        if type == "bar":
            plt.bar(x - width / 2, y[0:3], width=width, label='mężczyźni', color='#00ffaa')
            plt.bar(x + width / 2, y[3:6], width=width, label='kobiety', color='#cc00cc')
        elif type == "scatter":
            plt.scatter(labels, y[0:3], label='mężczyźni', color='#00ffaa')
            plt.scatter(labels, y[3:6], label='kobiety', color='#cc00cc')

        plt.gca().set(ylabel=title, axisbelow=True)
        plt.xticks(range(3), [textwrap.fill(label, 15) for label in labels], wrap=True)

        plt.grid(axis='y', which='major', linewidth=2)
        plt.grid(axis='y', which='minor')
        plt.tick_params(axis='x', which='minor', bottom=False)
        plt.minorticks_on()

        plt.legend()
        plt.tight_layout()

        plt.savefig(folder + title + '.png', dpi=300)
        plt.show()
        plt.close()

    # liczba krawędzi
    save_chart('liczba krawędzi', [g.ecount() for g in g_list])
    # liczba węzłów
    save_chart('liczba węzłów', [g.vcount() for g in g_list])
    # maksymalny stopień
    save_chart('maksymalny stopień', [g.maxdegree() for g in g_list])
    # średni stopień
    save_chart('średni stopień', [mean(g.degree()) for g in g_list])
    # maksymalna waga
    save_chart('maksymalna waga', [max(g.es['weight']) for g in g_list])
    # współczynnik gronowania
    save_chart('współczynnik gronowania', [g.transitivity_avglocal_undirected() for g in g_list])
    # największy spójny komponent

    ycomp = []
    for g in g_list:
        components = g.clusters()
        size_list = [components.size(components.membership[i]) for i in range(len(components.membership))]
        ycomp.append(max(size_list))
    save_chart('największy spójny komponent', ycomp)
    # współczynnik korelacji liniowej Pearsona
    save_chart('współczynnik korelacji liniowej Pearsona', [g.assortativity_degree(directed=False) for g in g_list])

# analiza utworzonych grafów
def analyse_graphs(dg_list):
    plt_dist_powerlaw(dg_list, 'k', 'rozkład stopni')
    plt_dist_powerlaw(dg_list, 'w', 'rozkład wag')
    plt_dist_powerlaw(dg_list, 's', 'rozkład sił')

    plt_dist_powerlaw_cutoff(dg_list, 'k', 'rozkład stopni cutoff')
    plt_dist_powerlaw_cutoff(dg_list, 'w', 'rozkład wag cutoff')
    plt_dist_powerlaw_cutoff(dg_list, 's', 'rozkład sił cutoff')

    plt_y_k(dg_list, 'knn', 'zależność knn od k')
    plt_y_k(dg_list, 'c', 'zależność c od k')
    plt_y_k(dg_list, 's', 'zależność s od k')

    analysis_chart([dg.g for dg in dg_list], 'graph/analysis_chart/')

#klasa typu grafu
class DiseaseGraph:
    def __init__(self, plec, choroby_smiec):
        self.g = create_graph(plec, choroby_smiec)
        self.plec = plec
        if plec == "m":
            self.plec_pelna_nazwa = "mężczyźni"
            self.color = 'blue'
            if choroby_smiec == "wszyst":
                self.choroby_color = '#00332b'
            elif choroby_smiec == "podst":
                self.choroby_color = '#009969'
            elif choroby_smiec == "roz":
                self.choroby_color = '#1aff9f'
        elif plec == "k":
            self.plec_pelna_nazwa = "kobiety"
            self.color = 'magenta'
            if choroby_smiec == "wszyst":
                self.choroby_color = '#1c0349'
            elif choroby_smiec == "podst":
                self.choroby_color = '#7d0bda'
            elif choroby_smiec == "roz":
                self.choroby_color = '#df3dff'
        self.choroby_smiec = choroby_smiec
        if choroby_smiec == "wszyst":
            self.choroby_smiec_pelna_nazwa = "wszystkie kody"
            self.marker = 'o'
            # self.choroby_color = 'red'
        elif choroby_smiec == "podst":
            self.choroby_smiec_pelna_nazwa = "bez kodów śmieciowych podstawowych"
            # self.marker = '^'
            self.marker = 'o'
            # self.choroby_color = 'orange'
        elif choroby_smiec == "roz":
            self.choroby_smiec_pelna_nazwa = "bez kodów śmieciowych rozszerzonych"
            # self.marker = 's'
            self.marker = 'o'
            # self.choroby_color = 'yellow'
        self.nazwa = plec + '_' + choroby_smiec

        self.g_name = 'graph'

#klasa typu filtrowanego grafu
class FilteredGraph(DiseaseGraph):
    def __init__(self, plec, choroby_smiec, min_weight):
        super().__init__(plec, choroby_smiec)
        self.g = self.create_filtered_graph(min_weight)
        self.g_name = 'filtered_graph'

    def create_filtered_graph(self, min_weight):
        subg = create_subgraph_es(self.g, min_weight)
        return subg

#wczytanie kodów ICD-10
def load_ICD10():
    df = pd.read_csv("icd10_pl.csv", sep=';')  # ICD10.csv dla nazw ang
    return df

# zapis hubów do pliku
def write_hubs_to_file(g, path):
    with open(path, 'w') as f:
        #posortowana lista stopni, bez powtórzeń
        sorted_degree_list = sorted(g.degree(), reverse=True)
        sorted_degree_list = list(dict.fromkeys(sorted_degree_list))
        # wczytanie kodów ICD-10
        df = load_ICD10()
        #wypisywanie hubów
        for i in range(10):
            hub = g.vs.select(_degree=sorted_degree_list[i])
            for h in hub:
                print(sorted_degree_list[i], ";", h['counts'], ";", h['label'], ";",
                      df[df['ICD10_Code'] == h['label']]['WHO_Full_Desc'].iloc[0],
                      file=f)

#utworzenie listy phi
def create_phi_and_J_list(dg):
    if dg.plec == 'm':
        N = 201191  # mezczyzni
    elif dg.plec == 'k':
        N = 185100  # kobiety
    phi_list = []
    J_list = []
    for es in dg.g.es:
        i = dg.g.vs[es.source]
        j = dg.g.vs[es.target]
        Ni = i['counts']
        Nj = j['counts']

        Nij = es['weight']
        Nni = N - Ni
        Nnj = N - Nj
        #obliczenie phi
        phi = (Nij * N - Ni * Nj) / (math.sqrt(Ni * Nni * Nj * Nnj))
        if phi > 0:
            phi_list.append(phi)
            J_list.append(1)
        else:
            phi_list.append(0)
            J_list.append(0)
    return phi_list, J_list

#utworzenie listy skorygowanego phi
def create_phi_tilde_list(g):
    avg_phi_list = []
    # obliczenie średniego phi węzła
    for i in g.vs.indices:
        if sum(g.es.select(_source=i)['J']) > 0:
            avg_phi_list.append(sum(g.es.select(_source=i)['phi']) / sum(g.es.select(_source=i)['J']))
        else:
            avg_phi_list.append(np.nan)
    g.vs['avg_phi'] = avg_phi_list

    phi_tilde_list1 = []
    phi_tilde_list2 = []
    # obliczenie skorygowanego skierowanego phi
    for es in g.es:
        i = g.vs[es.source]
        if i['avg_phi'] != 0:
            phi_tilde = es['phi'] / i['avg_phi']
            phi_tilde_list1.append(phi_tilde)
        else:
            phi_tilde_list1.append(0)

    for es in g.es:
        i = g.vs[es.target]
        if i['avg_phi'] != 0:
            phi_tilde = es['phi'] / i['avg_phi']
            phi_tilde_list2.append(phi_tilde)
        else:
            phi_tilde_list2.append(0)
    # obliczenie skorygowanego phi
    phi_tilde_list = list(np.maximum(phi_tilde_list1, phi_tilde_list2))
    return phi_tilde_list

#utworzenie listy unormowanej wagi:
def create_norm_weight_list(g):
    norm_weight_list = []
    for es in g.es:
        i = g.vs[es.source]
        j = g.vs[es.target]
        Ni = i['counts']
        Nj = j['counts']
        norm_weight_list.append(es["weight"] / (Ni + Nj - es["weight"]))
    return norm_weight_list

def create_multiple_weights_graph(dg_list):

    fg_list = create_fg_list(dg_list, 100)

    for fg in fg_list:
        phi_list, J_list = create_phi_and_J_list(fg)
        fg.g.es['phi'] = phi_list
        fg.g.es['J'] = J_list
        fg.g.es['phi_tilde'] = create_phi_tilde_list(fg.g)
        fg.g.es['norm_weight'] = create_norm_weight_list(fg.g)
    return fg_list

#utworzenie macierzy korelacji
def corr_matrix(mwg):
    import pandas as pd
    import seaborn as sns

    data = {r'$w$': mwg.g.es['weight'] ,
            r'$w_{norm}$': mwg.g.es['norm_weight'] ,
            r'$\phi^*$': mwg.g.es['phi_tilde']
            }

    df = pd.DataFrame(data, columns=['$w$', r'$w_{norm}$',r'$\phi^*$'])

    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True, cmap = sns.cubehelix_palette(as_cmap=True,reverse=True))
    plt.show()

# utworzenie listy obiektów DiseaseGraph
def create_dg_list():
    dg1 = DiseaseGraph("m", "wszyst")
    dg2 = DiseaseGraph("m", "podst")
    dg3 = DiseaseGraph("m", "roz")
    dg4 = DiseaseGraph("k", "wszyst")
    dg5 = DiseaseGraph("k", "podst")
    dg6 = DiseaseGraph("k", "roz")
    dg_list = [dg1, dg2, dg3, dg4, dg5, dg6]
    return dg_list

# utworzenie listy obiektów FilteredGraph
def create_fg_list(dg_list, min_weight):
    fg_list = [FilteredGraph(dg.plec, dg.choroby_smiec, min_weight) for dg in dg_list]
    return fg_list

# zapis grafu w formacie gml
def write_g_to_gml(dg_list):
    for dg in dg_list:
        dg.g.vs['strength'] = dg.g.strength(weights=dg.g.es['weight'])
        dg.g.vs['name_letter'] = [x[0] for x in dg.g.vs['label']]
        dg.g.write_gml('vertex_edge_data/' + dg_list[0].g_name + '/' + dg.nazwa + '_gml.gml')


def main():
    dg_list = create_dg_list()
    analyse_graphs(dg_list)
    write_g_to_gml(dg_list)

    fg_list = create_fg_list(dg_list,min_weight = 100)
    for fg in fg_list:
        write_hubs_to_file(fg.g,"hubs" + fg.nazwa+ ".txt")

    mwg_list = create_multiple_weights_graph(dg_list)
    for mwg in mwg_list:
        corr_matrix(mwg)
    write_g_to_gml(mwg_list)






if __name__ == '__main__':
    main()
