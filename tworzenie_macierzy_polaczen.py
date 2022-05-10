import pandas as pd
import numpy as np


# wczytanie oryginalnej tabeli danych
def load_original_table():
    data = pd.read_csv("Dane.txt", dtype={'id': int, 'karta': 'UInt8',
                                          'plec': 'UInt8', 'wiek_okres': str,
                                          'wiek_o': 'UInt8', 'wiek': 'UInt8',
                                          'miejsce_zgonu': 'UInt8',
                                          'woj': 'UInt8', 'kod_koder': str,
                                          'kod_karta': str, 'wtrna1': str,
                                          'wtrna2': str, 'wtrna3': str,
                                          'wtrna4': str, 'wtrna5': str,
                                          'wtrna6': str, 'wtrna7': str,
                                          'wtrna8': str, 'wtrna9': str,
                                          'wtrna10': str, 'bezpo1': str,
                                          'bezpo2': str, 'bezpo3': str,
                                          'bezpo4': str, 'bezpo5': str,
                                          'bezpo6': str, 'bezpo7': str,
                                          'bezpo8': str}, keep_default_na=False)
    return data

# przetworzenie tabeli
def create_data2(data):
    # wczytanie tylko kart typu 2
    data2 = data.loc[(data['karta'] == 2)]
    data2 = data2.reset_index(drop=True)
    # zmiana 4-znakowego kodu na 3-znakowy
    data2['kod_koder'] = data2['kod_koder'].str.slice(0, 3, 1)
    # usuniêcie pola kod_karta
    data2 = data2.drop('kod_karta', axis=1)
    return data2

# wypisanie iloœci kart
def number_of_all_and_men_and_women(data):
    print("all",len(data))
    print("men",len(data[(data['plec'] == 1)]))
    print("women",len(data[(data['plec'] == 2)]))


class DiseaseCodesType:
    def __init__(self, plec, choroby_smiec):
        self.plec = plec
        if plec == "m":
            self.plec_nr = 1
        elif plec == "k":
            self.plec_nr = 2

        self.choroby_smiec = choroby_smiec
        if choroby_smiec == "wszyst":
            self.lista = 'Wszystkie'
        elif choroby_smiec == "podst":
            self.lista = 'Lista_podstawowa'
        elif choroby_smiec == "roz":
            self.lista = 'Lista_rozszerzona'

    # tworzenie listy unikalnych kodów
    def create_uniq_arr(self, data2):
        #wczytanie tylko kart dla danej p³ci
        df = data2[data2['plec'] == self.plec_nr]
        #sp³aszczenie tabeli i wybranie unikalnych kodów, usuniêcie pustych wpisów ""
        flat_arr = df[
            ['kod_koder', 'wtrna1', 'wtrna2', 'wtrna3',
             'wtrna4', 'wtrna5', 'wtrna6', 'wtrna7',
             'wtrna8', 'wtrna9','wtrna10', 'bezpo1',
             'bezpo2', 'bezpo3', 'bezpo4', 'bezpo5',
             'bezpo6', 'bezpo7', 'bezpo8']].values.ravel()
        uniq_arr = pd.unique(flat_arr)
        index = np.argwhere(uniq_arr == '')
        uniq_arr = np.delete(uniq_arr, index)
        self.uniq_arr = uniq_arr

        # odrzucenie kodów z list kodów œmieciowych
        if self.lista == 'Lista_podstawowa' or self.lista == 'Lista_rozszerzona':
            df_tc = pd.read_csv("kody_smieciowe_3lit.csv", sep=";",
                                dtype={'Lista_podstawowa': str, 'Lista_rozszerzona': str})
            uniq_arr = [x for x in uniq_arr if x not in df_tc[self.lista].unique()]
            uniq_arr = np.array(uniq_arr)
            self.uniq_arr = uniq_arr

    # zapis listy iloœci wyst¹pieñ chorób
    def save_ill_counts(self, data2):
        # tworzenie listy unikalnych kodów w wierszu, usuniêcie pustych wpisów ""
        def create_uniq_row(data2_row):
            uniq_ill = data2_row[8:27].unique()
            index = np.argwhere(uniq_ill == '')
            uniq_ill = np.delete(uniq_ill, index)
            return pd.Series(dict(zip(list(range(8, 27)), uniq_ill)))

        # zastosowanie funkcji create_uniq_row tylko do kart osób o danej p³ci
        data3 = data2[data2['plec'] == self.plec_nr].apply(create_uniq_row, axis=1)
        # sp³aszczenie tabeli
        ill_counts_series = data3.stack().value_counts(dropna=True)
        ill_counts_series = ill_counts_series.reindex(self.uniq_arr)
        # zapis
        ill_counts_series.to_csv('ill_counts/ill_counts_' + self.plec + '_' + self.choroby_smiec + '.csv')

    # utworzenie i zapis macierzy s¹siedztwa
    def create_and_save_adj_matrix(self, data2):
        # utworzenie pustej tabeli
        adj_matrix = pd.DataFrame(columns=self.uniq_arr, index=self.uniq_arr)
        adj_matrix = adj_matrix.fillna(0)
        # wype³nienie macierzy s¹siedztwa dla jednego wiersza bazy kart
        def fill_a_m_row(data2_row, adj_matrix):
            # tworzenie listy unikalnych kodów w wierszu, usuniêcie pustych wpisów ""
            uniq_ill = pd.Series(pd.unique(data2_row[8:27]))
            uniq_ill.drop(uniq_ill.index[uniq_ill == ''], inplace=True)
            # iterowanie po liœcie unikalnych kodów i zwiêkszanie odpowiednich pól macierzy s¹siedztwa
            for j in uniq_ill:
                for i in uniq_ill:
                    if (i != j):
                        if (self.lista == 'Wszystkie'):
                            adj_matrix.at[i, j] = adj_matrix.at[i, j] + 1
                        elif (self.lista == 'Lista_podstawowa' and (
                                i in list(adj_matrix.columns) and j in list(adj_matrix.columns))):
                            adj_matrix.at[i, j] = adj_matrix.at[i, j] + 1
                        elif (self.lista == 'Lista_rozszerzona' and (
                                i in list(adj_matrix.columns) and j in list(adj_matrix.columns))):
                            adj_matrix.at[i, j] = adj_matrix.at[i, j] + 1

        # zastosowanie funkcji fill_a_m_row tylko do kart osób o danej p³ci
        data2[data2['plec'] == self.plec_nr].apply(fill_a_m_row, adj_matrix=adj_matrix, axis=1)
        #zapis
        adj_matrix.to_csv('adj_matrix/adj_matrix_' + self.plec + '_' + self.choroby_smiec + '.csv', index=False)

#utworzenie listy obiektów rodzajów kodów
def create_dt_list():
    dt1 = DiseaseCodesType("m", "wszyst")
    dt2 = DiseaseCodesType("m", "podst")
    dt3 = DiseaseCodesType("m", "roz")
    dt4 = DiseaseCodesType("k", "wszyst")
    dt5 = DiseaseCodesType("k", "podst")
    dt6 = DiseaseCodesType("k", "roz")
    dt_list = [dt1, dt2, dt3, dt4, dt5, dt6]
    return dt_list


def main():
    data = load_original_table()
    data2 = create_data2(data)
    dt_list = create_dt_list()
    for dt in dt_list:
        dt.create_uniq_arr(data2)
        dt.save_ill_counts(data2)
        dt.create_and_save_adj_matrix(data2)




if __name__ == '__main__':
    main()
