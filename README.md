<p align="center">
<img src="https://github.com/gabriela6/analysing-comorbidity-networks-in-Poland/blob/main/images/graph_men_all_diseases.png" height="200"> <img src="https://github.com/gabriela6/analysing-comorbidity-networks-in-Poland/blob/main/images/graph_women_all_diseases.png" height="200">
</p>

# analysing-comorbidity-networks-in-Poland
The study has been done as an engineering thesis “Analysing comorbidity networks in Poland based on data from death reports“ at Warsaw University of Technology, Faculty of Physics. My thesis supervisor was Dr Anna Chmiel from Physics of Complex Systems Division. 

## Summary
In this study, a weighted comorbidity network in Poland was created and examined based on data from death certificates from 2013. The nodes in the network are diseases, and the weight of the connection is the number of people who have suffered from the two diseases. The networks were analyzed by gender. The impact of garbage codes on the comorbidity network was investigated by analyzing networks with all disease codes and networks with rejected garbage codes from the short and extended lists. Garbage disease codes have been shown to be one of the largest hubs in the comorbidity network, which, due to their inaccuracy, make it difficult to study the relationships between diseases and to isolate clearly defined diseases most conducive to multimorbidity.

For the created networks, their basic properties were calculated, such as the average degree, number of edges, and the greatest common component. Examination of the value of the Pearson linear correlation coefficient and the dependence of the mean degree of the nearest neighbor on the node degree showed the disassortative nature of the network. The dependence of the cluster coefficient on the degree of node for the comorbidity network indicates their hierarchical structure. There is also a correlation between the strength of the node and its degree. The distributions of vertex degrees, connection weights and node strengths as well as their power character were also examined.

The network was visualized and the connections between diseases were analyzed. The differences between the networks for women and men were compared. For networks with rejected garbage codes from the extended lists, the largest clusters are the cardiovascular disease cluster (highly internally cross-linked) and the cancer cluster. The established networks also include a small cluster of diseases of the respiratory system, a cluster of diseases of the digestive system (divided into 2 parts for women), and a cluster of external causes of illness and their consequences for men.

The study also analyzed alternative weights - standardized weight and corrected phi coefficient, allowing for the comparison of the strength of connections regardless of the frequency of diseases. A strong correlation was found between these weights.

## Data source 
The analysis was based on data from the 2013 collection of death certificates from all over Poland. They were kept for the purpose of testing the software that automatically encodes the causes of death[24], and were later used in the "Causes of death in Poland" grant of a Migration Research Center[25]. The dataset is not publicly available. 
The database is a csv file with 387 989 lines, where each line contains data from one death certificate.
The data columns contain the following information used in this work:
- type of death certificate
  - type 1 - children under 1 year of age
  - type 2 - people aged 1 year and more
- the sex of the deceased
  - 1 - male
  - 2 - woman
- age in completed years
- the initial cause of death code as defined by the doctor-coder (four-character ICD-10 code)
- secondary cause of death code (three-character ICD-10 code)
- immediate cause of death code (three-character ICD-10 code)

## Used technologies
- Python 3.8 with libraries: pandas, powerlaw, igraph, Matplotlib, NumPy, SciPy, scikit-learn
- Cytoscape 3.7.0

## Results and Conclusions
Short introduction to the topic with most relevant results and conclusions are presented in powerpoint presentation “prezentacja obrona” included in this repository. Presentation was written in Polish. 

## Thesis
Full thesis text is available in National Repository of Written Diploma Theses https://polon.nauka.gov.pl/orpd/login

## Source code
Source code is located in this repository in two .py files. Comments in code are written in Polish. 

## Sources
[1] The Academy of Medical Sciences. Multimorbidity: a priority for global health research. [Online] [Cited: 18 January 2022.] https://acmedsci.ac.uk/policy/policy-projects/multimorbidity.

[2] Chudasama, Y. V. et al., Patterns of multimorbidity and risk of severe SARS-CoV-2 infection: an observational study in the U.K. BMC Infectious Diseases. 2021, 21(908).

[3] Jak GUS prowadzi statystykę zgonów. GUS. [Online] [Cited: 18 styczeń 2022.] https://stat.gov.pl/obszary-tematyczne/ludnosc/statystyka-przyczyn-zgonow/jak-gus-prowadzi-statystyke-zgonow,8,1.html.

[4] Friendly, M., Andrews, R. J. The radiant diagrams of Florence Nightingale. SORT. 2021, 45(1), pp. 2-4.

[5] Fihel, A. Investigating multiple-cause mortality in Poland. Studia Demograficzne. 2020, 178(2).

[6] Barabási, A.-L., Albert, R. Emergence of scaling in random networks. Science. 1999, 286(5439).

[7] Goh, K. I. et al. The human disease network. Proceedings of the National Academy of Sciences. 2007, 104(21).

[8] Lee, D. S. The implications of human metabolic network topology for disease comorbidity. Proceedings of the National Academy of Sciences. 2008, 105(29).

[9] Chmiel, A., Klimek, P., Thurner, S. Spreading of diseases through comorbidity networks across life and gender. New Journal of Physics. 2014, 16(115013).

[10] Fronczak, P., Fronczak, A. Świat sieci złożonych Od fizyki do Internetu. Warszawa : Wydawnictwo Naukowe PWN, 2009. ISBN 978-8-30-115987-0.

[11] Barrat, A. The architecture of complex weighted networks. PNAS. 2004, 101(11).

[12] Stawińska-Witoszyńska, B., Gałęcki, J., Wasilewski, W. Poradnik szkoleniowy dla lekarzy orzekających o przyczynach zgonów i wystawiających kartę zgonu. Warszawa : Narodowy Instytut Zdrowia Publicznego - Państwowy Zakład Higieny, 2019. pp. 8,9.ISBN 978-83-65870-19-3.

[13] Międzynarodowa Statystyczna Klasyfikacja Chorób i Problemów Zdrowotnych – X Rewizja, Tom I. Centrum Systemów Informacyjnych Ochrony Zdrowia, 2012.

[14] WHO methods and data sources for global causes of death 2000‐2011. Genewa : Department of Health Statistics and Information Systems WHO, 2013. pp. 7,11.

[15] Albert, R., Barabási, A.-L. Statistical mechanics of complex networks. Reviews of Modern Physics. 2002, 74(47).

[16] Ravasz, E. et al. Hierarchical organization of modularity in metabolic networks. Science. 2002, 297(5586), p.1552.

[17] Clauset, A., Shalizi, C. R., Newman, M. E. J. Power-Law Distributions in Empirical Data. SIAM Review. 2009, 51(4).

[18] Myung, I. J. Tutorial on maximum likelihood estimation. Journal of Mathematical Psychology. 2003, 47(1).

[19] Power-law Distributions in Empirical Data. [Online] [Cited: 19 January 2022.] https://aaronclauset.github.io/powerlaws/.

[20] Alstott, J., Bullmore, E., Plenz, D. powerlaw: A Python Package for Analysis of Heavy-Tailed Distributions. PLoS ONE. 2014, 9(4).

[21] Klaus, A., Yu, S., Plenz, D. Statistical Analyses Support Power Law Distributions Found in Neuronal Avalanches. PLoS ONE. 2011, 6(5), p. 8.

[22] The miracle of the bootstrap. The Stats Geek. [Online] [Cited: 19 January 2022.] https://thestatsgeek.com/2013/07/02/the-miracle-of-the-bootstrap/.

[23] Aleta, A., Meloni, S., Moreno, Y. A Multilayer perspective for the analysis of urban transportation systems. Scientific Reports. 2017, 7(44359).

[24] Wasilewski, W. Analiza porównawcza kodowania przyczyn zgonów oprogramowaniem IRIS i metodą manualną. Urząd Statystyczny w Olsztynie. 2013.

[25] Przyczyny zgonu w Polsce. Ośrodek Badań nad Migracjami. [Online] [Cited: 19 January 2022.] http://www.migracje.uw.edu.pl/projects/przyczyny-zgonu-w-polsce/.

[26] yFiles Layout Algorithms. yworks. [Online] [Cited: 19 January 2022.] https://www.yworks.com/products/yfiles-layout-algorithms-for-cytoscape.

[27] Postawy Polaków wobec palenia tytoniu – Raport 2019 r. GUS. [Online] [Cited: 21 January 2021.] https://www.gov.pl/web/gis/postawy-polakow-wobec-palenia-tytoniu--raport-2017.
