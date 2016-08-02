Modeling: no categories


Starting RFC GridSearchCV

Fitting 12 folds for each of 360 candidates, totalling 4320 fits
[Parallel(n_jobs=-1)]: Done 121 tasks      | elapsed:    8.4s
[Parallel(n_jobs=-1)]: Done 322 tasks      | elapsed:   32.0s
[Parallel(n_jobs=-1)]: Done 572 tasks      | elapsed:   54.0s
[Parallel(n_jobs=-1)]: Done 922 tasks      | elapsed:  1.6min
[Parallel(n_jobs=-1)]: Done 1372 tasks      | elapsed:  2.6min
[Parallel(n_jobs=-1)]: Done 1922 tasks      | elapsed:  3.8min
[Parallel(n_jobs=-1)]: Done 2572 tasks      | elapsed:  5.2min
[Parallel(n_jobs=-1)]: Done 3322 tasks      | elapsed:  6.8min
[Parallel(n_jobs=-1)]: Done 4172 tasks      | elapsed:  8.9min
[Parallel(n_jobs=-1)]: Done 4320 out of 4320 | elapsed:  9.3min finished

Best Estimator for noYelp

RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='entropy', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

Best Score

0.744186046512

Random Forest feature importances

1. insp_badge feature 3 (0.180930)
2. time_diff feature 8 (0.151762)
3. yelp_reviews feature 1 (0.151257)
4. avg_high_temp feature 7 (0.139022)
5. crime_count feature 4 (0.111649)
6. construction_count feature 6 (0.095335)
7. prev_crit_viol feature 9 (0.073616)
8. yelp_rating feature 0 (0.053540)
9. risk feature 2 (0.037239)
10. 311_count feature 5 (0.005649)
['insp_badge', 'time_diff', 'yelp_reviews', 'avg_high_temp', 'crime_count', 'construction_count', 'prev_crit_viol', 'yelp_rating', 'risk']

 data scaled

   yelp_rating  yelp_reviews      risk  insp_badge  crime_count  311_count  \
0     1.013547     -0.223129 -1.438907   -0.766949     0.205311  -0.146306
1     1.837467     -0.363466 -1.438907    0.755635     1.737676  -0.146306
2     1.837467     -0.363466 -1.438907   -0.870386     1.737676  -0.146306
3     1.837467     -0.363466 -1.438907   -0.651101     0.205311  -0.146306
4     0.189627     -0.319611  0.225580   -0.630413     0.205311  -0.146306

   construction_count  avg_high_temp  time_diff  prev_crit_viol
0            0.569727      -0.200818  -1.690167       -0.839478
1            1.045103       0.060054   0.307477        1.504669
2            1.045103      -0.916382  -0.227028        2.509304
3           -0.143337       0.730044   0.593626       -0.839478
4            1.520479       0.505317  -0.270220        0.500035

Starting KNeighborsClassifier GridSearchCV

Fitting 12 folds for each of 160 candidates, totalling 1920 fits
[Parallel(n_jobs=-1)]: Done 628 tasks      | elapsed:    1.3s
[Parallel(n_jobs=-1)]: Done 1920 out of 1920 | elapsed:    3.4s finished

Best Estimator for noYelp

KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=19, p=2,
           weights='uniform')

Best Score

0.720930232558
Score on scaled data 0.73456295108259828


Computing results


 data scaled

Accuracy Score:  0.751
F1 Score:  0.136

Classification Report:

                precision    recall  f1-score   support

Less_crit_viol       0.78      0.95      0.85       158
More_crit_viol       0.33      0.09      0.14        47

   avg / total       0.68      0.75      0.69       205


Confusion matrix, without normalization

[[150   8]
 [ 43   4]]

Saved confusion matrix plot.


Normalized Confusion Matrix

[[ 0.95  0.05]
 [ 0.91  0.09]]

Saved confusion matrix plot.


Saved ROC curve.

Modeling: categories


Starting RFC GridSearchCV

Fitting 12 folds for each of 360 candidates, totalling 4320 fits
[Parallel(n_jobs=-1)]: Done  52 tasks      | elapsed:    2.0s
[Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed:   26.0s
[Parallel(n_jobs=-1)]: Done 476 tasks      | elapsed:   48.8s
[Parallel(n_jobs=-1)]: Done 826 tasks      | elapsed:  1.7min
[Parallel(n_jobs=-1)]: Done 1276 tasks      | elapsed:  2.9min
[Parallel(n_jobs=-1)]: Done 1826 tasks      | elapsed:  4.3min
[Parallel(n_jobs=-1)]: Done 2476 tasks      | elapsed:  5.9min
[Parallel(n_jobs=-1)]: Done 3226 tasks      | elapsed:  7.7min
[Parallel(n_jobs=-1)]: Done 4076 tasks      | elapsed:  9.8min
[Parallel(n_jobs=-1)]: Done 4320 out of 4320 | elapsed: 10.6min finished

Best Estimator for Yelp

RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

Best Score

0.740176423416

Random Forest feature importances

1. insp_badge feature 3 (0.126803)
2. time_diff feature 8 (0.097183)
3. avg_high_temp feature 7 (0.096094)
4. yelp_reviews feature 1 (0.091887)
5. crime_count feature 4 (0.082300)
6. construction_count feature 6 (0.068464)
7. prev_crit_viol feature 9 (0.063620)
8. yelp_rating feature 0 (0.041750)
9. risk feature 2 (0.027447)
10. sandwiches feature 112 (0.012875)
11. tradamerican feature 129 (0.009931)
12. breakfast_brunch feature 28 (0.009832)
13. burgers feature 32 (0.009323)
14. newamerican feature 100 (0.009080)
15. mexican feature 95 (0.009051)
16. bakeries feature 20 (0.008686)
17. coffee feature 44 (0.008251)
18. bars feature 22 (0.007992)
19. delis feature 52 (0.007944)
20. chinese feature 40 (0.007432)
21. latin feature 89 (0.006875)
22. cafes feature 34 (0.006680)
23. italian feature 83 (0.006581)
24. french feature 63 (0.005598)
25. hotdogs feature 77 (0.005399)
26. salad feature 110 (0.005395)
27. tex-mex feature 126 (0.005077)
28. indpak feature 80 (0.005036)
29. mediterranean feature 94 (0.004623)
30. pizza feature 105 (0.004546)
31. bagels feature 19 (0.004279)
32. asianfusion feature 18 (0.004179)
33. hotels feature 78 (0.004100)
34. pubs feature 107 (0.004076)
35. 311_count feature 5 (0.004043)
36. ethiopian feature 59 (0.003660)
37. thai feature 127 (0.003592)
38. vegetarian feature 131 (0.003576)
39. buffets feature 31 (0.003453)
40. lounges feature 93 (0.003293)
41. steak feature 121 (0.003230)
42. greek feature 70 (0.003169)
43. grocery feature 71 (0.003144)
44. peruvian feature 103 (0.003083)
45. cajun feature 36 (0.002958)
46. diners feature 54 (0.002812)
47. sushi feature 122 (0.002730)
48. bbq feature 23 (0.002716)
49. mideastern feature 96 (0.002621)
50. donuts feature 57 (0.002494)
51. wine_bars feature 135 (0.002427)
52. gluten_free feature 67 (0.002305)
53. vietnamese feature 133 (0.002277)
54. social_clubs feature 114 (0.002219)
55. hotdog feature 76 (0.002060)
56. african feature 16 (0.001978)
57. halal feature 73 (0.001972)
58. seafood feature 113 (0.001929)
59. cocktailbars feature 43 (0.001919)
60. hookah_bars feature 75 (0.001904)
61. afghani feature 15 (0.001822)
62. spanish feature 118 (0.001778)
63. belgian feature 26 (0.001769)
64. sportsbars feature 120 (0.001753)
65. chicken_wings feature 39 (0.001672)
66. cafeteria feature 35 (0.001598)
67. gourmet feature 69 (0.001581)
68. Wine & Spirits feature 13 (0.001562)
69. tapasmallplates feature 124 (0.001541)
70. southern feature 117 (0.001535)
71. salvadoran feature 111 (0.001532)
72. icecream feature 79 (0.001529)
73. breweries feature 29 (0.001451)
74. divebars feature 56 (0.001396)
75. gelato feature 65 (0.001365)
76. soup feature 116 (0.001353)
77. german feature 66 (0.001212)
78. cuban feature 49 (0.001212)
79. restaurants feature 109 (0.001166)
80. musicvenues feature 99 (0.001126)
81. pakistani feature 102 (0.001084)
82. tapas feature 123 (0.001058)
83. jazzandblues feature 85 (0.000990)
84. churches feature 42 (0.000971)
85. Sandwiches feature 12 (0.000908)
86. japanese feature 84 (0.000890)
87. lawyers feature 90 (0.000854)
88. desserts feature 53 (0.000849)
89. cookingschools feature 47 (0.000824)
90. comfortfood feature 46 (0.000802)
91. Burgers feature 10 (0.000801)
92. danceclubs feature 51 (0.000795)
93. bangladeshi feature 21 (0.000789)
94. foodtrucks feature 62 (0.000765)
95. landmarks feature 88 (0.000761)
96. turkish feature 130 (0.000758)
97. british feature 30 (0.000748)
98. creperies feature 48 (0.000741)
99. puertorican feature 108 (0.000736)
100. coffeeroasteries feature 45 (0.000721)
101. cheesesteaks feature 38 (0.000678)
102. korean feature 87 (0.000678)
103. Convenience Stores feature 11 (0.000659)
104. soulfood feature 115 (0.000649)
105. gastropubs feature 64 (0.000600)
106. tea feature 125 (0.000599)
107. discountstore feature 55 (0.000544)
108. brasseries feature 27 (0.000539)
109. juicebars feature 86 (0.000530)
110. irish feature 81 (0.000508)
111. gyms feature 72 (0.000504)
112. drugstores feature 58 (0.000496)
113. wholesale_stores feature 134 (0.000495)
114. ethnicmarkets feature 60 (0.000480)
115. falafel feature 61 (0.000431)
116. chocolate feature 41 (0.000422)
117. publicservicesgovt feature 106 (0.000415)
118. lebanese feature 91 (0.000408)
119. sports_clubs feature 119 (0.000398)
120. catering feature 37 (0.000361)
121. adultentertainment feature 14 (0.000333)
122. mini_golf feature 97 (0.000330)
123. cupcakes feature 50 (0.000328)
124. beerbar feature 24 (0.000250)
125. golf feature 68 (0.000242)
126. healthtrainers feature 74 (0.000233)
127. modern_european feature 98 (0.000200)
128. nonprofit feature 101 (0.000197)
129. irish_pubs feature 82 (0.000175)
130. beergardens feature 25 (0.000146)
131. burmese feature 33 (0.000138)
132. libraries feature 92 (0.000134)
133. tobaccoshops feature 128 (0.000071)
134. apartments feature 17 (0.000053)
135. venues feature 132 (0.000024)
136. pianobars feature 104 (0.000000)
['insp_badge', 'time_diff', 'avg_high_temp', 'yelp_reviews', 'crime_count', 'construction_count', 'prev_crit_viol', 'yelp_rating', 'risk', 'sandwiches']

 data scaled

   yelp_rating  yelp_reviews      risk  insp_badge  crime_count  311_count  \
0     1.013547     -0.223129 -1.438907   -0.766949     0.205311  -0.146306
1     1.837467     -0.363466 -1.438907    0.755635     1.737676  -0.146306
2     1.837467     -0.363466 -1.438907   -0.870386     1.737676  -0.146306
3     1.837467     -0.363466 -1.438907   -0.651101     0.205311  -0.146306
4     0.189627     -0.319611  0.225580   -0.630413     0.205311  -0.146306

   construction_count  avg_high_temp  time_diff  prev_crit_viol    ...      \
0            0.569727      -0.200818  -1.690167       -0.839478    ...
1            1.045103       0.060054   0.307477        1.504669    ...
2            1.045103      -0.916382  -0.227028        2.509304    ...
3           -0.143337       0.730044   0.593626       -0.839478    ...
4            1.520479       0.505317  -0.270220        0.500035    ...

   tex-mex  thai  tobaccoshops  tradamerican  turkish  vegetarian  venues  \
0      0.0   0.0           0.0           0.0      0.0         0.0     0.0
1      0.0   0.0           0.0           0.0      0.0         0.0     0.0
2      0.0   0.0           0.0           0.0      0.0         0.0     0.0
3      0.0   0.0           0.0           0.0      0.0         0.0     0.0
4      0.0   0.0           0.0           0.0      0.0         0.0     0.0

   vietnamese  wholesale_stores  wine_bars
0         0.0               0.0        0.0
1         0.0               0.0        0.0
2         0.0               0.0        0.0
3         0.0               0.0        0.0
4         0.0               0.0        0.0

[5 rows x 136 columns]

Starting KNeighborsClassifier GridSearchCV

Fitting 12 folds for each of 160 candidates, totalling 1920 fits
[Parallel(n_jobs=-1)]: Done 720 tasks      | elapsed:    1.3s
[Parallel(n_jobs=-1)]: Done 1920 out of 1920 | elapsed:    2.9s finished

Best Estimator for Yelp

KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=20, p=2,
           weights='uniform')

Best Score

0.720930232558
Score on scaled data 0.73055332798716921


Computing results


 data scaled

Accuracy Score:  0.771
F1 Score:  0.145

Classification Report:

                precision    recall  f1-score   support

Less_crit_viol       0.78      0.97      0.87       158
More_crit_viol       0.50      0.09      0.15        47

   avg / total       0.72      0.77      0.70       205


Confusion matrix, without normalization

[[154   4]
 [ 43   4]]

Saved confusion matrix plot.


Normalized Confusion Matrix

[[ 0.97  0.03]
 [ 0.91  0.09]]

Saved confusion matrix plot.


Saved ROC curve.