 '''Deternmine the average number or routine inspections per week based on histroic data. Result is 9.48101265823'''

import pandas as pd

X = pd.read_csv('./../../data/modeling/combined_dataset/multiple_sources_dataset.csv')

X.insp_date = pd.to_datetime(X.insp_date, coerce=True)

result = (X.groupby([X["insp_date"].dt.year, X["insp_date"].dt.week]).doh_id.count()).mean()

print result
