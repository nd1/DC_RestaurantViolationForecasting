#Metrics Computation

**On average, the model finds 8 violations 10 day sooner for a 30 day period.**

The basis of this computation is a dataset created for metric evaluation.  The dataset takes inspections for a 3 month period (March - May 2016) and evaluates them as if they were all going to be inspected on March 1, 2016.

The dataset was created as follows:

* Previous violation features were created for all reports
* Original inspection dates are copied to a new feature
* The inspection date is changed to 3/1/16 for all inspections on or after that date
* The time since last report is computed using the updated inspection date
* 3 day average high temperature is changed to the value for 3/1/16
* The data for the 3 month period is pulled using the original inspection date

This data is used for prediction, and the metrics script outputs a csv with the original data, the predicted target, and the confidence value.

DC DOH was performing, on average, just over 9 routine inspections per week based on the pulled dataset. Using this information, the data was evaluated with the assumption 2 routine inspections would be performed per day.

DC DOH data for March is as follows:

* 41 routine inspections performed
* 141 critical violations identified

Using the premise that there were 23 business days in March for inspections and 2 routine inspections would be performed per day, the inspections for the 3 month period were prioritized using the confidence value computed by the model. Dates for inspection were then assigned.

The difference in days for locating a critical violation were summed (233) and the mean calculated (10.13). With the reprioritized inspections, 180 critical violations were identified in March (39 more) at 5 additional locations, giving an average of 7.8 additional violations on average. 