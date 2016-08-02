# Data Sets Overview

### Health inspection results
*DC Restaurant Inspection data*
http://opendatadc.org/dataset/restaurant-inspection-data

Contains information on type of facility being inspected, as well as historical information on health inspections for 3 years.

Data pull complete, organizing/ cleaning data

### Weather (temperature over the previous 3 days):

*ASOS Network*
http://mesonet.agron.iastate.edu/request/download.phtml?network=MD_ASOS

Good for pulling in static historical data; would need more work to automate daily pulls

*Wunderground*
http://wunderground.com

Historical data for previous 3 days can be pulled automatically and dynamically via R (investigate if python is an option)
 
Instructions for R can be found here: http://allthingsr.blogspot.com/2012/04/getting-historical-weather-data-in-r.html

data pull pending

### 311 Service Requests

*DC 311 requests database*
http://opendata.dc.gov/datasets/14faf3d4bfbe4ca4a713bf203a985151_0?geometry=-77.784%2C38.745%2C-76.136%2C39.065&orderByFields=ADDDATE+ASC&filterByExtent=false

Data to look for in this set:  
* Nearby garbage and sanitation complaints  
* Sewer overflow/ line breaks  
* Water line breaks  
* Rodent/ pest complaints  

**Notes:** data is only available for last 30 days. First pull -  5/23/16 1633

historical data has been pulled from here as of 5/30/2016- http://dev.seeclickfix.com/
attempt was made to retrieve data from 4/1/2013 forward. Need to verify data pull. Data currently stored in MongoDB

### Nearby burglaries

*DC Crime Data*
http://opendata.dc.gov/datasets?q=crime+incidents&sort_by=relevance

**Notes:** All available data downloaded 5/25/16 1101 (2012- 5/25)

### Alcohol consumption license 

*DC Alcoholic Beverage Regulation Administration (ABRA)*
http://opendata.dc.gov/datasets/cabe9dcef0b344518c7fae1a3def7de1_5

**Notes:** Downloaded 5/25/16

### Length of time establishment has been operating

*DCRA Business License Data*
http://opendata.dc.gov/datasets?q=basic+business+license&sort_by=relevance

**Notes:** downloaded 5/25/16. Need to explore the utility of the data.


### Permits for residential or commercial construction

*DDOT Tops*
http://opendata.dc.gov/datasets/fc7da7bd29d4493481b17d032e117d09_0

**Notes:** downloaded 5/25/16, last update of records- 5/1/16. Recheck on 6/1

### Age of building

This data is not available for mass downlaod and will be considered optional, pending time availability. 

Look into - Building Permits Database by Brian Kraft. Housed in Washintoniana collect fo DCPL for Histoic data http://dclibrary.org/node/35928

### Yelp

Data initially downloaded 6/1/2016
Needs to be updated once Health Inspection results are ready.
