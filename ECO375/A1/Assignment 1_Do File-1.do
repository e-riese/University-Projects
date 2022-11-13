use "/Users/lizzieriese/Desktop/S21/ECO375/Assignment 1/Data/Assignment01_Data.dta"

* loading the data

twoway (scatter cases_log lagvacc_per)

* looking at initial scatter plot

regress cases_log lagvacc_per

* performing the regression SLR

rvfplot

* looking at residual vs fit plot

regress cases_log lagvacc_per i.age i.week

* doing regression number 2 MLR, it automatically omits base case 0-14 years

rvfplot

* looking at residual vs fit plot for regression 2

regress cases_log lagvacc_per lagvacc_per2 i.age i.week

* doing regression number 3 MLR

rvfplot

* looking at residual vs fit plot for regression 3

regress cases_log i.age lagvacc_0_10 lagvacc_1~20 lagvacc_~100 i.week

* doing regression number 4 MLR

rvfplot

* looking at residual vs fit plot for regression 4

summarize age cases_rate cases_log lagvacc_per lagvacc_per2 lagvacc_0 lagvacc_0_10 lagvacc_10_20 lagvacc_20_100

* statsitical summaries calculated

tab age

* properly checking how many in each age group

tab lagvacc_0 
tab lagvacc_0_10 
tab lagvacc_10_20
tab lagvacc_20_100

* checking frequency for each percentage band

pwcorr lagvacc_per lagvacc_per2 age week
pwcorr age lagvacc_0_10 lagvacc_1~20 lagvacc_~100 week

* checking correlations between independent variables
