use "/Users/lizzieriese/Desktop/S21/ECO375/Assignment 2/daAssignment2.dta"

cd $work 
*log using "/Users/lizzieriese/Desktop/S21/ECO375/ECO375_A2.log"

use daAssignment2.dta, clear
xtset code_numeric year_numeric
* setting up for xtreg function later
* indicates time 't' and individual 'i' in dataset

* SUMMARY STATISTICS
summarize fhpolrigaug
summarize lrgdpch
summarize lpop
summarize education
summarize age_veryyoung


* REGRESSION 1
reg fhpolrigaug L1.lrgdpch, robust
* regressing the democracy measure on the gdp measure 
* adjustments for the lag and robust standard errors included



* REGRESSION 2
reg fhpolrigaug L1.lrgdpch, cluster(code_numeric)
* clustering the standard errors at the country level  



* REGRESSION 3
xtreg fhpolrigaug L1.lrgdpch ,fe cluster(code_numeric)
* now adding fixed effects for country 



* REGRESSION 4
xtreg fhpolrigaug L1.lrgdpch i.year_numeric ,fe cluster(code_numeric)
* adding the year control as a dummy variable

testparm i.year_numeric
* testing the exclusion of the year fixed effects 



* REGRESSION 5
xtreg fhpolrigaug L1.lrgdpch L1.lpop L1.education L1.age_veryyoung L1.age_young L1.age_old L1.age_veryold i.year_numeric ,fe cluster(code_numeric)
* adding the demographics lagged with the middle age group as the base case

gen demo_sample_0=(e(sample)==1)
* generating the subsample 

xtreg fhpolrigaug L1.lrgdpch i.year_numeric if demo_sample_0==1 ,fe cluster(code_numeric) 
* running the regression specifications 4 on the subsample 



* REGRESSION 6
xtreg fhpolrigaug L1.lrgdpch L1.lpop L1.education L1.age_veryyoung L1.age_young L1.age_old L1.age_veryold i.year_numeric ,fe cluster(code_numeric)
* is this not the same regression as 5 but before the sub sampling

testparm i.year_numeric
* testing the exclusion of year fixed effects 

testparm L1.age_veryyoung L1.age_young L1.age_old L1.age_veryold L1.age_midage
* testing the exclusion of the age variables

testparm L1.lpop L1.education 
* testing the exclusion of all demographic variables


******
* MLR Assumptions
corr fhpolrigaug L1.lrgdpch L1.lpop L1.education L1.age_veryyoung L1.age_young L1.age_old L1.age_very old year_numeric code_numeric
* check for multicollinearity 
******



***EXTENSION ***


* REGRESSION 7 
* looking at non-linear effects of income on democracy
* this one includes demographic controls and tests to see if they're necessary
xtreg fhpolrigaug L1.lrgdpch L1.lrgdpch2 L1.lpop L1.education L1.age_veryyoung L1.age_young L1.age_old L1.age_veryold i.year_numeric ,fe cluster(code_numeric) 
 

testparm i.year_numeric
* testing the exclusion of year fixed effects 

testparm L1.age_veryyoung L1.age_young L1.age_old L1.age_veryold L1.age_midage
* testing the exclusion of the age variables

testparm L1.lpop L1.education 
* testing the exclusion of all demographic variables



* REGRESSION 8 

xtreg fhpolrigaug L1.lrgdpch L1.lrgdpch2 L1.age_veryyoung L1.age_young L1.age_old L1.age_veryold i.year_numeric ,fe cluster(code_numeric) 
* removed the demographic controls because both times they didn't test as
* statistically significant in the 7th regression

testparm i.year_numeric
* testing the exclusion of year fixed effects 

testparm L1.age_veryyoung L1.age_young L1.age_old L1.age_veryold L1.age_midage
* testing the exclusion of the age variables


*log close
