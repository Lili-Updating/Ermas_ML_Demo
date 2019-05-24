* import data;
proc import datafile="kaggle_fraud_creditcard.csv"
     out=kaggle_fraud
     dbms=csv
     replace;
run;

* preprocess;
proc standard data=kaggle_fraud mean=0 std=1 out=kaggle_fraud_1;
  var Time Amount;
run;

* split training data and test data;
proc sort data=kaggle_fraud_1;
   by class;
run;

proc surveyselect data=kaggle_fraud_1 out=split seed=42 samprate=.7 outall;
strata class;
run;

data training test;
Set split;
if selected = 1 then output training;
else output test;
run;

* train logistic regression classifier & perform predictions under different probability cutoffs;
ods graphics on;
proc logistic Data = training descending;
model class = amount time v1--v28 / ctable pprob=(0 to 0.2 by 0.001);
score data=test out = logit_test fitstat outroc=roc;
run;

