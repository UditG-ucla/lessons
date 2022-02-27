## ML models used on different datasets, and reading references

**Practical applications**
* **PCA** based approach to finding relative value trades on Treasury Yield Curve
* **Imbalanced learning** on Customer Attrition dataset (proprietary) with 1:50 imbalance - using techniques from data resampling, cost-sensitive learning, ensemble-learning and anomaly algorithm approaches

**Linear regression and regularization:** 
* Advertising dataset (from ISLR)
* [Ames housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

**Logistic regression and regularization:**
* Hearing test dataset
* [IRIS dataset (for multiclass)](https://www.kaggle.com/uciml/iris)
* [Heart disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
* Online advertising dataset
* [Titanic dataset](https://www.kaggle.com/c/titanic/data)

**KNN classification:**
* Gene expression/ Cancer dataset
* [Mine/ Rock dataset](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29)
  
**SVM classification / regression:**
* [Cement slump dataset](https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test)
* Wine Fraud dataset

**Decision Trees / Random Forest / Ada & Gradient Boosting / XGBoost**
* [Palmer Penguins dataset](https://archive-beta.ics.uci.edu/ml/datasets/palmer+penguins-3)
* [Banknote Authentication dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
* [Rock Density Xray dataset (regression)](https://www.kaggle.com/ahmedmohameddawoud/rock-density-xray/activity)
* [Mushroom dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)
* [Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction/data?select=test.csv)
* [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) - churn analysis notebook

**Neural Network (basic)**
* [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
* Customer Attrition - proprietary dataset

**NLP / Naive Bayes**
* [Large Movie Review dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

**Recommender System**
* [Grouplens Movie Rating dataset](https://grouplens.org/datasets/movielens/)

---
**References & Fun reads:**
* TensorFlow - [playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.88859&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

* Trees
  * [Random Decision Forests (1995, Ho)](https://www4.stat.ncsu.edu/~lu/ST7901/reading%20materials/Ho1995.pdf)
  * [Leo Brieman's website](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)
  * [Stochastic Gradient Boosting (1998, Friedman)](https://jerryfriedman.su.domains/ftp/stobst.pdf)
  * [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting (1996, Schapire et al)](https://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf)
  * [Gentle Introduction to Gradient Boosting (slides)](http://www.chengli.io/tutorials/gradient_boosting.pdf)

* Suppor Vector Machines
  * [Theoretical Foundation of the Potential Function (1964, Aizerman et al)](https://cs.uwaterloo.ca/~y328yu/classics/kernel.pdf)
  * [Training Algorithm for Optimal Margin Classifier (1992, Vapnik et al)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.21.3818&rep=rep1&type=pdf)
  * [Support Vector Networks (1995, Vapnik et al)](http://image.diku.dk/imagecanon/material/cortes_vapnik95.pdf)
  * [Tutorial on SV Regression (2003)](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=4448154647BC7B10C991CEF2236BBA38?doi=10.1.1.114.4288&rep=rep1&type=pdf)

* Imbalanced Learning
  * [Isolation-Based Anomaly Detection (2012, Liu et al.)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkdd11.pdf)

* Misc
  * Imbalanced learning - [different approaches](https://imbalanced-learn.org/stable/auto_examples/applications/plot_impact_imbalanced_classes.html)
  * Cross Validation - [scoring objects](https://scikit-learn.org/stable/modules/model_evaluation.html), [different approaches](https://scikit-learn.org/stable/modules/cross_validation.html)
  * Feature Engineering - [MLM article](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/), [HSE lectures](https://www.coursera.org/learn/competitive-data-science/lecture/1Nh5Q/overview)
 
