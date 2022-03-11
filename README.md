## Data Science Portoflio of ML models used with different datasets (and reading references)

**Practical applications**
* **PCA** based approach to finding relative value trades on Treasury Yield Curve [notebook](https://github.com/uditgt/Data_science_python/blob/main/PCA%20-%20Treasury%20Rates.ipynb)
* **Imbalanced learning** on Customer Attrition dataset (proprietary) with 1:50 imbalance - using techniques from data resampling, cost-sensitive learning, ensemble-learning and anomaly algorithm approaches [notebook](https://github.com/uditgt/Data_science_python/blob/main/Imbalanced%20Learning%20-%20Customer%20Attrition%20Study.ipynb)

**Decision Trees / Random Forest / Ada & Gradient Boosting...**
* Palmer Penguins [notebook](https://github.com/uditgt/Data_science_python/blob/main/DS%20-%2010%20Trees%2C%20Forest%2C%20Boosting.ipynb)
* Banknote Authentication (refer above notebook)
* Rock Density Xray (regression) (refer above notebook)
* Mushroom dataset (refer above notebook)
* Telco Customer Churn [notebook](https://github.com/uditgt/Data_science_python/blob/main/DS%20-%2010%20Telco%20Churn%20Analysis.ipynb)

 **...XGBoost**
* Car Sales Price Prediction [notebook](https://github.com/uditgt/Data_science_python/blob/main/Example%20-%20XGBoost%20-%20Car%20Price%20Prediction.ipynb)
* Santander Customer Transaction Prediction [notebook](https://github.com/uditgt/Data_science_python/blob/main/DS%20-%2011%20XGBoost.ipynb)
* Titanic [notebook](https://github.com/uditgt/Data_science_python/blob/main/Example%20-%20Titanic.ipynb)

**Neural Networks**
* Tweet Classification (bi-directional LSTM) [notebook](https://github.com/uditgt/Data_science_python/blob/main/Example%20-%20LSTM%20-%20Tweet%20Emotions.ipynb)
* Fake News detection [notebook](https://github.com/uditgt/Data_science_python/blob/main/Example%20-%20LSTM%20-%20Fake%20News.ipynb)
* Telco Churn [notebook](https://github.com/uditgt/Data_science_python/blob/main/DS%20-%2013%20Simple%20Neural%20Network.ipynb)
* Customer Attrition (simple NN) - proprietary dataset [notebook](https://github.com/uditgt/Data_science_python/blob/main/Imbalanced%20Learning%20-%20Customer%20Attrition%20Study.ipynb)
 
**SVM classification / regression:**
_includes building pipeline_
* Cement slump dataset [notebook](https://github.com/uditgt/Data_science_python/blob/main/DS%20-%209%20SVM.ipynb)
* Wine Fraud dataset (refer above notebook)

**KNN classification:**
_includes building pipeline_
* Gene expression/ Cancer dataset [notebook](https://github.com/uditgt/Data_science_python/blob/main/DS%20-%208%20KNN%20Classification.ipynb)
* Mine/ Rock dataset (refer above notebook)
 
**Logistic regression and regularization:**
_includes threshold tuning_
* Hearing test [notebook](https://github.com/uditgt/Data_science_python/blob/main/DS%20-%207%20Logistic%20Regression.ipynb)
* Heart disease (refer above notebook)
* Online advertising (refer above notebook)
* IRIS (refer above notebook)
* Titanic (refer above notebook, +XGBoost notebook)

**Linear regression and regularization:** 
* Advertising dataset (from ISLR) [notebook](https://github.com/uditgt/Data_science_python/blob/main/DS%20-%206%20Linear%20Regression.ipynb)
* Ames housing dataset [notebook](https://github.com/uditgt/Data_science_python/blob/main/Example%20-%20Ames%20Housing%20dataset.ipynb)

**NLP / Naive Bayes**
* Large Movie Review dataset [notebook](https://github.com/uditgt/Data_science_python/blob/main/DS%20-%2012%20NLP%20%26%20Naive%20Bayes.ipynb)

**Recommender System**
* Grouplens Movie Rating dataset [notebook](https://github.com/uditgt/Data_science_python/blob/main/Example%20-%20Recommendation%20System.ipynb)

---
**References & Fun reads:**

* XGBoost Tuning - Owen Zhang's [utube](https://www.youtube.com/watch?v=LgLcfZjNF44) and [slides](https://www.slideshare.net/ShangxuanZhang/winning-data-science-competitions-presented-by-owen-zhang), MLM [post](https://machinelearningmastery.com/configure-gradient-boosting-algorithm/), Tianqi Chen [ppt](https://speakerdeck.com/datasciencela/tianqi-chen-xgboost-overview-and-latest-news-la-meetup-talk), [Martin Jullum](https://static1.squarespace.com/static/59f31b56be42d6ba6ad697b2/t/5a72f3ee8165f596c6ec1ee7/1517482994580/Presentatation+BI+lunch+XGBoost.pdf)

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
  * Kaggle winners - [Heritage Provider Network](https://foreverdata.org/1015/content/milestone1-2.pdf)
  * PPTs - 
  * Technical Papers - [separate repo](https://github.com/uditgt/Literature)
