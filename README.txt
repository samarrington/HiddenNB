This project is an implementation of the algorithm described in "A Novel Bayes Model: Hidden Naive Bayes" by Jiang, Zhang, and Chai.  This paper, along with others relating to the Hidden Naive Bayes Classifier and other similar classifiers, are included in the references folder. 

Generally speaking, the Hidden Naive Bayes Classifier dampens the strong indepence assumption of the typical Naive Bayes Classifier by adding a hidden parent to each attribute which represents the influence of all other attributes on said attribute.

Files included in repository:

* Hidden_NB.py -- the file which contains the algorithm implemented with a python class.  Running the script at the command line with a *.csv filename as the first argument and a target as the second will output a classification report for both the Multinomial Naive Bayes Classifier and the Hidden Naive Bayes Classifier.
* evaluation.ipynb -- jupyter notebook containing an comparison of the classification ability of the Hidden Naive Bayes Classifier and 
    the Multinomial Naive Bayes Classifier.  The algorithms are applied to the balance-scale dataset which is also contained within the repository. 
* evaluation.html -- the evaluation.ipynb converted to an html file with embedded images.
* references -- folder which contains reference papers related to the Hidden Naive Bayes Classifier and similar classifiers
* balance-scale.data -- the classification dataset used for comparision.  The dataset was optained here: http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/
