* This is the base implementation of the algorithm that is in need of improvements.  Improvements necessary are listed here as well as under "Issues".

This project is an implementation of the algorithm described in "A Novel Bayes Model: Hidden Naive Bayes" by Jiang, Zhang, and Chai.  This paper, along with others relating to the Hidden Naive Bayes Classifier and other similar classifiers, are included in the references folder. 

Generally speaking, the Hidden Naive Bayes Classifier dampens the strong indepence assumption of the typical Naive Bayes Classifier by adding a hidden parent to each attribute which represents the influence of all other attributes on said attribute.

Files included in repository:

* Hidden_NB.py -- the file which contains the algorithm implemented with a python class.  Running the script at the command line with a *.csv filename as the first argument and a target as the second will output a classification report for both the Multinomial Naive Bayes Classifier and the Hidden Naive Bayes Classifier.
* evaluation.ipynb -- jupyter notebook containing an comparison of the classification ability of the Hidden Naive Bayes Classifier and the Multinomial Naive Bayes Classifier.  The algorithms are applied to the balance-scale dataset which is also contained within the repository. 
* evaluation.html -- the evaluation.ipynb converted to an html file with embedded images.
* references -- folder which contains reference papers related to the Hidden Naive Bayes Classifier and similar classifiers
* balance-scale.data -- the classification dataset used for comparision.  The dataset was optained here: http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/


Future work (listed in issues as well):
* allow for probability of classes to be returned.  Applying the log-exp-sum trick may help here
* look into divide by zero warning; source may be in underlying python functions as no "visible" division occurs on line where warning occurs
* allow for array types other than dataframe to be used as input.  This should only require a few small changes
* General Efficiency Improvements:
    * test other base data structures other than DataFrames in terms of efficiency
    * algorithm structure supports parralel computing which can be leveraged
    * look into switching to rewriting in cython
    
