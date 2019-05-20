#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:07:18 2019

@author: sam
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from itertools import product
from collections import defaultdict

class Hidden_NB():
    """
    Hidden_NB implements the Hidden Naive Bayes Classifer from
    "A Novel Bayes Model: Hidden Naive Bayes" by Jiang, Zhang, and Chai. This paper, 
    along with other related papers, are included in the repository under references.
    
    The general premise of the classifier is to dampen the strong independence
    assumptions of the standard Naive Bayes Classifier by including hidden parents
    to each attribute which represent the influence of all the other attributes
    on said attribute.
    """
    
    def __init__(self):
        pass
    
    
    def fit(self, X, y):
        """Trains the algorithm given predictors array X and target array y.
        
        Keyword arguments:
        X -- the array of predictors to use for training
        y -- the target class variable array to use for training
        
        """
        
        
        self._set_attributes(X, y)
                
        # Calculate conditionals p(a_i, a_c, c) for each attribute pair
        # for each class.self.predictors
        # May be able to use another another function to perform this faster
        predictor_names = product(X.columns, repeat=2)
        self._attribute_pair_conditionals = defaultdict(lambda: defaultdict(dict))
        for Ai, Aj in predictor_names:
            for c in self.classes:
                self._attribute_pair_conditionals[c][Ai][Aj] = self._generate_attribute_pair_conditional(Ai, Aj, c)
        
        self._calculate_weights()
        
    def predict(self, X):
        """Predicts the value of target given array of predictors X.  fit method
        must be called on class before predict method.
        
        Keyword arguments:
        X -- the array of predictors used to make prediction
        
        returns: predicted class for each record in X
        """
        
        return X.apply(self._classify_record, axis=1)
    
    def _set_attributes(self, X, y):
        
        self.classes = y.unique()
        self._n = X.shape[0]
        self._p_c = (y.value_counts()+1.0)/(self._n + 1.0)
        self.predictors = X
        self.target = y

    def _classify_record(self, record):
        # Classifies a given record based on log probability
        # In the future, use log-sum-exp trick to allow for probability estimates
        
        classifications = {}
        for c in self.classes:
            hidden_parents = np.array([self._generate_hidden_parent_prob(record, Ai, ai, c) for Ai, ai in record.items()])
            classifications[c] = np.sum(np.log(hidden_parents))+np.log(self._p_c[c])
    
        return pd.Series(classifications).idxmax()


    def _generate_hidden_parent_prob(self, record, Ai, ai, c):
        # Calculates P(a_i|a_hpi, c) for a given record.  
        # *This needs to be optimized as it's currently the most
        # time intensive part of the algorithm
        
        
        class_conditionals = self._attribute_pair_conditionals[c]
        attribute_conditionals = class_conditionals[Ai]
        
        # More vectorization may improve computational time here.  A majority
        # of time spent is on get_item while the 
        hidden_parent = 0
        for Aj, aj in record.items():
            conditional = attribute_conditionals[Aj]
            
            if ((aj in conditional.index) and (ai in conditional.columns)):
                hidden_parent += self.weights.loc[Ai,Aj]*attribute_conditionals[Aj].loc[aj, ai]
        
        return hidden_parent
    

    def _normalize(self, predictions):
        # Will be used during probability estimate implementation
        # Not currently used
        normalizer = sum(predictions.values())
        
        for c in predictions.keys():
            predictions[c] = predictions[c]/normalizer
            
        return predictions  
    

    def _generate_attribute_pair_conditional(self, Ai, Aj, c):
        # Calculates conditional for single pair of attributes for all
        # Looking at P(A1|A2, C)
        
        # Need both for same name as well and efficiency
        Ai_series = self.predictors.loc[self.target==c, Ai]
        Aj_series = self.predictors.loc[self.target==c, Aj]
            
        Ai_Aj = self.predictors.loc[self.target==c, [Ai, Aj]]
            
        # Create smoother for numerator
        smoother = 1.0/Ai_series.nunique() 
            
        if (Ai==Aj):
                
            counts = Ai_series.value_counts()
            smooth_crosstab = pd.DataFrame(0, index=counts.index, columns=counts.index)
                
            # Add diagonal
            for i, count in counts.items():
                smooth_crosstab.loc[i,i] = count
            smooth_crosstab = smooth_crosstab + smoother
                
        else:
            smooth_crosstab = Ai_Aj.groupby([Aj, Ai]).size().unstack(Ai, fill_value=0) + smoother
                
        normalizer = Aj_series.value_counts(sort=False) + 1.0
        conditionals = smooth_crosstab.divide(normalizer, axis=0)
            
        return conditionals
    

    def _conditional_MI(self, Ai, Aj):
        # Calculates conditional mutual information for a single pair of attributes 
        CMI = 0
        if (Ai==Aj):
            return CMI
        else:
            for c in self.classes:
                Ai_Aj = self.predictors.loc[self.target==c, [Ai, Aj]]
                CMI = CMI + mutual_info_score(Ai_Aj[Ai].astype('str'), Ai_Aj[Aj].astype('str'))
            
        return CMI
    

    def _all_conditional_MI(self):
        # Calculates conditional mutual information for all attributes
        
        
        all_CMI_array =  [[self._conditional_MI(Ai, Aj) for Ai in self.predictors.columns]
                                                        for Aj in self.predictors.columns]
        all_CMI = pd.DataFrame(all_CMI_array, index=self.predictors.columns, columns=self.predictors.columns)
        
        return all_CMI
    
    def _calculate_weights(self):
        
        all_CMI = self._all_conditional_MI()
        weights = all_CMI.div(all_CMI.sum())
        self.weights = weights
    
    
def main():
    # The main function will serve as a testing script where a classification
    # report will be outputed for both the Hidden Naive Bayes Classifier
    # and the Multinomial Naive Bayes Classifier on a selected *.csv dataset
    # and target.
    # 
    # First arg -- *.csv datafile filename
    # Second arg -- the target variable
    
    
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import sys
    np.random.seed(53293)
    
    file = sys.argv[1]
    target = sys.argv[2]
    
    # Get data and split data
    data = pd.read_csv(file)
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                         random_state=21)
    
    print("Multinomial Naive Bayes Classification Report")
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_predict = mnb.predict(X_test)
    print(classification_report(y_test, y_predict))
    
    print("Hidden Naive Bayes Classification Report")
    hnb = Hidden_NB()
    hnb.fit(X_train, y_train)
    y_predict = hnb.predict(X_test)
    print(classification_report(y_test, y_predict))
    
if __name__ == '__main__':
    main()