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
from datetime import datetime
from collections import defaultdict
from generate import convert_NA
from sklearn.metrics import roc_auc_score
from scipy.misc import logsumexp
'''
The below algorithm currently only works on the 'Microsoft Malware Detection' data set.
To generalize, the algorithm would need to allow greater >2 classes.  In addition, log probilities may need to be used.  
For this problem, that was necessary.

'''
class Hidden_NB():
    
    # Will add parameters to initialization later when making alogirhtm more general
    def __init__(self):
        pass
    
    
    def _set_attributes(self, train, target, key):
        
        # Not all of these may end up being used, remove unnessary at the end
        self.columns = train.columns
        self.key = key
        self.classes = train[target].unique()
        self.target = train[target]
        self.predictors = train.drop(columns=[key, target])
        
        
        self._n = len(train.index)
        self._p_c = (train[target].value_counts()+1.0)/(self._n + 1.0)
    
    
    def fit(self, train, target, keys):
        
        self._set_attributes(train, target, keys)
        predictor_names = product(self.predictors.columns, repeat=2)
        
        # Calculate conditionals for each attribute pair
        self._attribute_pair_conditionals = defaultdict(lambda: defaultdict(dict))
        for Ai, Aj in predictor_names:
            for c in self.classes:
                self._attribute_pair_conditionals[c][Ai][Aj] = self._generate_attribute_pair_conditional(Ai, Aj, c)
        
        # Calculate Weights
        self._calculate_weights()
        
    def predict(self, test):
        return test.drop(columns=[self.key]).apply(self._predict_record, axis=1)
        
    
    def _predict_record(self, record):

        predictions = {}
        
        
        for c in self.classes:
            
            hidden_parents = np.array([self._generate_hidden_parent(record, Ai, ai, c) for Ai, ai in record.items()])
            predictions[c] = np.prod(hidden_parents)
        
        predictions = self._normalize(predictions)
        return predictions['1']
        
        
        
    # Converting some dictionaries to DataFrame with multilevel index may help here
    def _generate_hidden_parent(self, record, Ai, ai, c):
                
        class_conditionals = self._attribute_pair_conditionals[c]
        attribute_conditionals = class_conditionals[Ai]
        
        # More vectorization may improve computational time here
        hidden_parent = 0
        for Aj, aj in record.items():
            conditional = attribute_conditionals[Aj]
            
            if ((aj in conditional.index) and (ai in conditional.columns)):
                hidden_parent += self.weights.loc[Ai,Aj]*attribute_conditionals[Aj].loc[aj, ai]
        
        return hidden_parent
    
    def _normalize(self, predictions):
        
        normalizer = sum(predictions.values())
        
        for c in predictions.keys():
            predictions[c] = predictions[c]/normalizer
            
        return predictions  
    

    
    # Calculates conditional for single pair of attributes for all
    # Looking at P(A1|A2)
    # change to log probabilities if necessary
    def _generate_attribute_pair_conditional(self, Ai, Aj, c):
            
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
        
        CMI = 0
        if (Ai==Aj):
            return CMI
        else:
            for c in self.classes:
                Ai_Aj = self.predictors.loc[self.target==c, [Ai, Aj]]
                CMI = CMI + mutual_info_score(Ai_Aj[Ai].astype('str'), Ai_Aj[Aj].astype('str'))
            
        return CMI
    
    # Symmetry may speed up the process here if computationally expensive
    def _all_conditional_MI(self):
        
        
        all_CMI_array =  [[self._conditional_MI(Ai, Aj) for Ai in self.predictors.columns]
                                                        for Aj in self.predictors.columns]
        all_CMI = pd.DataFrame(all_CMI_array, index=self.predictors.columns, columns=self.predictors.columns)
        
        return all_CMI
    
    def _calculate_weights(self):
        
        all_CMI = self._all_conditional_MI()
        weights = all_CMI.div(all_CMI.sum())
        self.weights = weights
    
    
def main():

    train = pd.read_csv("./Data/train_10000_30.csv", dtype='category', keep_default_na=False)
    test = pd.read_csv("./Data/test_10000_30.csv", dtype ='category', keep_default_na=False)
    
    # Drop time dependent variables high uniques until they can be handled
    if 'Census_FirmwareVersionIdentifier' in train.columns:
        train = train.drop(columns = ['AvSigVersion', 'Census_FirmwareVersionIdentifier'])
        test = test.drop(columns = ['AvSigVersion', 'Census_FirmwareVersionIdentifier']) 
        
        
    train = convert_NA(train, "HasDetections")
    test = convert_NA(test, "HasDetections")
    
    true_values = test["HasDetections"]
    print(true_values)
    test = test.drop(columns=['HasDetections'])
    
    startTime = datetime.now()
    nb = Hidden_NB()
    nb.fit(train, "HasDetections", "MachineIdentifier")
    test['HasDetections'] = nb.predict(test)
    print("Completion time", datetime.now() - startTime)
    #print(test["HasDetections"])
    
    not_na = test['HasDetections'].notna()
    score = roc_auc_score(true_values[not_na].values.astype(int), test.loc[not_na,'HasDetections'].values)
    print(test['HasDetections'])
    print("AUC: ", score)
    
    
    
    
if __name__ == '__main__':
    main()