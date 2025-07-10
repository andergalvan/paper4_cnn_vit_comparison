# Reference: https://github.com/ma-xu/Open-Set-Recognition/blob/master/Utils/evaluation.py

import numpy as np

from sklearn.metrics import f1_score, classification_report, precision_score, recall_score


class Evaluation(object):

    def __init__(self, predict, label, target_names):
        
        self.predict = predict
        self.label = label
        self.target_names = target_names

        self.accuracy = self._accuracy()
        self.f1_micro = self._f1_micro()
        self.f1_macro = self._f1_macro()
        self.f1_weighted = self._f1_weighted()
        self.precision_micro = self._precision(average='micro')
        self.precision_macro = self._precision(average='macro')
        self.precision_weighted = self._precision(average='weighted')
        self.recall_micro = self._recall(average='micro')
        self.recall_macro = self._recall(average='macro')
        self.recall_weighted = self._recall(average='weighted')
        self.class_report = self._classification_report()


    def _accuracy(self) -> float:
        
        """Returns the accuracy score of the labels and predictions."""

        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct)/float(len(self.predict))

    def _f1_micro(self) -> float:

        """ Returns the F1-measure with a micro average of the labels and predictions."""

        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='micro', zero_division=0.0)

    def _f1_macro(self) -> float:

        """ Returns the F1-measure with a macro average of the labels and predictions. """

        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='macro', zero_division=0.0)

    def _f1_weighted(self) -> float:
        
        """ Returns the F1-measure with a weighted macro average of the labels and predictions. """

        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='weighted', zero_division=0.0)
    
    def _precision(self, average) -> float:
        
        """ Returns the precision score for the label and predictions. Observes the average type. """
        
        assert len(self.label) == len(self.predict)
        return precision_score(self.label, self.predict, average=average, zero_division=0.0)

    def _recall(self, average) -> float:
        
        """ Returns the recall score for the label and predictions. Observes the average type. """
        
        assert len(self.label) == len(self.predict)
        return recall_score(self.label, self.predict, average=average, zero_division=0.0)

    
    def _classification_report(self):

        """ Returns the classification report. """

        assert len(self.predict) == len(self.label)
        report = classification_report(self.label, self.predict, target_names=self.target_names, zero_division=0.0)
        return report