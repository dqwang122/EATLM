from sklearn.metrics import matthews_corrcoef

from paragen.metrics import PairwiseMetric, register_metric


@register_metric
class MatthewsCorr(PairwiseMetric):
    """
    MatthewsCorr evaluates matthews correlation of produced hypotheses labels by comparing with references.
    """

    def __init__(self, is_labeling):
        super().__init__()
        self._is_labeling = is_labeling

    def eval(self):
        """
        Calculate the spearman correlation of produced hypotheses comparing with references
        Returns:
            score (float): evaluation score
        """
        if self._score is not None:
            return self._score
        else:
            if self._is_labeling:
                hypotoken, reftoken = [], []
                for hypo, ref in zip(self.hypos, self.refs):
                    hypotoken.extend(hypo)
                    reftoken.extend(ref)
            else:
                reftoken, hypotoken = self.refs, self.hypos
            reftoken = [1 if r >= 0.5 else 0 for r in reftoken]
            hypotoken = [1 if h >= 0.5 else 0 for h in hypotoken]
            self._score = matthews_corrcoef(reftoken, hypotoken)
        return self._score
