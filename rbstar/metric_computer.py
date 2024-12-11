from rbstar.rb_metrics import RBMetric
from enum import Enum

class Metric(Enum):
    RBP = 'RBP'
    RBO = 'RBO'
    RBA = 'RBA'
    RBR = 'RBR'

class MetricComputer:
    def __init__(self, rb_metric, metric_type):
        self.rb_metric = rb_metric
        self.metric_type = metric_type

    def __call__(self, args):
        """Process a single query"""
        _, qid, obs, ref = args
        self.rb_metric._observation = obs
        self.rb_metric._reference = ref
        
        if self.metric_type == Metric.RBP:
            return qid, self.rb_metric.rb_precision()
        elif self.metric_type == Metric.RBR:
            return qid, self.rb_metric.rb_recall()
        else:  # RBO or RBA
            return qid, (self.rb_metric.rb_overlap() if self.metric_type == Metric.RBO 
                        else self.rb_metric.rb_alignment()) 