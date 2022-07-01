from .RDA import GridSearchRDA
from Voting import GridSearchVoting

class GridSearchCV(object):
    def __init__(self, model, param_grid, metric='accuracy'):
        if self.name == "RDA":
            pass