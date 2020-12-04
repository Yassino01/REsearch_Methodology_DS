class OnSubsetWrapper:
    """ 
        Prend en paramètre un modèle et des indices de variables 
        et lors de l'apprentissage entraîne ce modèle que sur les ves variable spécifiées. 
    """

    def __init__(self, model, var_ind):
        self.model = model
        self.var_ind = var_ind.copy()
    
    def fit(self, X, y):
        new_X = X[:, self.var_ind] # on ne prend que les variables spécifiées
        self.model.fit( new_X, y )

    def score(self, X, y):
        new_X = X[:, self.var_ind] # on ne prend que les variables spécifiées
        return self.model.score( new_X, y )
    
    def predict(self, X):
        new_X = X[:, self.var_ind] # on ne prend que les variables spécifiées
        return self.model.predict(new_X)
    