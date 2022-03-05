import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Linear(LinearRegression):
    # Inicializar clase (metodo super())
    
    def __init__(self):
        super().__init__()   
    
    # Agregar metodo de cálculo de error
    def MSE(self, X, y_true):
        y_pred = self.predict(X)
        return mean_squared_error(y_true, y_pred)
    
    # Agregar metodo de plot de visualización de la linea
    def plot_line(self, X, y):
        plt.scatter(X,y)
        plt.plot(X, self.coef_*X + self.intercept_, c='r')
    
    # Agregar metodo de visualización de predicción
    def see_Predict(self, X, y_true):
        y_predict = self.predict(X)
        
        plt.figure(figsize=(10,7))
        plt.scatter(X,y_predict, c='b', marker="^", edgecolors='b')
        plt.scatter(X,y_true, c='g')
        plt.plot(X, self.coef_*X + self.intercept_, c='r')
    
    def residual(self, X, y_true):
        y_predict = self.predict(X)
        residuals = y_predict - y_true
        
        fig, ax = plt.subplots(1,2, figsize=(12,7))
        ax[0].hist(residuals, bins=60)
        ax[1].scatter(y_predict, y_true)

class NormalEquation():
    
    def __init__(self):
        self.w = 0
    
    def fit(self, X, y, bias=True):
        """
        Calculate the slope and intercept of a linear model, through the normal equation.
        
            Parameters:
                X: numpy.ndarray of dimensions (#, )
                    Features of our data
                    
                y: numpy.ndarray of dimensions (#, )
                    Results of our data   
        
        """
        X = self.checkArray(X)
        y = self.checkArray(y)
        
        if bias:
            X = self.biasing(X)
        
        t = np.linalg.pinv(X.T.dot(X))
        t2 = y.T.dot(X)
        self.w = t.dot(t2)
    
    @classmethod
    def biasing(self, array):
        
        array = np.c_[array, np.ones(array.shape[0])]
        return array
        
    
    @classmethod
    def checkArray(self, array):
        
        if not isinstance(array, (np.ndarray, np.generic)):
            array = np.array(array)
        return array
        
    
    def MSE(self, X, y_true):
        y_pred = self.predict(X)
        return mean_squared_error(y_true, y_pred)
    
    def plot(self, X, y, pred=False):
        
        plt.scatter(X,y)
        if pred:
            plt.plot(X, y,c='r')
        else:
            plt.plot(X,self.w[0]*X + self.w[1],c='r')
        
    
    def predict(self, X):
        
        y_pred = self.w[0]*X + self.w[1]
        return y_pred