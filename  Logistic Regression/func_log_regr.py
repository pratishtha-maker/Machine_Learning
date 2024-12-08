import numpy as np

# shuffle before training
def shuffle(x_data, shuffle_count):
    
    shuffled_idx = []

    row = x_data.shape[0]
    for _ in range(shuffle_count):
        shuffled_data = np.arange(row)
        np.random.default_rng().shuffle(shuffled_data)
        shuffled_idx.append(shuffled_data)

    return shuffled_idx
##
class Log_reg_algo():
    def __init__(self, nweights, estimate="ml") -> None:
        if estimate == "ml":
            self.loss = self._get_ml_loss
            self._get_grad = self._get_ml_grad
        elif estimate == "map":
            self.loss = self._get_map_loss
            self._get_grad = self._get_map_grad
        else:
            raise Exception("Sorry, Invalid Input!")
        
        self._weights = np.zeros(nweights)

    
    def train(self, x, y, var=0, d=1, lr=1, T=100):
        m = len(x)
        shuffled = shuffle(x, T)

        for t in range(T):
            lri = self._lr_sch(t, lr, d)
            for i in shuffled[t]:
                xi = x[i]
                yi = y[i]
                
                grad = self._get_grad(xi, yi, m, var)
                self._weights = self._weights - lri * grad



    def _lr_sch(self, t, g0, d):
        return g0 / (1 + g0 / d * t)


    def _get_ml_loss(self, x, y, _):
        return np.log(1 + np.exp(-y * np.dot(self._weights, x)))


    def _get_ml_grad(self, x, y, m, _):
        ywx = y * np.dot(self._weights, x)
        log_derivatives = 1 / (1 + np.exp(ywx))
        return -m * log_derivatives * y * x


    def _get_map_loss(self, x, y, var):
        return np.log(1 + np.exp(-y * np.dot(self._weights, x))) + var * np.dot(self._weights, self._weights)


    def _get_map_grad(self, x, y, m, var):
        ywx = y * np.dot(self._weights, x)
        log_derivatives = 1 + np.exp(ywx)
        return -m * x * y / log_derivatives + self._weights / var


    def get_pred(self, xs):
        predictions = []

        for x in xs:
            predictions.append(self.predict(x))
        
        return predictions


    def predict(self, x):
        return -1 if np.dot(self._weights, x) <= 0 else 1
