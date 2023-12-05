import numpy as np
import matplotlib.pyplot as plt
import math

class GradientDescent:
    def __init__(self, data : np.ndarray, features):
        features += 1
        self.features = features

        self.y = data[:, 0]
        self.x = data[:, 1:]
        
        self.y = self.y.reshape(self.y.size, 1)

        ones_column = np.ones((self.x.shape[0], 1), dtype=self.x.dtype)
        self.x = np.hstack((ones_column, self.x))
        
        self.norm_constant = self.y.size

        self.theta = np.zeros((features, 1))

        self.cost_list = []
        self.gradients = []

    def find_gradient(self):
        tmp_val = np.dot(self.x, self.theta)
        d_theta_1 = -np.sum(self.y * self.x / (1 + np.exp(self.y * tmp_val) + 1e-15), axis=0)

        return d_theta_1

    def update_theta(self,learning_rate):
        d_theta_1= self.find_gradient()
        d_theta_1 = d_theta_1.reshape(self.features, 1)

        self.theta = self.theta - learning_rate * d_theta_1

        self.gradients.append(d_theta_1)
    

    def compute_cost(self):
        cost = 0
        tmp_val = self.x.dot(self.theta)

        cost = np.sum(np.logaddexp(0, -self.y * tmp_val))
        cost /= self.norm_constant
        self.cost_list.append(cost)

        return cost
    

    def get_current_accuracy(self, data : np.ndarray, graph_flag, save_path = "accuracy_graph.png"):
        characteristic_amount = 0

        theta = self.theta
        
        for line in data:
            actual_class = line[0]
            line = np.insert(line[1:], 0, 1)
            pred_class = line.dot(theta)
            pred_class = np.sign(pred_class)
            # print("true = %d, pred = %g" % (actual_class, pred_class))

            characteristic_amount += (pred_class == actual_class)

        accuracy = characteristic_amount / data.shape[0]

        if graph_flag:
            rng = [np.log(x + 1e-15)  for x in range(1, len(self.cost_list) + 1)]
            plt.rcParams ['figure.figsize'] = [10, 8]
            plt.plot(rng, self.cost_list)
            plt.plot(0,0)
            plt.grid()
            plt.xlabel("ln(Iteration)")
            plt.ylabel("Cost value")
            plt.title("Cost value(ln(Iteration))")
            # plt.show()
            plt.savefig(save_path)
            plt.clf()

        return accuracy

    def get_precision_recall(self, data):
        tp = 0
        tn = 0
        fn = 0
        fp = 0

        theta = self.theta

        for line in data:
            actual_class = line[0]
            line = np.insert(line[1:], 0, 1)
            pred_class = line.dot(theta)
            pred_class = np.sign(pred_class)
            
            if actual_class == 1 and pred_class == 1:
                tp += 1
            elif actual_class == 1 and pred_class == -1:
                fn += 1
            elif actual_class == -1 and pred_class == 1:
                fp += 1
            elif actual_class == -1 and pred_class == -1:
                tn += 1

        precision = 0
        recall = 0
        F_score = 0

        if tp + fp != 0:
            precision = tp / (tp + fp)
        if tp + fn != 0:
            recall = tp / (tp + fn)
        if precision + recall != 0:
            F_score = 2 * precision * recall / (precision + recall)
        return precision, recall, F_score

    def get_gradient_norm(self, graph_flag : bool, save_path = "gradient_norm.png"):
        gradient_norms = 1 / self.norm_constant * np.linalg.norm(self.gradients, axis=1)

        if graph_flag:
            rng = [np.log(x + 1e-15) for x in range(1, len(gradient_norms) + 1)]
            plt.rcParams ['figure.figsize'] = [10, 8]
            plt.plot(rng, gradient_norms)
            plt.plot(0,0)
            plt.grid()
            plt.xlabel("ln(Iteration)")
            plt.ylabel("Gradient norm value")
            plt.title("Gradient norm(ln(Iteration))")
            plt.savefig(save_path)
            plt.clf()
        return gradient_norms[-1]

