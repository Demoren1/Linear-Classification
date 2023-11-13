import numpy as np
import matplotlib.pyplot as plt

class LinearClassification:
    def __init__(self, data : np.ndarray, private_data : np.ndarray, features):
        features += 1
        self.features = features

        self.y = data[:, 0]
        self.x = data[:, 1:]
        self.y_private = private_data[:, 0]
        self.x_private = private_data[:, 1:]
        
        self.y = self.y.reshape(self.y.size, 1)
        self.y_private = self.y_private.reshape(self.y_private.size, 1)

        ones_column = np.ones((self.x.shape[0], 1), dtype=self.x.dtype)
        self.x = np.hstack((ones_column, self.x))

        ones_column = np.ones((self.x_private.shape[0], 1), dtype=self.x_private.dtype)
        self.x_private = np.hstack((ones_column, self.x_private))
        
        self.norm_constant = self.y.size + self.y_private.size

        self.theta = np.zeros((features, 1))
        self.old_thetas = []
        self.old_thetas.append(np.copy(self.theta))

        self.ordinary_steps = [np.zeros((features, 1))]
        self.aggressive_steps = [np.zeros((features, 1))]

        self.cost_list = []

    def find_gradient(self):
        tmp_val = np.dot(self.x, self.theta)
        d_theta_1 = -np.sum(self.y * self.x / (1 + np.exp(self.y * tmp_val)), axis=0)

        tmp_val_private = np.dot(self.x_private, self.theta)
        d_theta_2 = -np.sum(self.y_private * self.x_private / (1 + np.exp(self.y_private * tmp_val_private)), axis=0)

        return d_theta_1, d_theta_2
    

    def update_theta(self,learning_rate_1, learning_rate_2, momentum):
        d_theta_1, d_theta_2 = self.find_gradient()
        d_theta_1 = d_theta_1.reshape(self.features, 1)
        d_theta_2 = d_theta_2.reshape(self.features, 1)


        self.ordinary_steps.append(self.theta - learning_rate_1 * d_theta_1)
        self.aggressive_steps.append(np.array(self.aggressive_steps[-1] - learning_rate_2 * d_theta_2))

        self.theta = momentum * self.aggressive_steps[-1] + (1 - momentum) * self.ordinary_steps[-1]
        
        self.old_thetas.append(np.copy(self.theta))


    def compute_cost(self):
        cost = 0

        tmp_val = self.x.dot(self.theta)
        cost += np.sum(np.log(1 / (1 + np.exp(-self.y * tmp_val))))

        tmp_val_private = self.x_private.dot(self.theta)
        cost += np.sum(np.log(1 / (1 + np.exp(-self.y_private * tmp_val_private))))

        cost /= self.norm_constant

        self.cost_list.append(cost)
        return cost
    

    def get_current_accuracy(self, data : np.ndarray, graph_flag, save_path = "accuracy_graph.png"):
        characteristic_amount = 0

        old_thetas = np.array(self.old_thetas)
        theta = 1 / len(old_thetas) * np.sum(old_thetas, axis=0)

        # print("theta = ", theta)
        
        for line in data:
            actual_class = line[0]
            line = np.insert(line[1:], 0, 1)
            pred_class = line.dot(theta)
            pred_class = np.sign(pred_class)
            # print("true = %d, pred = %g" % (actual_class, pred_class))

            characteristic_amount += (pred_class == actual_class)

        accuracy = characteristic_amount / data.shape[0]

        if graph_flag:
            rng = [x for x in range(len(self.cost_list))]
            plt.rcParams ['figure.figsize'] = [10, 8]
            plt.plot(self.cost_list, rng)
            plt.plot(0,0)
            plt.grid()
            # plt.show()
            plt.savefig(save_path)
            plt.clf()

        return accuracy

    def get_precision_recall(self, data):
        tp = 0
        tn = 0
        fn = 0
        fp = 0

        old_thetas = np.array(self.old_thetas)
        theta = 1 / len(old_thetas) * np.sum(old_thetas, axis=0)

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

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F_score = 2 * precision * recall / (precision + recall)

        return precision, recall, F_score
