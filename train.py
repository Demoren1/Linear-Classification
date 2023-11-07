import linear_classification
import numpy as np
import config as cfg

def main():
  file = "datasets/mushrooms.svm"

  iteration = 100
  learning_rate_1 = 0.0001
  learning_rate_2 = 0.0001
  momentum = 0.5

  file_conf = cfg.config(file, 123, 2)
  features = file_conf.features

  general_data = file_conf.data

  data1 = []
  data2 = []

  for vector in general_data:
      if (vector[0] >= 0):
          data1.append(vector)
      else:
          data2.append(vector)

  general_data = np.array(general_data)
  data1 = np.array(data1)
  data2 = np.array(data2)

  model = linear_classification.LinearClassification(data1, data2, features)
  train_model(model, iteration, learning_rate_1, learning_rate_2, momentum)
  # model.show_graphs(momentum)

  print("accuracy is ", model.get_current_accuracy(general_data, 0.5))


def train_model(model, iteration, learning_rate_1, learning_rate_2, momentum):
  for i in range(iteration):
      print(i)
      model.update_theta(learning_rate_1, learning_rate_2, momentum)
      model.compute_cost()
  return

main()