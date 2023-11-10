import linear_classification
import numpy as np
import config as cfg

def main():
  file1 = "datasets/mushrooms.svm"
  file2 = "datasets/mushrooms2.svm"

  test_file = "datasets/mushrooms_test.svm"

  iteration = 100
  learning_rate_1 = 0.0001
  learning_rate_2 = 0.0001
  momentum = 1

  features = 112
  shift = 2

  file1_conf = cfg.config(file1, features, shift)
  file2_conf = cfg.config(file2, features, shift)
  test_conf = cfg.config(test_file, features, shift)

  data1 = file1_conf.data
  data2 = file2_conf.data

  data1 = np.array(data1)
  data2 = np.array(data2)

  test_data = np.array(test_conf.data)
 
  model = linear_classification.LinearClassification(data1, data2, features)
  train_model(model, iteration, learning_rate_1, learning_rate_2, momentum, test_data)
  # model.show_graphs(momentum)

  print("accuracy is ", model.get_current_accuracy(test_data))


def train_model(model, iteration, learning_rate_1, learning_rate_2, momentum, test_data):
  for i in range(iteration):
      print(i)
      # if i % 50 == 0:
      #   print("accuracy is ", model.get_current_accuracy(test_data))
         
      model.update_theta(learning_rate_1, learning_rate_2, momentum)
      model.compute_cost()
  return

main()