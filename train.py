import numpy as np
import shutil
import os

import linear_classification
import config as cfg


def main():
  clear_results()

  file1 = "datasets/mushrooms.svm"
  file2 = "datasets/mushrooms2.svm"

  test_file = "datasets/mushrooms_test.svm"

  result_file = open("results/results.txt", "w")
  result_graph_path = "results/mushrooms_"

  iteration = 1000
  learning_rate_1 = 0.0001
  learning_rate_2 = 0.0001
  # momentum = 1

  features = 112
  shift = 0

  file1_conf = cfg.config(file1, features, shift)
  file2_conf = cfg.config(file2, features, shift)
  test_conf = cfg.config(test_file, features, shift)

  data1 = file1_conf.data
  data2 = file2_conf.data

  data1 = np.array(data1)
  data2 = np.array(data2)

  test_data = np.array(test_conf.data)
  
  for momentum in range(0, 11):
    model = linear_classification.LinearClassification(data1, data2, features)
    graph_path = result_graph_path + str(momentum) + ".png"
    momentum /= 10


    train_model(model, iteration, learning_rate_1, learning_rate_2, momentum, test_data)
    print("momentum is", momentum)
    print("accuracy is ", model.get_current_accuracy(test_data, 1, graph_path))
    print("precision is %g\nrecall is %g\nF1 is %g" % model.get_precision_recall(test_data))

    # result_file.write("momentum is %g\n" % momentum)
    # result_file.write("accuracy is %g\n" % model.get_current_accuracy(test_data, 0))
    # result_file.write("precision is %g\nrecall is %g\nF1 is %g\n" % model.get_precision_recall(test_data))

  result_file.close()

  return 0


def train_model(model : linear_classification.LinearClassification, iteration, learning_rate_1, learning_rate_2, momentum, test_data):
  for i in range(iteration):         
      model.update_theta(learning_rate_1, learning_rate_2, momentum)
      model.compute_cost()
  return

def clear_results():
  shutil.rmtree("results")
  os.makedirs("results")

main()