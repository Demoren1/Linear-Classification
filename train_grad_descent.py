import numpy as np
import shutil
import os

import gradient_descent
import config as cfg


def main():
  clear_results()

  file1 = "datasets/mushrooms.svm"
  test_file = "datasets/mushrooms.svm"

  result_file = open("tmp_results/results.txt", "w")
  result_graph_path = "tmp_results/mushrooms_"

  iteration = 5000
  learning_rate_1 = 0.00001

  features = 112
  shift = 0

  file1_conf = cfg.config(file1, features, shift)
  test_conf = cfg.config(test_file, features, shift)

  data1 = file1_conf.data

  data1 = np.array(data1)
  test_data = np.array(test_conf.data)
  
  # for momentum in range(0, 11):
  # model = linear_classification.LinearClassification(data1, data2, features)
  # graph_path = result_graph_path + str(momentum) + ".png"
  #   momentum /= 10

  accuracy_path = result_graph_path + str("test") + "_accuracy.png"
  norm_path = result_graph_path + str("test") + "_norm.png"

  model = gradient_descent.GradientDescent(data1, features)

  train_model(model, iteration, learning_rate_1)
  print("accuracy is ", model.get_current_accuracy(test_data, 1, accuracy_path))
  print("precision is %g\nrecall is %g\nF1 is %g" % model.get_precision_recall(test_data))
  print("gradient norm is %g\n" % model.get_gradient_norm(1, norm_path))

  result_file.write("accuracy is %g\n" % model.get_current_accuracy(test_data, 0))
  result_file.write("precision is %g\nrecall is %g\nF1 is %g\n" % model.get_precision_recall(test_data))
  result_file.write("gradient norm is %g\n" % model.get_gradient_norm(0))

  result_file.close()

  return 0


def train_model(model : gradient_descent.GradientDescent, iteration, learning_rate):
    for i in range(iteration):
        if i % 1000 == 0:
           print("Iteration = %d" % i)
        model.update_theta(learning_rate)
        model.compute_cost()
    return

def clear_results():
  shutil.rmtree("tmp_results")
  os.makedirs("tmp_results")

main()