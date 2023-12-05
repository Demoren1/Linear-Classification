import numpy as np
import shutil
import os

import linear_classification
import config as cfg


def main():

  file1 = "datasets/a9a2.svm"
  file2 = "datasets/a9a.svm"

  test_file = "datasets/a9a.svm"

  features = 123
  shift = 0

  result_file = open("tmp_results/results.txt", "w")
  result_graph_path = "tmp_results/a9a2_"
  clear_results("tmp_results")

  iteration = 1000
  learning_rate_1 = 0.0001
  learning_rate_2 = 0.0001
  momentum = 1

  file1_conf = cfg.config(file1, features, shift)
  file2_conf = cfg.config(file2, features, shift)
  test_conf = cfg.config(test_file, features, shift)

  data1 = file1_conf.data
  data2 = file2_conf.data

  data1 = np.array(data1)
  data2 = np.array(data2)

  test_data = np.array(test_conf.data)

  accuracy_path = result_graph_path + str(momentum) + "_accuracy.png"
  norm_path = result_graph_path + str(momentum) + "_norm.png"
  model = linear_classification.LinearClassification(data1, data2, momentum, features)
  train_model(model, iteration, learning_rate_1, learning_rate_2)
  print("momentum is", momentum)
  print("accuracy is ", model.get_current_accuracy(test_data, 1, accuracy_path))
  print("precision is %g\nrecall is %g\nF1 is %g" % model.get_precision_recall(test_data))
  print("gradient norm is %g\n" % model.get_gradient_norm(1, norm_path))

  result_file.write("momentum is %g\n" % momentum)
  result_file.write("accuracy is %g\n" % model.get_current_accuracy(test_data, 0))
  result_file.write("precision is %g\nrecall is %g\nF1 is %g\n" % model.get_precision_recall(test_data))
  result_file.write("gradient norm is %g\n" % model.get_gradient_norm(0))

  result_file.close()

  return 0


def train_model(model : linear_classification.LinearClassification, iteration, learning_rate_1, learning_rate_2):
  for i in range(iteration):    
      if i % 100 == 0:
         print("Iteration = %d" % i)     
      model.update_theta(learning_rate_1, learning_rate_2)
      model.compute_cost()
  return

def clear_results(delete_path):
  shutil.rmtree(delete_path)
  os.makedirs(delete_path)

main()