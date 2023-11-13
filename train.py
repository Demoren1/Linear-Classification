import linear_classification
import numpy as np
import config as cfg

def main():
  file1 = "datasets/mushrooms.svm"
  file2 = "datasets/mushrooms2.svm"

  test_file = "datasets/mushrooms_test.svm"

  iteration = 0
  learning_rate_1 = 0.0001
  learning_rate_2 = 0.0001
  # momentum = 0.5

  features = 113
  shift = 2

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
    momentum /= 10
    print("momentum is", momentum)

    train_model(model, iteration, learning_rate_1, learning_rate_2, momentum, test_data)
    print("accuracy is ", model.get_current_accuracy(test_data, 1))
    print("precision is %g\nrecall is %g\nF1 is %g" % model.get_precision_recall(test_data))


def train_model(model : linear_classification.LinearClassification, iteration, learning_rate_1, learning_rate_2, momentum, test_data):
  for i in range(iteration):
      # if i % 2 == 0 and i != 0:
      #   print(i)
      #   print("accuracy is ", model.get_current_accuracy(test_data, 0))
      #   print("precision is %g\nrecall is %g\nF1 is %g" % model.get_precision_recall(test_data))
         
      model.update_theta(learning_rate_1, learning_rate_2, momentum)
      model.compute_cost()
  return

main()