import numpy as np
data = np.array([[1,2,3], [2,4,6], [3,6,9]])# row wise data, 3rd column is output value

def normalize(inputs):
  pass
def initialize_weights(row):
  """
    row: one row of predictor values of the data.
    :return: list (1,n)
  """
  weights = np.random.rand(1,len(row)) # random wts between 0 and 1. Random values uniformly chosen
  return weights


def activate(activation_function, input_sum):
  """
  str activation_function: name of activation function to be used on the input_sum.
  float input_sum: the dot product of inputs and their corresponding weights
  :return: Float value
  """
  if activation_function == "Logistic Regression":
    activated_value = 1/(1 + exp(-input_sum))
  return activated_value


def forward_pass(row, weights, activation_function):
  """
  np.array (1, n) row: one row of predictor values of the data
  np.array (1, n) weights:
  str activation_function: name of activation function to be used on the input_sum.
  :return: Float  value
  """
  #inputs = np.multiply(weights, row)
  # need dot product as need to multiply element wise and then add products to subsequently pass through activation function
  input_sum = np.sum(np.multiply(weights, row))
  activated_output = activate(activation_function, input_sum)
  return activated_output#also the predicted output

def calculate_error(predicted_output, actual_output, type_of_error):
  """
  np.array(m,1) predicted_output: predicted outputs for each data row.
  np.array(m,1) actual_output: actual output for each row of data
  str type_of_error: the kind of error that has to be caclulated -
                      (1) SSE -  Sum of Squared Error
  :return: float
  """
  if type_of_error == "SSE":
    error = np.subtract(predicted_output, actual_output) # difference between predicted and actual outputs
    squared_error = np.square(error) #squaring each error
    sum_of_squares =np.sum(squared_error)
    return sum_of_squares



def backward_pass(actual_output, predicted_output, weights, error, learning_rate):
  """
  float error: the error value of the difference between the predicted and actual output
  float learning_rate: learning rate i.e. rate at which changes are made
  """
  pass
