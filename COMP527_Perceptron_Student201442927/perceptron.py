"""Implementation of the Perceptron Algorithm for COMP-527 Assignment."""

import numpy as np

class Dataset():
    
    def __init__(self, filename):
        """
        Initialize dataset by reading in labelled data from file.
        
        Args:
            filename (str): Name of file in local directory.
        """
        
        # open the file
        with open(filename,'r') as f:
            file_data = f.read()

        # split lines
        split_data = file_data.split('\n')
        # remove final empty string
        split_data.pop()
        
        self.size = len(split_data)
        
        self.classifications = []

        # creat list to store data
        data = []
        for i, datum in enumerate(split_data):
            
            # split the data-vector from the class-label
            split = datum.split(',class-')
            class_name = split[1]
            
            if class_name not in self.classifications:
                self.classifications.append(class_name)
            
            # split the elements of the data-vector
            list_of_strings = split[0].split(',')
            
            # include bias as first component of vector
            list_vector = [1]
            
            # convert the elements of the data-vector...
            # ... from text strings to floating-point numbers
            for string in list_of_strings:
                element = float(string)
                list_vector.append(element)
            
            # convert the list of floats to a numpy array vector
            x = np.array(list_vector)
            
            # load the label and vector into the data dict
            data.append({'x': x, 'class_name': class_name})
            
        self.data = data
        self.features = len(data[0]['x']) - 1


class Perceptron:
    
    def __init__(self, positive_label='1', negative_label=False):
        """Initialize Perceptron with given labels.
        
        Args:
            positive_label (str): Class names to give label `+1`
                Default to '1'.
            negative_label (str): Class names to give label `-1`
                Default to False for 1-vs-Rest approach.
        """
        
        self.positive_label = positive_label
        self.negative_label = negative_label
    
    def _label(self, D):
        """Label elements of dataset D, given Perceptron's positive and negative labels.
        
        Args:
            D (Dataset): Dataset in the form of our declared Dataset class.
        """
        
        # store labels in dict
        y = {}
        for i in range(D.size):
            class_name = D.data[i]['class_name']
            
            if class_name == self.positive_label:
                y[i] = 1
            elif self.negative_label:
                if class_name == self.negative_label:
                    y[i] = -1
                else:
                    y[i] = 0
            else:
                y[i] = -1
                
        return y
    
    def train(
        self,
        max_iterations = 10000, 
        max_updates = 100000, 
        D = Dataset('train.data'),
        regularisation_coefficient = 0
        ):
        """Train Perceptron for given number of iterations or updates.
        
        Args:
            max_iterations (int): 
                Maximum number of times to iterate through dataset.
                Defaults to ten thousand.
            max_updates (int): Max times to update Perceptron weights.
                Defaults to one hundred thousand.
            D (Dataset): training dataset.
                Defaults to loading new Dataset from 'train.data' file.
            regularisation_coefficient (float): 
                L^2 regularisation term. Defaults to zero.
        """
        
        # store each iteration of weights in dict
        w = {}
            
        # initialize weight_vector
        w[0] = np.zeros(D.features + 1)
        
        # get labels for D dataset
        y = self._label(D)
        
        k = 0
        updates_after_iteration = {}
        for n in range(max_iterations):
            
            for i in range(D.size):
            
                x = D.data[i]['x']

                if y[i]==0:
                    pass
                
                elif y[i]*(w[k].dot(x)) <= 0:
                    # ie. if incorrectly classified
                    w[k+1] = w[k] + y[i]*x - 2 * regularisation_coefficient * w[k]
                    k = k+1
                    if k == max_updates:
                        break
            
            updates_after_iteration[n] = k        
            if n>0 and updates_after_iteration[n] == updates_after_iteration[n-1]:
                break
            if k >= max_updates:
                break
        
        self.weights = w[k]
        self.updates = k
        self.history = w
        
    def test(self, D = Dataset('test.data'), silence=False):
        """Test given Perceptron on given Test Dataset and return score.
        
        Args:
            D (Dataset):
                Defaults to loading new Dataset from 'test.data' file.
            silence (Boolean): If 'True' then will not print score.
        """
        
        w = self.weights
        
        # get labels for test dataset
        y = self._label(D)
     
        right = 0
        wrong = 0
        succeeds = []
        fails = []
        
        for i in range(D.size):
            
            x = D.data[i]['x']
            if y[i] == 0:
                pass
            elif y[i] * w.dot(x) > 0:
                right += 1
                succeeds.append((i, x, y[i]))
            else:
                wrong += 1
                fails.append((i, x, D.data[i]['class_name']))
        
        self.succeeds = succeeds
        self.fails = fails
        
        self.score = right/(right+wrong) * 100
        if silence == False:
            print(f'Success rate: {self.score}%')
        return 

if __name__ == '__main__':
    print('perceptron.py is running')
    
    print('loading training data')
    train = Dataset('train.data')
    
    print('initializing Perceptrons')
    a = Perceptron('1','2')
    b = Perceptron('2','3')
    c = Perceptron('1','3')
    
    print('training Perceptrons on training data with max_iterations=20')
    a.train(max_iterations=20, D=train)
    b.train(max_iterations=20, D=train)
    c.train(max_iterations=20, D=train)
    
    print('loading testing data')
    test = Dataset('test.data')
    
    print('testing trained Perceptrons on testing data')
    a.test()
    b.test()
    c.test()
