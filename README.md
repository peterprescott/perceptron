# COMP527 Data Mining: Perceptron Assignment.

In this zipfile you should find:
- this README.md text file.
- a PDF providing the answers to the questions.
- a Python program called `perceptron.py`.
- and I have also included the data files: `train.data`, `test.data`, and an experimental `random.data`.

To run the program you need to have NumPy installed in your Python environment.

If you run the file directly from the command line, you should see the following:
```
$ python perceptron.py

perceptron.py is running
loading training data
initializing Perceptrons
training Perceptrons on training data with max_iterations=20
loading testing data
testing trained Perceptrons on testing data
Success rate: 100.0%
Success rate: 50.0%
Success rate: 100.0%
```

If run directly the program automatically trains and tests Perceptrons for each binary case: '1' vs. '2', '2' vs. '3', and '1' vs. '3'.

Or you can import the program from the Python interpreter and play with it as you like:
```
$ python

>>> from perceptron import *
>>> p = Perceptron()
>>> p.train()
>>> p.test()
Success rate: 100.0%
```

The Perceptron functions are set with defaults to make instant interactivity as simple as possible. But you can engage with these explicitly however you like.

There are three relevant functions: `.__init__()` (which is called when you first initialize the `Perceptron()`), `.train()`, and `.test()`

`.__init__()` takes two arguments:
            `positive_label` (str): Class names to give label `+1`
                Default to '1'.
            `negative_label` (str): Class names to give label `-1`
                Default to False for 1-vs-Rest approach.

`.train()` can take four arguments:
            `max_iterations` (int): 
                Maximum number of times to iterate through dataset.
                Defaults to ten thousand.
            `max_updates` (int): Max times to update Perceptron weights.
                Defaults to one hundred thousand.
            `D` (Dataset): training dataset.
                Defaults to loading new Dataset from 'train.data' file.
            `regularisation_coefficient` (float): 
                L^2 regularisation term. Defaults to zero.                

`.test()` can take two arguments:
            `D` (Dataset): 
                Defaults to loading new Dataset from 'test.data' file.
            `silence` (Boolean): If 'True' then will not print score.

Once we have initialized our Perceptron as say `p = Perceptron()` we can check `p.positive_label` and `p.negative_label`.

Once we have trained it `p.train()` we can check `p.weights` and `p.updates`, or if we really want, the full `p.history`.

And once we have tested it `p.test()`, we can see `p.score` as well as precisely which elements were correctly classified or not with `p.succeeds` and `p.fails`.

```
>>> binary_classifier = Perceptron('2','3')
>>> binary_classifier.train(max_iterations = 50)
>>> binary_classifier.updates
100
>>> binary_classifier.train(max_updates = 100)
>>> binary_classifier.updates
100
>>> binary_classifier.test()
Success rate: 70.0%

>>> one_vs_rest = Perceptron('2')
>>> one_vs_rest.train(regularisation_coefficient=0.1)
>>> one_vs_rest.weights
array([-0.55555556, -1.94444444, -2.05555556, -6.22222222, -3.83333333])
>>> one_vs_rest.test()
Success rate: 66.66666666666666%
```

The `perceptron.py` program also includes a Dataset object to load the datafiles. 

Once we have initialized our Dataset as ay `D = Dataset()`, we can check `D.size` for the number of datapoints; `D.features` for the number of features contained in each value; `D.classifications` for a list of unique classification names; or indeed `D.data` for a full list of all the datapoints as dicts.

```
>>> train = Dataset('train.data')
>>> train.size
120
>>> train.features
4
>>> train.classifications
['1', '2', '3']
>>> train.data[0]
{'x': array([1. , 5.1, 3.5, 1.4, 0.2]), 'class_name': '1'}

>>> test = Dataset('test.data')
>>> test.size
30
>>> test.classifications
['1', '2', '3']
>>> test.features
4

```

In theory, this should allow us to load any data in the form of a Comma-Separated Values file, where all the values are floats except for the final column, which is a classification in the form `class-{X}` where {X} is the classification.

```
>>> random = Dataset('random.data')
>>> random.size
33
>>> random.features
6
>>> random.classifications
['Elephant', 'Tiger']
>>> jungle_guide = Perceptron('Elephant','Tiger')
>>> jungle_guide.train(D=random)
>>> jungle_guide.test(D=random)
Success rate: 54.54545454545454%
```
