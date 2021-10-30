# Intro_To_ML_cw1
#### The Coursework 1 of `Introduction to Machine Learning`.
## Instruction for Program
#### This program is written based on Python, simply run `main.py` directly to start the script.
#### To generate tree visualisation diagram, run `visualize` function in `utils.py`.

## Files in this project:
### `main.py` 
### Run this file to start the project.
### `utils.py`
### Read this file to get an idea of how this project is implemented step by step. 
### `Decision_tree.py` 
### The decision tree is implemented in this file as a class. Please check it for 
- decision_tree_learning()
- find_split()
- prune()
- predict()
- ...
### `metric.py`
### The evaluation functions are implemented in this file in a class named *MyMetric*. Check this file for imformation about
- confusion_matrix()
- accuracy()
- precision()
- recall()
- f1()
- ...

## Implementation of Decision Tree (If you are interested...)
### Step 1: Read files
#### Two files `clean.txt` and `noisy.txt` are given, in each of them there are signal data (features) for  each of the seven wifis and a lable indicating which room it is.
### Step 2: Creating Decision Trees
#### 2.1 Defining information gain function. Functions in `utils.py`

#### 2.2 find_split function and Decision tree creating. Implemented in `Decision_tree.py`
Decision tree is defined as a class with *find_split* and *decision_tree_learning* as functions in it.
Nodes are defined as classes, also in this file.

### Step 3: Evaluation
#### 3.1 Cross validation. This is implemented in `utils.py`

