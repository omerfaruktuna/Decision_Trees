# Decision_Trees
Decision Tree Classifier from scratch using Python

#Inspired from Josh Gordon
#https://www.youtube.com/watch?v=LDRbO9a6XPU
#Instead of Gini Impurity, I have chosed to use Entropy for finding Information Gain in order to decide creating child nodes

For the input dataset, you should add a file named dataset.txt to the same directory as main.py

All the fetaures are seperated by comma in the dataset file, and the last element of each row is the label.

For example, for below file: Green and 3 are 2 different feaures and Apple is the label for first input data.

Green,3,Apple
Yellow,3,Apple
Red,1,Grape
Red,1,Grape
Yellow,3,Lemon

You should add your data headers to main.py file as in below format:

header = ["color", "diameter", "label"]
