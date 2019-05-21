#Inspired from Josh Gordon
#https://www.youtube.com/watch?v=LDRbO9a6XPU
#Instead of Gini Impurity, I have chosed to use Entropy for finding Information Gain in order to decide creating child nodes

import math

dataset_file = open('dataset4.txt', 'r')

dataset = []

x = 0

for line in dataset_file.readlines():
  dataset.append([])
  strp_line = line.strip()
  currentline = strp_line.split(",")
  for i in range(len(currentline)):
    dataset[x].append("{}".format(currentline[i]))
  x +=1

dataset_file.close()

header = ["color", "diameter", "label"]

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):

        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows 

def entropy(rows):
    counts = class_counts(rows)
    prob_of_lbl_arr = []
    entropy = 0

    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        prob_of_lbl_arr.append(prob_of_lbl)

    for i in range(len(prob_of_lbl_arr)):
      entropy+=math.log2(prob_of_lbl_arr[i])*prob_of_lbl_arr[i]
    if entropy == 0:
      return entropy
    else:
      return (-entropy)

def info_gain_entropy(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

def find_best_split(rows):

    best_gain = 0
    best_question = None
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):

        values = set([row[col] for row in rows])

        for val in values:

            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain_entropy(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)
       
class  Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def  build_tree(rows):

    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):

    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    print (spacing + str(node.question))

    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

my_tree = build_tree(dataset)

print("---------------------------------")
print("Here is the tree\n")
print_tree(my_tree)

print("---------------------------------")

def  classify(row, node):

    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

#print(classify(dataset[0], my_tree))



def  print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

print("Test Output is:\n")

print(print_leaf(classify(dataset[0], my_tree)))

print("\n")
