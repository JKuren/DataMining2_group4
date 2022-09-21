# ------------------------- Created by Yi Zhao, IT department, Uppsala University, on 2022.9 ---------------------------------------
from re import T
from textwrap import indent
from xml.dom.minidom import Element
import numpy as np
import copy
import pickle


class FP_node: # a node in a FP tree
    def __init__(self, name, parent):  # input: when we newly build a node in a FP tree, we know its path from 'root' (string) and the predecessor node's name (string)
        self.name = name        #  the node's name (string, e.g. 'rootABC'), recording the path from root node to it
        self.parent = parent    #  the name (string, e.g. 'rootAB') of the node's predecessor node  in a FP-tree
        self.children = []      #  the names (list of strings, e.g. ['rootABCD', 'rootABCF']) of the node's successor nodes in a FP-tree
        self.count = 0          #  the node's count (int) = the number of appearance the itemset (in original FP-tree) and can be revised (in conditional sub-FP-trees)


class Header_table(): # a header table
    def __init__(self, freq_items): # input: the singleton items whose support is higher than the minimum support, sorted by their supports (list of strings)
        self.freq_items = freq_items  # record the frequent items
        # build an empty header table
        self.table = {}
        for item in freq_items:
            self.table[item] = []

    def revise_table(self, node_names): # fill the table according to a given fp-tree. Input: the names of nodes
        for node_name in node_names:
            if node_name != 'root':
                self.table[node_name[-1]].append(node_name)  # node_name[-1] = the ending letter

    def output_sum_counts(self, FP_tree): # compute the sum counts for all nodes with the same ending letter
        sum_counts = {}
        for item in self.freq_items:
            sum_count = 0
            for node_name in self.table[item]:
                sum_count += FP_tree.nodes[node_name].count
            sum_counts[item] = sum_count
        return sum_counts   # (dictionary with string keys and int values)


class FP_tree:
    def __init__(self):
        self.nodes = {}  # the nodes the tree, (dictionary with string keys (i.e. node names) and FP-node values)
        self.nodes['root'] = FP_node('root', None) # add the root node

    def delete_unsupported_nodes(self, min_supp): # delete all the nodes that do not satisfy the minimum support
        node_names = list(self.nodes.keys())
        # record all the nodes that do not satisfy the minimum support
        nodes_to_delete = []
        for node_name in node_names:
            if (self.nodes[node_name].count < min_supp) & (node_name != 'root'):
                nodes_to_delete.append(node_name)
        # revise the connection relationship before deleting nodes
        for node_name in nodes_to_delete:
            parent = self.nodes[node_name].parent
            self.nodes[parent].children.remove(node_name)
            for child in self.nodes[node_name].children:
                if child not in nodes_to_delete:
                    self.nodes[child].parent = parent
        # deleting nodes
        for node_name in nodes_to_delete:
                self.nodes.pop(node_name)

    def construct_tree_and_table(self, transactions, min_supp): # construct a FP-tree and build the corresponding header table
        freq_items = find_frequent_items(transactions, min_supp) # the fist scan of dataset, to find the singleton items whose support is higher than the minimum support.
        transactions = sort_and_cut_transactions(transactions, freq_items) # delete the infrequent (i.e. < min_supp) items from a transaction and sorting remaining items in descending order od supports
        for tran in transactions:  # the second scan of dataset, tran is a row of data in transactions
            footprint = 'root' # use footprint to record the current location in a fp-tree
            for item in tran:
                if footprint + item in self.nodes[footprint].children:
                    footprint = footprint+item
                else:
                    self.nodes[footprint + item] = FP_node(footprint + item, footprint)  # add new node to the tree
                    self.nodes[footprint].children.append(footprint + item)  # update the connection among nodes
                    footprint = footprint + item  # update the current location
                self.nodes[footprint].count += 1   # update the count
        # accordingly build the header table
        self.header_table = Header_table(freq_items)
        self.header_table.revise_table(self.nodes.keys())


    def construct_conditional_FPtree(self, suffix, min_supp): # build a fp-sub-tree with given suffix (char, e.g. 'E')
        sub_tree = copy.deepcopy(self)  # The sub-tree is also an object belonging to Class FP-tree
        # Step 1: build the sub-tree with given suffix
        sub_tree.nodes = {}  #clear all the nodes
        sub_tree.nodes['root'] = copy.deepcopy(self.nodes['root'])
        for leaf in self.header_table.table[suffix]:
            back_footprint = leaf  # use back_footprint to record the current location of a fp-tree when scanning
            pre_back_footprint = []  # use pre_back_footprint to record the previous location in a fp-tree when scanning
            # build the nodes in the sub-tree and update the count at the same time
            while back_footprint != []:
                if back_footprint in sub_tree.nodes.keys():
                    sub_tree.nodes[back_footprint].count += sub_tree.nodes[pre_back_footprint].count
                    sub_tree.nodes[back_footprint].children.append(pre_back_footprint)
                else:
                    sub_tree.nodes[back_footprint] = copy.deepcopy(self.nodes[back_footprint])
                    if back_footprint in self.header_table.table[suffix]:
                        sub_tree.nodes[back_footprint].children = []
                    else:
                        sub_tree.nodes[back_footprint].count = sub_tree.nodes[pre_back_footprint].count
                        sub_tree.nodes[back_footprint].children = [pre_back_footprint]
                if back_footprint == 'root':
                    break
                else:
                    pre_back_footprint = back_footprint
                    back_footprint = self.nodes[back_footprint].parent
        # Step 2: deleting the nodes whose names ending with suffix
        for node_name in self.header_table.table[suffix]:
            parent = sub_tree.nodes[node_name].parent
            sub_tree.nodes[parent].children.remove(node_name)
            sub_tree.nodes.pop(node_name)
        # Step 3: deleting the nodes that do not satisfy the minimum support
        sub_tree.delete_unsupported_nodes(min_supp)
        # build the corresponding header table
        sub_tree.header_table = Header_table(self.header_table.freq_items)
        sub_tree.header_table.revise_table(sub_tree.nodes.keys())
        return sub_tree


#TODO 1: import the data stored in trans1000.pkl. HINT: use function pickle.load(...)
def import_data():
    with open('trans1000.pkl', 'rb') as f:
        transactions = pickle.load(f)
    return transactions['transactions']   #output type: list of list of char, c.f. the small dataset

#TODO 2: find all the items whose supports are not less than the minimum support, and sort items in descending order of their supports
# HINT:
# Step 1: count and record the support of each singleton item
#         two nesting For loops needed for scan the dataset
#         Use a dictionary (named item_support for example) where the key is the item name (a char e.g. 'A') and the value is the support of this item (int value). Use item_support['A']=5 to record that the support of item A is 5.
# Step 2: rule out items with support less than min_supp
#         Use item_support.pop('A') to delete the information about item 'A' in dictionary item_support
# Step 3: count and record the support of each singleton item
#         Use item_support.keys() and item_support.values() to extract the keys and values in dictionary item_support, respectively
#         Use data type conversion like np.array() and list()
#         Use np.argsort(...) to sort the values and output the ordered indexes
def find_frequent_items(transactions, min_supp):  #transactions: list of list of char, min_supp: int, the minimum support
    item_support ={}
    for i in transactions:
        for j in i:
            if j not in item_support:
                item_support[j] = 1
            else:
                item_support[j] += 1
    item_support = {key:val for key, val in item_support.items() if val >= min_supp} # Delete all items with support less than min_supp
    sorted_dict =  {k: v for k, v in sorted(item_support.items(), key=lambda item: item[1], reverse=True)} # Sorts the dictionary by descending value 
    freq_items = list(sorted_dict)
    return freq_items  # output type: list of chars


# TODO 3: for any transaction, delete the items that are not in freq_items, and sort the remaining items with the given order of freq_items
#  (e.g., transactions = [['A', 'C', 'D']], freq_items =['C', 'B', 'E', 'A'], then sorted_cutted_transactions= [['C', 'A']])
#  HINT:
#  Do steps 1&2 for all transaction:
#         Step 1: For each item in a transaction, if it is in freq_items, record itself and its index in freq_items
#         Step 2: For each transaction, sort the items according to the indexes recorded in Step 1
#                 Use data type conversion like np.array() and list()
#                 Use np.argsort(...) to sort the values and output the ordered indexes
def sort_and_cut_transactions(transactions, freq_items):
    sorted_cutted_transactions = []
    freq_items = list(freq_items)

    order_dict = { tag : i for i,tag in enumerate(freq_items) }

    for i in transactions:
        temp_dict = {}
        for j in i:
            if j in freq_items:
                temp_dict[j] = order_dict[j]
                temp_dict = dict(sorted(temp_dict.items(), key=lambda item: item[1]))
        sorted_cutted_transactions.append(list(temp_dict))

    return sorted_cutted_transactions  # output type: list of list of chars

    

def find_frequent_itemsets(fp_tree, min_supp, post_suffix): #find the frequent itemsets with an original or conditional fp-tree, post_suffix(string): the ending part of letters that had been fixed
    frequent_itemsets = []  # build an empty list

    if max(fp_tree.header_table.output_sum_counts(fp_tree)) == 0: # when the fp-tree/fp-sub-tree only contains the root node
        return []
    else:
        for suffix in fp_tree.header_table.freq_items: # multiple sub-problems with different ending letters
            if fp_tree.header_table.output_sum_counts(fp_tree)[suffix] >= min_supp: # check whether the minimum support is satisfied
                frequent_itemsets.append(suffix + post_suffix)  # record the frequent itemsets related to the current tree's leaves
                # TODO 4: build a sub_fp_tree with suffix and min_supp
                # TODO 5: recursively call function find_frequent_itemsets(...) to find the frequent itemsets in smaller sub-trees, and add its result frequent_itemsets. HINT: using "sub_fp_tree, min_supp, suffix + post_suffix" as the actual parameters, and use += to concatenate two lists
                frequent_itemsets += find_frequent_itemsets(fp_tree.construct_conditional_FPtree(suffix, min_supp), min_supp, suffix+post_suffix)
    return frequent_itemsets  # output type: list of strings



#------------------------------------------ Useful code for debugging --------------------------------------------------
#---------- A small dataset for debugging ---------------
# transactions = [
#         ['A', 'B', 'C', 'E', 'F','Z'],
#         ['A', 'C', 'G'],
#         ['A', 'C', 'D', 'E', 'G'],
#         ['A', 'C', 'E', 'G', 'L'],
#         ['A', 'C', 'B'],
#         ['A', 'B', 'D'],
#         ['A', 'B'],
#         ['A', 'B']
#         ]
# transactions = [
#     [1,2,3,5,6,20],
#     [1,3,7],
#     [1,3,4,5,7],
#     [1,3,5,7,12],
#     [1,3,2],
#     [1,2,4],
#     [1,2],
#     [1,2],
#     ]

# min_supp = 3
# # test function find_frequent_item
# freq_items = find_frequent_items(transactions, min_supp)
# print('freq_items:', freq_items)
# # test function sort_and_cut_transactions
# transactions = sort_and_cut_transactions(transactions, freq_items)
# print('sorted and cut transactions:', transactions)
# # test function import_data
# transactions = import_data()
# print('transactions:',transactions)


#---------------------------------------------- The main process ----------------------------------------------------
#-----------------------------------   Step 1: Import the dataset and parameter ------------------------------------
transactions = import_data()
min_supp = 300
freq_items = find_frequent_items(transactions, min_supp)
#print(transactions)
#-----------------------------------------   Step 2: Build a FP-growth tree -----------------------------------------
fp_tree = FP_tree()
fp_tree.construct_tree_and_table(transactions, min_supp)
# #-----------------------------------------   Step 3: Find the frequent itemsets -------------------------------------
frequent_itemsets = find_frequent_itemsets(fp_tree, min_supp, '')
print('frequent_itemsets:', frequent_itemsets)