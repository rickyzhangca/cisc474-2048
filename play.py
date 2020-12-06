# -*- coding: utf-8 -*-
"""
Play the game with trained weight stored in trained
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from copy import deepcopy
import random 
import math
import time
from tkinter import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#conv layer1 depth
depth1 = 128

#conv layer2 depth
depth2 = 128

#input depth
input_depth = 16

#fully conneted hidden layer
hidden_units = 256

#output layer
output_units = 4

#shape of weights
conv1_layer1_shape = [2,1,input_depth,depth1]
conv1_layer2_shape = [2,1,depth1,depth2]
conv2_layer1_shape = [1,2,input_depth,depth1]
conv2_layer2_shape = [1,2,depth1,depth2]

fc_layer1_w_shape = [3*4*depth1*2+ 4*2*depth2*2 + 3*3*depth2*2,hidden_units]
fc_layer1_b_shape = [hidden_units]
fc_layer2_w_shape = [hidden_units,output_units]
fc_layer2_b_shape = [output_units]

parameters = dict()

path = r'./trained'
parameters['conv1_layer1'] = np.array(pd.read_csv(path + r'/conv1_layer1_weights.csv')['Weight']).reshape(conv1_layer1_shape)
parameters['conv1_layer2'] = np.array(pd.read_csv(path + r'/conv1_layer2_weights.csv')['Weight']).reshape(conv1_layer2_shape)
parameters['conv2_layer1'] = np.array(pd.read_csv(path + r'/conv2_layer1_weights.csv')['Weight']).reshape(conv2_layer1_shape)
parameters['conv2_layer2'] = np.array(pd.read_csv(path + r'/conv2_layer2_weights.csv')['Weight']).reshape(conv2_layer2_shape)
parameters['fc_layer1_w'] = np.array(pd.read_csv(path + r'/fc_layer1_weights.csv')['Weight']).reshape(fc_layer1_w_shape)
parameters['fc_layer1_b'] = np.array(pd.read_csv(path + r'/fc_layer1_biases.csv')['Weight']).reshape(fc_layer1_b_shape)
parameters['fc_layer2_w'] = np.array(pd.read_csv(path + r'/fc_layer2_weights.csv')['Weight']).reshape(fc_layer2_w_shape)
parameters['fc_layer2_b'] = np.array(pd.read_csv(path + r'/fc_layer2_biases.csv')['Weight']).reshape(fc_layer2_b_shape)

def new_game(n):
    matrix = []

    for i in range(n):
        matrix.append([0] * n)
    return matrix

def add_two(mat):
    empty_cells = []
    for i in range(len(mat)):
        for j in range(len(mat)):
            if(mat[i][j]==0):
                empty_cells.append((i,j))
    if(len(empty_cells)==0):
        return mat
    index_pair = empty_cells[random.randint(0,len(empty_cells)-1)]
    
    num = np.random.random(1)
    if(num>0.9):
        mat[index_pair[0]][index_pair[1]]=4
    else:
        mat[index_pair[0]][index_pair[1]]=2
    return mat

def game_state(mat):
    for i in range(len(mat)-1): #intentionally reduced to check the row on the right and below
        for j in range(len(mat[0])-1): #more elegant to use exceptions but most likely this will be their solution
            if mat[i][j]==mat[i+1][j] or mat[i][j+1]==mat[i][j]:
                return 'not over'
    for i in range(len(mat)): #check for any zero entries
        for j in range(len(mat[0])):
            if mat[i][j]==0:
                return 'not over'
    for k in range(len(mat)-1): #to check the left/right entries on the last row
        if mat[len(mat)-1][k]==mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1): #check up/down entries on last column
        if mat[j][len(mat)-1]==mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'

def reverse(mat):
    new=[]
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new

def transpose(mat):
    new=[]
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new

def cover_up(mat):
    new=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    done=False
    for i in range(4):
        count=0
        for j in range(4):
            if mat[i][j]!=0:
                new[i][count]=mat[i][j]
                if j!=count:
                    done=True
                count+=1
    return (new,done)

def merge(mat):
    done=False
    score = 0
    for i in range(4):
         for j in range(3):
             if mat[i][j]==mat[i][j+1] and mat[i][j]!=0:
                 mat[i][j]*=2
                 score += mat[i][j]   
                 mat[i][j+1]=0
                 done=True
    return (mat,done,score)


def up(game):
        game=transpose(game)
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        game=transpose(game)
        return (game,done,temp[2])

def down(game):
        game=reverse(transpose(game))
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        game=transpose(reverse(game))
        return (game,done,temp[2])

def left(game):
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        return (game,done,temp[2])

def right(game):
        game=reverse(game)
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        game=reverse(game)
        return (game,done,temp[2])
    

def findemptyCell(mat):
    count = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            if(mat[i][j]==0):
                count+=1
    return count

def change_values(X):
    power_mat = np.zeros(shape=(1,4,4,16),dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if(X[i][j]==0):
                power_mat[0][i][j][0] = 1.0
            else:
                power = int(math.log(X[i][j],2))
                power_mat[0][i][j][power] = 1.0
    return power_mat

controls = {0:up,1:left,2:right,3:down}

learned_graph = tf.Graph()

with learned_graph.as_default():
    
    #input data
    single_dataset = tf.placeholder(tf.float32,shape=(1,4,4,16))
    
    #weights
    
    #conv layer1 weights
    conv1_layer1_weights = tf.constant(parameters['conv1_layer1'],dtype=tf.float32)
    conv1_layer2_weights = tf.constant(parameters['conv1_layer2'],dtype=tf.float32)
    
    #conv layer2 weights
    conv2_layer1_weights = tf.constant(parameters['conv2_layer1'],dtype=tf.float32)
    conv2_layer2_weights = tf.constant(parameters['conv2_layer2'],dtype=tf.float32)
    
    #fully connected parameters
    fc_layer1_weights = tf.constant(parameters['fc_layer1_w'],dtype=tf.float32)
    fc_layer1_biases = tf.constant(parameters['fc_layer1_b'],dtype=tf.float32)
    fc_layer2_weights = tf.constant(parameters['fc_layer2_w'],dtype=tf.float32)
    fc_layer2_biases = tf.constant(parameters['fc_layer2_b'],dtype=tf.float32)
    
    
    #model
    def model(dataset):
        #layer1
        conv1 = tf.nn.conv2d(dataset,conv1_layer1_weights,[1,1,1,1],padding='VALID') 
        conv2 = tf.nn.conv2d(dataset,conv2_layer1_weights,[1,1,1,1],padding='VALID') 

        #layer1 relu activation
        relu1 = tf.nn.relu(conv1)
        relu2 = tf.nn.relu(conv2)

        #layer2
        conv11 = tf.nn.conv2d(relu1,conv1_layer2_weights,[1,1,1,1],padding='VALID') 
        conv12 = tf.nn.conv2d(relu1,conv2_layer2_weights,[1,1,1,1],padding='VALID') 

        conv21 = tf.nn.conv2d(relu2,conv1_layer2_weights,[1,1,1,1],padding='VALID') 
        conv22 = tf.nn.conv2d(relu2,conv2_layer2_weights,[1,1,1,1],padding='VALID') 

        #layer2 relu activation
        relu11 = tf.nn.relu(conv11)
        relu12 = tf.nn.relu(conv12)
        relu21 = tf.nn.relu(conv21)
        relu22 = tf.nn.relu(conv22)

        #get shapes of all activations
        shape1 = relu1.get_shape().as_list()
        shape2 = relu2.get_shape().as_list()

        shape11 = relu11.get_shape().as_list()
        shape12 = relu12.get_shape().as_list()
        shape21 = relu21.get_shape().as_list()
        shape22 = relu22.get_shape().as_list()

        #expansion
        hidden1 = tf.reshape(relu1,[shape1[0],shape1[1]*shape1[2]*shape1[3]])
        hidden2 = tf.reshape(relu2,[shape2[0],shape2[1]*shape2[2]*shape2[3]])

        hidden11 = tf.reshape(relu11,[shape11[0],shape11[1]*shape11[2]*shape11[3]])
        hidden12 = tf.reshape(relu12,[shape12[0],shape12[1]*shape12[2]*shape12[3]])
        hidden21 = tf.reshape(relu21,[shape21[0],shape21[1]*shape21[2]*shape21[3]])
        hidden22 = tf.reshape(relu22,[shape22[0],shape22[1]*shape22[2]*shape22[3]])

        #concatenation
        hidden = tf.concat([hidden1,hidden2,hidden11,hidden12,hidden21,hidden22],axis=1)

        #full connected layers
        hidden = tf.matmul(hidden,fc_layer1_weights) + fc_layer1_biases
        hidden = tf.nn.relu(hidden)

        #output layer
        output = tf.matmul(hidden,fc_layer2_weights) + fc_layer2_biases

        #return output
        return output

    #for single example
    single_output = model(single_dataset)

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }

CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }

FONT = ("Verdana", 40, "bold")

learned_sess = tf.Session(graph=learned_graph)

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()
        
        self.wait_visibility()
        self.after(10,self.make_move)
        
    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(4)

        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()
        
    def make_move(self):
        output = learned_sess.run([single_output],feed_dict = {single_dataset:change_values(self.matrix)})
        move = np.argmax(output[0])
        self.matrix,done,tempo = controls[move](self.matrix)
        done=True
        
        #if game_state(self.matrix)=='win':
        #    self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
        #    self.grid_cells[1][2].configure(text="Win!",bg=BACKGROUND_COLOR_CELL_EMPTY)
        #    done=False
        if game_state(self.matrix)=='lose':
            self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Lose!",bg=BACKGROUND_COLOR_CELL_EMPTY)
            done=False
            
        self.matrix = add_two(self.matrix)
        self.update_grid_cells()
        
        
        if(done==True):
            self.after(7,self.make_move)
        else:
            time.sleep(3)
            self.init_matrix()
            self.update_grid_cells()
            self.after(7,self.make_move)

    def generate_next(self):
        empty_cells = []
        for i in range(len(mat)):
            for j in range(len(mat)):
                if(mat[i][j]==0):
                    empty_cells.append((i,j))
        if(len(empty_cells)==0):
            return 0,false
        index_pair = empty_cells[random.randint(0,len(empty_cells)-1)]
        index = index_pair
        self.matrix[index[0]][index[1]] = 2

root = Tk()
gamegrid = GameGrid()
root.mainloop()

