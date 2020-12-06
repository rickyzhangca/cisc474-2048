import numpy as np
import random
import math

# initialize a new game
def new_game(n):
    matrix = np.zeros([n,n])
    return matrix

# add 2 or 4 in the matrix
def add_two(mat):
    empty_cells = []
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if(mat[i][j]==0):
                empty_cells.append((i,j))
    if(len(empty_cells)==0):
        return mat
    
    index_pair = empty_cells[random.randint(0,len(empty_cells)-1)]
    
    prob = random.random()
    if(prob>=0.9):
        mat[index_pair[0]][index_pair[1]]=4
    else:
        mat[index_pair[0]][index_pair[1]]=2
    return mat

# to check state of the game
def game_state(mat):
    #if 2048 in mat:
    #    return 'win'
    
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
            
    return np.transpose(mat)

def cover_up(mat):
    new = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    done = False
    for i in range(4):
        count = 0
        for j in range(4):
            if mat[i][j]!=0:
                new[i][count] = mat[i][j]
                if j!=count:
                    done=True
                count+=1
    return (new,done)

def merge(mat):
    done=False
    score = 0
    visited = []
    for i in range(4):
         for j in range(3):
             if (not(mat[i][j] in visited)) and (not(mat[i][j+1] in visited)) and mat[i][j]==mat[i][j+1] and mat[i][j]!=0:
                 mat[i][j]*=2
                 score += mat[i][j]   
                 mat[i][j+1]=0
                 visited.append(mat[i][j])
                 visited.append(mat[i][j+1])
                 done=True
    for i in range(3):
         for j in range(4):
             if (not(mat[i][j] in visited)) and (not(mat[i+1][j] in visited)) and mat[i+1][j]==mat[i][j] and mat[i][j]!=0:
                 mat[i][j]*=2
                 score += mat[i][j]   
                 mat[i+1][j]=0
                 visited.append(mat[i][j])
                 visited.append(mat[i+1][j])
                 done=True
    return (mat,done,score)

# up move
def up(game):
        game = transpose(game)
        game,done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        game = transpose(game)
        return (game,done,temp[2])

# down move
def down(game):
        game=reverse(transpose(game))
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        game=transpose(reverse(game))
        return (game,done,temp[2])

# left move
def left(game):
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        return (game,done,temp[2])

# right move
def right(game):
        game=reverse(game)
        game,done=cover_up(game)
        temp=merge(game)
        game=temp[0]
        done=done or temp[1]
        game=cover_up(game)[0]
        game=reverse(game)
        return (game,done,temp[2])

controls = {0:up,1:left,2:right,3:down}

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

# find the number of empty cells in the game matrix.
def findemptyCell(mat):
    count = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            if(mat[i][j]==0):
                count+=1
    return count