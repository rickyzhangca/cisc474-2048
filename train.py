import tensorflow as tf
import numpy as np
from copy import deepcopy
import random 
import math
import matplotlib.pyplot as plt
import game
import time

################################################################################################################

'''
setup tensorflow
'''
def setup():
    # python -m venv --system-site-packages .\venv
    # Set-ExecutionPolicy Unrestricted -Scope Process
    # .\venv\Scripts\activate

    # disable tf2's eager execution to use tf1 style pipeline
    tf.compat.v1.disable_eager_execution()
    # check cuda info
    print(tf.test.is_built_with_cuda()) 
    print(tf.config.list_physical_devices('GPU'))
    
setup()

'''
save a given list to txt or csv file using w+ policy
'''
def save(path, name, lis, mode):
    file = open(path + name + mode,'w+')
    if mode == '.txt':  
        for i in range(len(lis)):
            file.write(str(lis[i])+"\n")     
        file.close()
    elif mode == '.csv':
        file.write('Sno,Weight\n') ###
        for i in range(lis.shape[0]):
            file.write(str(i) + ',' + str(lis[i][0])+'\n') 
    file.close()
    print(path + name + mode + " is written")

################################################################################################################

###

# hyper parameters
start_learning_rate = 0.0005

# gamma for Q-learning
gamma = 0.9

# epsilon greedy approach
epsilon = 0.9

# to store states and lables of the game for training states
replay_memory = list()

# labels of the states
replay_labels = list()

# capacity of memory
mem_capacity = 6000

# first convolution layer depth
depth1 = 128

# second convolution layer depth
depth2 = 128

# batch size for batch gradient descent
batch_size = 512

# input units
input_units = 16

# fully connected layer neurons
hidden_units = 256

# output neurons = number of moves
output_units = 4

# input data
tf_batch_dataset = tf.compat.v1.placeholder(tf.float32,shape=(batch_size,4,4,16))
tf_batch_labels  = tf.compat.v1.placeholder(tf.float32,shape=(batch_size,output_units))
single_dataset   = tf.compat.v1.placeholder(tf.float32,shape=(1,4,4,16))

#CONV LAYERS
#conv layer1 weights
conv1_layer1_weights = tf.Variable(tf.random.truncated_normal([1,2,input_units,depth1],mean=0,stddev=0.01))
conv2_layer1_weights = tf.Variable(tf.random.truncated_normal([2,1,input_units,depth1],mean=0,stddev=0.01))

#conv layer2 weights
conv1_layer2_weights = tf.Variable(tf.random.truncated_normal([1,2,depth1,depth2],mean=0,stddev=0.01))
conv2_layer2_weights = tf.Variable(tf.random.truncated_normal([2,1,depth1,depth2],mean=0,stddev=0.01))

#FUllY CONNECTED LAYERS
expand_size = 2*4*depth2*2 + 3*3*depth2*2 + 4*3*depth1*2
fc_layer1_weights = tf.Variable(tf.random.truncated_normal([expand_size,hidden_units],mean=0,stddev=0.01))
fc_layer1_biases = tf.Variable(tf.random.truncated_normal([1,hidden_units],mean=0,stddev=0.01))
fc_layer2_weights = tf.Variable(tf.random.truncated_normal([hidden_units,output_units],mean=0,stddev=0.01))
fc_layer2_biases = tf.Variable(tf.random.truncated_normal([1,output_units],mean=0,stddev=0.01))

#model
def model(dataset):
    # layer1
    conv1 = tf.nn.conv2d(dataset,conv1_layer1_weights,[1,1,1,1],padding='VALID') 
    conv2 = tf.nn.conv2d(dataset,conv2_layer1_weights,[1,1,1,1],padding='VALID') 
    
    # layer1 relu activation
    relu1 = tf.nn.relu(conv1)
    relu2 = tf.nn.relu(conv2)
    
    # layer2
    conv11 = tf.nn.conv2d(relu1,conv1_layer2_weights,[1,1,1,1],padding='VALID') 
    conv12 = tf.nn.conv2d(relu1,conv2_layer2_weights,[1,1,1,1],padding='VALID') 

    conv21 = tf.nn.conv2d(relu2,conv1_layer2_weights,[1,1,1,1],padding='VALID') 
    conv22 = tf.nn.conv2d(relu2,conv2_layer2_weights,[1,1,1,1],padding='VALID') 

    # layer2 relu activation
    relu11 = tf.nn.relu(conv11)
    relu12 = tf.nn.relu(conv12)
    relu21 = tf.nn.relu(conv21)
    relu22 = tf.nn.relu(conv22)
    
    # get shapes of all activations
    shape1 = relu1.get_shape().as_list()
    shape2 = relu2.get_shape().as_list()
    
    shape11 = relu11.get_shape().as_list()
    shape12 = relu12.get_shape().as_list()
    shape21 = relu21.get_shape().as_list()
    shape22 = relu22.get_shape().as_list()

    # expansion
    hidden1 = tf.reshape(relu1,[shape1[0],shape1[1]*shape1[2]*shape1[3]])
    hidden2 = tf.reshape(relu2,[shape2[0],shape2[1]*shape2[2]*shape2[3]])
    
    hidden11 = tf.reshape(relu11,[shape11[0],shape11[1]*shape11[2]*shape11[3]])
    hidden12 = tf.reshape(relu12,[shape12[0],shape12[1]*shape12[2]*shape12[3]])
    hidden21 = tf.reshape(relu21,[shape21[0],shape21[1]*shape21[2]*shape21[3]])
    hidden22 = tf.reshape(relu22,[shape22[0],shape22[1]*shape22[2]*shape22[3]])

    # concatenation
    hidden = tf.concat([hidden1,hidden2,hidden11,hidden12,hidden21,hidden22],axis=1)

    # full connected layers
    hidden = tf.matmul(hidden,fc_layer1_weights) + fc_layer1_biases
    hidden = tf.nn.relu(hidden)

    # output layer
    output = tf.matmul(hidden,fc_layer2_weights) + fc_layer2_biases
    
    # return output
    return output

# for single example
single_output = model(single_dataset)

# for batch data
logits = model(tf_batch_dataset)

# loss
loss = tf.square(tf.subtract(tf_batch_labels,logits))
loss = tf.reduce_sum(loss,axis=1) #keep_dims=True
loss = tf.reduce_mean(loss)/2.0

# optimizer
global_step = tf.Variable(0)  # count the number of steps taken.
learning_rate = tf.compat.v1.train.exponential_decay(float(start_learning_rate), global_step, 1000, 0.90, staircase=True)
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

# loss
losses = []

# scores
scores = []

# to store final parameters
final_parameters = {}

# number of episodes
M = 20000

with tf.compat.v1.Session() as session:

    tf.compat.v1.global_variables_initializer().run()
    print("Initialized")

    #for episode with max score
    maximum = -1
    episode = -1

    #total_iters 
    total_iters = 1

    #number of back props
    back=0

    for ep in range(M):
        start_time = time.time()
        global board
        board = game.new_game(4)
        game.add_two(board)
        game.add_two(board)
        
        #whether episode finished or not
        finish = 'not over'
        
        #total_score of this episode
        total_score = 0
        
        #iters per episode
        local_iters = 1
        
        while(finish=='not over'):
            prev_board = deepcopy(board)
            
            #get the required move for this state
            state = deepcopy(board)
            state = game.change_values(state)
            state = np.array(state,dtype = np.float32).reshape(1,4,4,16)
            feed_dict = {single_dataset:state}
            control_scores = session.run(single_output,feed_dict=feed_dict)
            
            #find the move with max Q value
            control_buttons = np.flip(np.argsort(control_scores),axis=1)
            
            #copy the Q-values as labels
            labels = deepcopy(control_scores[0])
            
            #generate random number for epsilon greedy approach
            num = random.uniform(0,1)
            
            #store prev max
            prev_max = np.max(prev_board)
            
            #num is less epsilon generate random move
            if(num<epsilon):
                #find legal moves
                legal_moves = list()
                for i in range(4):
                    temp_board = deepcopy(prev_board)
                    temp_board,_,_ = game.controls[i](temp_board)
                    if(np.array_equal(temp_board,prev_board)):
                        continue
                    else:
                        legal_moves.append(i)
                if(len(legal_moves)==0):
                    finish = 'lose'
                    continue
                
                #generate random move.
                con = random.sample(legal_moves,1)[0]
                
                #apply the move
                temp_state = deepcopy(prev_board)
                temp_state,_,score = game.controls[con](temp_state)
                total_score += score
                finish = game.game_state(temp_state)
                
                #get number of merges
                empty1 = game.findemptyCell(prev_board)
                empty2 = game.findemptyCell(temp_state)
                
                if(finish=='not over'):
                    temp_state = game.add_two(temp_state)

                board = deepcopy(temp_state)

                #get next max after applying the move
                next_max = np.max(temp_state)
                
                #reward math.log(next_max,2)*0.1 if next_max is higher than prev max
                labels[con] = (math.log(next_max,2))*0.1
                
                if(next_max==prev_max):
                    labels[con] = 0
                
                #reward is also the number of merges
                labels[con] += (empty2-empty1)
                
                #get the next state max Q-value
                temp_state = game.change_values(temp_state)
                temp_state = np.array(temp_state,dtype = np.float32).reshape(1,4,4,16)
                feed_dict = {single_dataset:temp_state}
                temp_scores = session.run(single_output,feed_dict=feed_dict)
                    
                max_qvalue = np.max(temp_scores)
                
                #final labels add gamma*max_qvalue
                labels[con] = (labels[con] + gamma*max_qvalue)
            
            #generate the the max predicted move
            else:
                for con in control_buttons[0]:
                    prev_state = deepcopy(prev_board)
                    
                    #apply the LEGAl Move with max q_value
                    temp_state,_,score = game.controls[con](prev_state)
                    
                    #if illegal move label = 0
                    if(np.array_equal(prev_board,temp_state)):
                        labels[con] = 0
                        continue
                        
                    #get number of merges
                    empty1 = game.findemptyCell(prev_board)
                    empty2 = game.findemptyCell(temp_state)

                    
                    temp_state = game.add_two(temp_state)
                    board = deepcopy(temp_state)
                    total_score += score

                    next_max = np.max(temp_state)
                    
                    #reward
                    labels[con] = (math.log(next_max,2))*0.1
                    if(next_max==prev_max):
                        labels[con] = 0
                    
                    labels[con] += (empty2-empty1)

                    #get next max qvalue
                    temp_state = game.change_values(temp_state)
                    temp_state = np.array(temp_state,dtype = np.float32).reshape(1,4,4,16)
                    feed_dict = {single_dataset:temp_state}
                    temp_scores = session.run(single_output,feed_dict=feed_dict)

                    max_qvalue = np.max(temp_scores)

                    #final labels
                    labels[con] = (labels[con] + gamma*max_qvalue)
                    break
                    
                if(np.array_equal(prev_board,board)):
                    finish = 'lose'
            
            #decrease the epsilon value
            if((ep>10000) or (epsilon>0.1 and total_iters%2500==0)):
                epsilon = epsilon/1.005
                
            
            #change the matrix values and store them in memory
            prev_state = deepcopy(prev_board)
            prev_state = game.change_values(prev_state)
            prev_state = np.array(prev_state,dtype=np.float32).reshape(1,4,4,16)
            replay_labels.append(labels)
            replay_memory.append(prev_state)
            
            
            #back-propagation
            if(len(replay_memory)>=mem_capacity):
                back_loss = 0
                batch_num = 0
                z = list(zip(replay_memory,replay_labels))
                np.random.shuffle(z)
                np.random.shuffle(z)
                replay_memory,replay_labels = zip(*z)
                
                for i in range(0,len(replay_memory),batch_size):
                    if(i + batch_size>len(replay_memory)):
                        break
                        
                    batch_data = deepcopy(replay_memory[i:i+batch_size])
                    batch_labels = deepcopy(replay_labels[i:i+batch_size])
                    
                    batch_data = np.array(batch_data,dtype=np.float32).reshape(batch_size,4,4,16)
                    batch_labels = np.array(batch_labels,dtype=np.float32).reshape(batch_size,output_units)
                
                    feed_dict = {tf_batch_dataset: batch_data, tf_batch_labels: batch_labels}
                    _,l = session.run([optimizer,loss],feed_dict=feed_dict)
                    back_loss += l 
                    
                    # print("Mini-Batch - {} Back-Prop : {}, Loss : {}".format(batch_num,back,l))
                    batch_num +=1
                back_loss /= batch_num
                losses.append(back_loss)
                
                #store the parameters in a dictionary
                final_parameters['conv1_layer1_weights'] = session.run(conv1_layer1_weights)
                final_parameters['conv1_layer2_weights'] = session.run(conv1_layer2_weights)
                final_parameters['conv2_layer1_weights'] = session.run(conv2_layer1_weights)
                final_parameters['conv2_layer2_weights'] = session.run(conv2_layer2_weights)
                #final_parameters['conv1_layer1_biases'] = session.run(conv1_layer1_biases)
                #final_parameters['conv1_layer2_biases'] = session.run(conv1_layer2_biases)
                #final_parameters['conv2_layer1_biases'] = session.run(conv2_layer1_biases)
                #final_parameters['conv2_layer2_biases'] = session.run(conv2_layer2_biases)
                final_parameters['fc_layer1_weights'] = session.run(fc_layer1_weights)
                final_parameters['fc_layer2_weights'] = session.run(fc_layer2_weights)
                final_parameters['fc_layer1_biases'] = session.run(fc_layer1_biases)
                final_parameters['fc_layer2_biases'] = session.run(fc_layer2_biases)
                
                #number of back-props
                back+=1
                
                #make new memory 
                replay_memory = list()
                replay_labels = list()
                
            '''
            if(local_iters%400==0):
                print("Episode : {}, Score : {}, Iters : {}, Finish : {}".format(ep,total_score,local_iters,finish))
            '''
            local_iters += 1
            total_iters += 1
            
        scores.append(total_score)
        # print("Episode {} finished with score {}, result : {} board : {}, epsilon  : {}, learning rate : {} ".format(ep,total_score,finish,board,epsilon,session.run(learning_rate)))
          
        if(maximum < total_score):
            maximum = total_score
            episode = ep
        if((ep+1)%100==0):
            current_time = time.time()
            elapsed_time = current_time - start_time
            print("Episode {}-{} finished in {} seconds. Max score: {}. Loss: {}".format(ep-99, ep, elapsed_time, maximum, losses[len(losses)-1]))  


################################################################################################################

'''
save the records and outcomes
'''

# save scores and losses
save(path='./trained', name='/scores', lis=scores, mode='.txt')
save(path='./trained', name='/losses', lis=losses, mode='.txt')

# save weights
weights = ['conv1_layer1_weights','conv1_layer2_weights','conv2_layer1_weights','conv2_layer2_weights','fc_layer1_weights','fc_layer1_biases','fc_layer2_weights','fc_layer2_biases']
for w in weights:
    flatten = final_parameters[w].reshape(-1,1)
    save(path='./trained', name='/' + w, lis=flatten, mode='.csv')

################################################################################################################