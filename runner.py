from __future__ import division, print_function
import multiprocessing
import numpy as np
import pyrosim
import environments
import networkx as nx
import networks
import treenome
import evolver
from functools import partial
from multiprocessing.pool import ThreadPool
import dill as pickle 
import copy

BATCH_SIZE = multiprocessing.cpu_count()*2
Distance = 6
TIME_EVAL_START = .25
DEPTH = 2
linear = lambda x: x
neg_abs = lambda x: -np.abs(x)
neg_square = lambda x: -np.square(x)
sqrt_abs = lambda x: np.sqrt(np.abs(x))
neg_sqrt_abs = lambda x: -np.sqrt(np.abs(x))
DT = 0.01
EVAL_TIME = 200

activation_funcs = [np.sin, np.abs, np.square, neg_square, linear, neg_abs, sqrt_abs, neg_sqrt_abs ,np.tanh]
TREE = treenome.Tree(np.array([0, 0, .5]),
                np.array([0, 1, 0]),
                child_angle=np.pi/4.0,
                my_depth=0,
                max_depth=DEPTH,
                rotation_angle=np.pi/2.0,
                rotation_decay=.6,
                child_decay=.6
                )

def create_embedding(tree, num_hidden=0):
    g = nx.DiGraph()
    g.pos= {}
    g.color = {}
    node_index = 0
    motor_index = 0
    sensor_index = 0
    pos = {}
    for i in range(tree.get_num_nodes()):
        node = tree.get_node(i)
        if node.children:
            is_leaf = False
        else:
            is_leaf = True
            
        center = node.tip
        # add motors
        g.add_node(node_index, x=center[0], y=center[1], assoc_id=motor_index, neuron_type='motor')
        g.pos[node_index] = [center[0], center[1]]
        g.color[node_index] = (70/255,130/255,180/255)
        node_index += 1
        motor_index += 1

        # add sensor if leaf
        if is_leaf:
            g.add_node(node_index,x=center[0], y=center[1], assoc_id=sensor_index, neuron_type='sensor')
            g.pos[node_index] = [center[0]+.1*node.orientation[0], center[1]+.1*node.orientation[1]]
            g.color[node_index] = (255/255,248/255,220/255)
            node_index += 1
            sensor_index += 1
        # hidden
        # none yet

        # print (node.tip)

    # for v in g:
    #     print (g, v, g[v])
    # print ([g.color[v] for v in g])
    
    return g

def init_genome(i=0):
    """Initializes the treenome"""

    g = {}
    g['tree'] = copy.deepcopy(TREE)
    g['cppn'] = networks.CPPN(['x1','y1','x2','y2','bias'],['w','on'], activation_funcs)
    g['cppn'].batch_mutate(add_nodes=10,add_edges=10,mutate_functions=20,mutate_edges=50)
    g['embedding'] = create_embedding(TREE)
    g['motor_limits'] = np.ones(TREE.get_num_nodes())

    calc_network_weights(g)

    return g

def draw_embedding(genome):
    """Draws the network"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cm = plt.get_cmap('seismic')
    g = genome['embedding']
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    edges = g.edges()

    weights = [g[u][v]['weight'] for u,v in edges]
    cax = nx.draw(g, arrows=False, width=4, ax=ax, pos=[g.pos[v] for v in g], 
               node_color=[g.color[v] for v in g], edge_color=weights, edge_cmap=cm,
               edge_vmin=-1, edge_vmax=1)
    # ax.axis('on')
    ax.set_aspect('equal')
    ax.set_xlim([-1,1])
    ax.set_ylim([.4,1.6])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=1.0)
    sm = plt.cm.ScalarMappable(cmap=cm)
    sm._A = []
    #ax.set_facecolor((1,0,0))
    plt.colorbar(sm, cax=cax)    
    plt.show()

def mutation(genome, experiment):
    """Mutates genome based on which experiment is being preformed"""
    tree = genome['tree']
    cppn = genome['cppn']
    embedding = genome['embedding']

    prob_add_edge = .1
    prob_remove_edge = .05
    prob_add_node = .05
    prob_remove_node = .025

    prob_mutate_func = .1
    prob_mutate_edge_weight = .1

    num_nodes = nx.number_of_nodes(cppn.graph)
    num_synapses = nx.number_of_edges(cppn.graph)

    num_edges_to_add = 0
    num_edges_to_remove = 0
    num_nodes_to_add = 0
    num_nodes_to_remove = 0
    num_funcs_to_mutate = 0
    num_weights_to_mutate = 0

    # single mutations
    if (np.random.rand()<prob_add_edge):
        num_edges_to_add += 1
    if (np.random.rand()<prob_remove_edge):
        num_edges_to_remove += 1
    if (np.random.rand()<prob_add_node):
        num_nodes_to_add += 1
    if (np.random.rand()<prob_remove_node):
        num_nodes_to_remove += 1

    for _ in range(num_nodes):
        if (np.random.rand()<prob_mutate_func):
            num_funcs_to_mutate += 1

    for _ in range(num_synapses):
        if (np.random.rand()<prob_mutate_edge_weight):
            num_weights_to_mutate += 1

    cppn.batch_mutate(add_nodes=num_nodes_to_add, add_edges=num_edges_to_add,
                      remove_nodes=num_nodes_to_remove, remove_edges=num_edges_to_remove,
                      mutate_functions=num_funcs_to_mutate, mutate_edges=num_weights_to_mutate)

    calc_network_weights(genome)

    if experiment == 'morphology':
        pass
    return genome

def calc_network_weights(genome):
    """calculates the weights of the network using cppns"""
    embedding = genome['embedding']
    cppn = genome['cppn']

    for node1 in nx.nodes(embedding):
        for node2 in nx.nodes(embedding):
            if (node1!=node2) and (embedding.node[node2]['neuron_type']!='sensor'):
                inputs = {}
                inputs['x1'] = embedding.node[node1]['x']
                inputs['y1'] = embedding.node[node1]['y']
                inputs['x2'] = embedding.node[node1]['x']
                inputs['y2'] = embedding.node[node1]['y']
                inputs['bias'] = 1.0

                outputs = cppn.calculate_outputs(inputs)
                # weight = outputs['w']
                # embedding.add_edge(node1,node2,weight=weight)
                if (outputs['on']>.5):
                    weight = outputs['w']
                    #embedding.remove_edge(node1, node2)
                    # print(node1, node2, weight)
                    embedding.add_edge(node1, node2, weight=weight)
                else:
                    if (embedding.has_edge(node1, node2)):
                        embedding.remove_edge(node1, node2)

def send_indv_to_sim(sim, genome,env,eval_time):
    """Creates the necessary objects in pyrosim to send to simulator"""
    tree = genome['tree']
    cppn = genome['cppn']
    embedding = genome['embedding']

    # calc_network_weights(genome) moved to init and mutation
    # sim = pyrosim.Simulator(play_blind=True, play_paused=False, eval_time=eval_time, debug=False)
    tree.send_to_simulator(sim)

    for node_index in nx.nodes(embedding):
        curr_neuron = embedding.node[node_index]
        if (curr_neuron['neuron_type']=='motor'):
            # print(node_index, curr_neuron['assoc_id'], 'motor')
            sim.send_motor_neuron(joint_id=curr_neuron['assoc_id'])
        elif(curr_neuron['neuron_type']=='hidden'):
            pass
        elif(curr_neuron['neuron_type']=='sensor'):
            sim.send_sensor_neuron(sensor_id=curr_neuron['assoc_id'])
            #print(node_index, curr_neuron['assoc_id'], 'sensor')
    for synapse in embedding.edges_iter(data='weight'):
        source, target, weight = synapse
        sim.send_synapse(source, target, weight)

    if env:
        environments.send_to_simulator(sim, env, tree, Distance)

def eval_indv(genome, env,eval_time):
    """Evaluates one indv in one environment"""

    tree = genome['tree']
    cppn = genome['cppn']
    embedding = genome['embedding']

    calc_network_weights(genome)
    sim = pyrosim.Simulator(play_blind=True, play_paused=False, eval_time=eval_time, debug=False, dt=DT)
    tree.send_to_simulator(sim)

    for node_index in nx.nodes(embedding):
        curr_neuron = embedding.node[node_index]
        if (curr_neuron['neuron_type']=='motor'):
            # print(node_index, curr_neuron['assoc_id'], 'motor')
            sim.send_motor_neuron(joint_id=curr_neuron['assoc_id'])
        elif(curr_neuron['neuron_type']=='hidden'):
            sim.send_hidden_neuron()
        elif(curr_neuron['neuron_type']=='sensor'):
            sim.send_sensor_neuron(sensor_id=curr_neuron['assoc_id'])
            #print(node_index, curr_neuron['assoc_id'], 'sensor')
    for synapse in embedding.edges_iter(data='weight'):
        source, target, weight = synapse
        sim.send_synapse(source, target, weight)

    if env:
        environments.send_to_simulator(sim, env, tree, Distance)
    sim.start()

    raw_data = sim.wait_to_finish()
    return fitness_func(raw_data,env)

def multi_eval(index,genomes,env, eval_time):
    """Evaluates one genome and index in population in thread safe manor"""
    return eval_indv(genomes[index], env, eval_time)

def eval_pop(genomes, envs, eval_time):
    """Uses parallel threads to evaluate the population"""

    num_genomes = len(genomes)
    fitness_matrix = np.zeros((num_genomes, len(envs)))

    for i,env in enumerate(envs):
        send_func = partial(multi_eval, genomes=genomes, env=env, eval_time=eval_time)
        p = ThreadPool(4)

        fitness_matrix[:,i] = p.map(send_func, range(num_genomes) )

    return fitness_matrix    

def send_pop_to_simulator(genomes, envs, eval_time, batch_size=BATCH_SIZE):
    """Sends whole pop to list of environments"""

    num_genomes = len(genomes)
    fitness_matrix = np.zeros((num_genomes, len(envs)))
    for i,env in enumerate(envs):
        sims=[0]*batch_size
        genomes_to_eval = num_genomes
        while (genomes_to_eval>0):

            base = num_genomes-genomes_to_eval
            for j in range(min(genomes_to_eval,batch_size)):
                sims[j] = pyrosim.Simulator(play_blind=True, eval_time=eval_time)
                send_indv_to_sim(sims[j],genomes[base+j],env,eval_time)
                sims[j].start()

            for j in range(min(genomes_to_eval,batch_size)):
                data = sims[j].wait_to_finish()
                fitness_matrix[base+j, i] = look_at_fitness_func(data,env)
            genomes_to_eval -= batch_size
            # print (genomes_to_eval)

    # sort envs by fitness scores, worst -> best
    
    sorted_fitness = np.sort(fitness_matrix, axis=1)

    # create weight vector for weighted avg
    weight_vec = 1/(np.power(2, np.arange(1,len(envs)+1)))
    if len(envs) > 1:
        weight_vec[-1] = weight_vec[-2]

    #print(sorted_fitness)
    # take the weighted avg of the envs
    fitness = np.dot(sorted_fitness, weight_vec)

    return fitness

def look_at_fitness_func(raw_sensor_data, env):
    """Computes the time spent 'looking at' desired objects"""
    num_sensors,_,time= np.shape(raw_sensor_data)
    start_time = int(time*TIME_EVAL_START)
    time -= start_time

    is_seen = raw_sensor_data[num_sensors-len(env):, 0, start_time:]
    fitness = 0
    max_possible = 0
    min_possible = 0
    for i,cyl_str in enumerate(env):
        cyl = int(cyl_str)
        if cyl%2 ==1:
            multiplier = -1
            min_possible -= time
        else:
            multiplier = 1
            max_possible += time

        
        fitness += (np.sum(is_seen[i,:]*multiplier))
        #print(i,cyl,multiplier, np.sum(is_seen[i,:]),max_possible, min_possible, fitness)
    fitness = abs((fitness-min_possible)/(max_possible-min_possible))

    return fitness

def play(genome,env,eval_time):
    """Plays and shows the genome for the designated environment and eval time"""
    sim= pyrosim.Simulator(play_blind=False,play_paused=True, eval_time=eval_time, dt=DT)
    send_indv_to_sim(sim,genome,env,eval_time)
    sim.start()
    sim.wait_to_finish()

def save(evolution_run, file_name, checkpoint=None):
    list_to_save = []
    if (checkpoint == None):
        checkpoint = len(evolution_run)+1
    for generation, best_indv_data in enumerate(evolution_run):
        if (generation%checkpoint == 0 or generation == len(evolution_run)-1):
            gen_dictionary = {'generation': generation, 'fitness': best_indv_data['fitness'], 'genome':best_indv_data['genome']}
        else:
            gen_dictionary = {'generation': generation, 'fitness': best_indv_data['fitness']}
        
        list_to_save.append(gen_dictionary)

    with open(file_name,'w') as f:
        pickle.dump(list_to_save, f)
    print ('saved', list_to_save)

def _test_fitness(pop_size, envs=['0000','1111']):
    POP_SIZE = 30
    genomes = [0]*POP_SIZE
    # for i in range(POP_SIZE):
    p = ThreadPool()
    genomes = p.map(init_genome, range(POP_SIZE))
    fitness = send_pop_to_simulator(genomes, envs, eval_time=200)
    best = np.argmax(fitness)
    print(fitness[best])
    worst = np.argmin(fitness)
    print(fitness[worst])

def _test_embedding():
    import matplotlib.pyplot as plt
    g = init_genome()
    draw_embedding(g)
    g['cppn'].batch_mutate(add_nodes=1,add_edges=1,mutate_functions=1,mutate_edges=5)
    calc_network_weights(g)
    draw_embedding(g)

def _simple_evolve():
    eval_func = partial(send_pop_to_simulator, envs=['1110', '1101', '1011', '0111'], eval_time=EVAL_TIME)
    mutation_func = partial(mutation, experiment=1)
    genome_gen_func = init_genome

    evo_trial = evolver.AFPO(40, genome_gen_func=genome_gen_func, eval_func=eval_func, mutation_func=mutation_func)
    best_indvs = evo_trial.evolve(num_gens=5)
    #print (best_indv)
    play(best_indvs[-1]['genome'], '1111', EVAL_TIME)
    play(best_indvs[-1]['genome'], '1000', EVAL_TIME)
    #save(best_indvs, 'data/test_00.dat')

    #with open('data/test_00.dat','r') as f:
    #    best_loaded = pickle.load(f)

    #play(best_loaded[-1]['genome'], '0000', 100)
if __name__=="__main__":

    #_test_fitness(40)
    #_test_embedding()
    _simple_evolve()

