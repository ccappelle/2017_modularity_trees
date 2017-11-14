from __future__ import print_function, division
import random
import numpy as np
import networkx as nx
from copy import deepcopy


class CPPN(object):
    """A Compositional Pattern Producing Network"""

    def __init__(self, input_names, output_names, activation_functions, output_function=np.tanh):
        self.input_node_names = input_names
        self.output_node_names = output_names
        self.activation_functions = activation_functions

        self.graph = nx.DiGraph()
        self.set_minimal_graph(output_function)
        # self.mutate()

    def calculate_outputs(self, input_dict):
        for node in nx.nodes(self.graph):
            self.graph.node[node]['evaluated'] = False
            self.graph.node[node]['value'] = 0
        self.set_input_node_states(input_dict)

        out_dict = {}
        for output_node in nx.nodes(self.graph):
            if self.graph.node[output_node]['type'] == 'output':
                out_dict[output_node] = self.calculate_node_value(output_node)
        return out_dict

    def calculate_node_value(self, name):
        if self.graph.node[name]['evaluated']:
            return self.graph.node[name]['value']

        input_edges = self.graph.in_edges(nbunch=[name])
        new_value = 0

        for edge in input_edges:
            from_node, this_node = edge
            new_value += self.calculate_node_value(from_node)*self.graph.edge[
                from_node][this_node]['weight']

        self.graph.node[name]['value'] = self.graph.node[
            name]['function'](new_value)
        return self.graph.node[name]['value']

    def set_minimal_graph(self, output_function):
        """Create a simple graph with each input attached to each output"""
        for name in self.input_node_names:
            self.graph.add_node(name, type='input', function=None)
            self.graph.node[name]['value'] = 0
            self.graph.node[name]['evaluated'] = False

        for name in self.output_node_names:
            self.graph.add_node(name, type='output', function=output_function)
            self.graph.node[name]['value'] = 0
            self.graph.node[name]['evaluated'] = False

        for input_node in nx.nodes(self.graph):
            if self.graph.node[input_node]['type'] == 'input':
                for output_node in nx.nodes(self.graph):
                    if self.graph.node[output_node]['type'] == 'output':
                        self.graph.add_edge(
                            input_node, output_node, weight=0.0)

    def set_input_node_states(self, input_dict):
        """Sets the initial values of the input nodes"""

        for name in input_dict:
            self.graph.node[name]['value'] = input_dict[name]
            self.graph.node[name]['evaluated'] = True

    ###############################################
    #   Mutation functions
    ###############################################

    def add_random_node(self):
        # choose two random nodes (between which a link could exist)
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_edge = random.choice(self.graph.edges())
        node1 = this_edge[0]
        node2 = this_edge[1]

        # create a new node hanging from the previous output node
        new_node_index = self.get_max_hidden_node_index()
        self.graph.add_node(new_node_index, type="hidden",
                            function=random.choice(self.activation_functions))
        # random activation function here to solve the problem with admissible
        # mutations in the first generations
        self.graph.add_edge(new_node_index, node2, weight=1.0)

        # if this edge already existed here, remove it
        # but use it's weight to minimize disruption when connecting to the
        # previous input node
        if (node1, node2) in nx.edges(self.graph):
            weight = self.graph.edge[node1][node2]["weight"]
            self.graph.remove_edge(node1, node2)
            self. graph.add_edge(node1, new_node_index, weight=weight)
        else:
            self.graph.add_edge(node1, new_node_index, weight=1.0)
            # weight 0.0 would minimize disruption of new edge
            # but weight 1.0 should help in finding admissible mutations in the
            # first generations
        return ""

    def remove_random_node(self):
        hidden_nodes = list(set(self.graph.nodes()) -
                            set(self.input_node_names) - set(self.output_node_names))
        if len(hidden_nodes) == 0:
            return False
        this_node = random.choice(hidden_nodes)

        # if there are edge paths going through this node, keep them connected
        # to minimize disruption
        incoming_edges = self.graph.in_edges(nbunch=[this_node])
        outgoing_edges = self.graph.out_edges(nbunch=[this_node])

        # multiply dangling edges together to form new edge
        for incoming_edge in incoming_edges:
            for outgoing_edge in outgoing_edges:
                w = self.graph.edge[incoming_edge[0]][this_node]["weight"] * \
                    self.graph.edge[this_node][outgoing_edge[1]]["weight"]
                self.graph.add_edge(
                    incoming_edge[0], outgoing_edge[1], weight=w)

        self.graph.remove_node(this_node)
        return True

    # def add_random_edge2(self):
    #     non_edges = list(nx.non_edges(self.graph))
    #     for n in non_edges:
    #         print (n)

    def add_random_edge(self):

        done = False
        attempt = 0
        while not done:
            done = True

            # choose two random nodes (between which a link could exist, *but
            # doesn't*)
            node1 = random.choice(self.graph.nodes())
            node2 = random.choice(self.graph.nodes())
            while (not self.new_edge_is_valid(node1, node2)) and attempt < 999:
                node1 = random.choice(self.graph.nodes())
                node2 = random.choice(self.graph.nodes())
                attempt += 1
            if attempt >= 999:  # no valid edges to add found in 1000 attempts
                return False

            # create a link between them
            if random.random() > 0.5:
                self.graph.add_edge(node1, node2, weight=0.1)
            else:
                self.graph.add_edge(node1, node2, weight=-0.1)

            # If the link creates a cyclic graph, erase it and try again
            if self.has_cycles():
                self.graph.remove_edge(node1, node2)
                done = False
                attempt += 1
            if attempt >= 999:
                return False
        return True

    def batch_mutate(self, add_nodes=0, add_edges=0, remove_nodes=0, remove_edges=0, mutate_functions=0, mutate_edges=0):
        for _ in xrange(add_nodes):
            self.add_random_node()

        for _ in xrange(add_edges):
            self.add_random_edge()

        for _ in xrange(remove_nodes):
            self.remove_random_node()

        for _ in xrange(remove_edges):
            self.remove_random_edge()

        for _ in xrange(mutate_functions):
            self.mutate_function()

        for _ in xrange(mutate_edges):
            self.mutate_edge_weight()

    def remove_random_edge(self):
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_link = random.choice(self.graph.edges())
        self.graph.remove_edge(this_link[0], this_link[1])
        return ""

    def mutate_function(self, node_id=None):
        if node_id == None:
            this_node = random.choice(self.graph.nodes())
            while this_node in self.input_node_names:
                this_node = random.choice(self.graph.nodes())
        else:
            this_node = self.graph.node[this_node]

        old_function = self.graph.node[this_node]["function"]
        while self.graph.node[this_node]["function"] == old_function:
            self.graph.node[this_node]["function"] = random.choice(
                self.activation_functions)
        return old_function.__name__ + "-to-" + self.graph.node[this_node]["function"].__name__

    def mutate_edge_weight(self, mutation_std=0.5):
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_edge = random.choice(self.graph.edges())
        node1 = this_edge[0]
        node2 = this_edge[1]
        old_weight = self.graph[node1][node2]["weight"]
        new_weight = old_weight
        while old_weight == new_weight:
            new_weight = random.gauss(old_weight, mutation_std)
            new_weight = max(-1.0, min(new_weight, 1.0))
        self.graph[node1][node2]["weight"] = new_weight
        return float(new_weight - old_weight)

    ###############################################
    #   Helper functions for mutation
    ###############################################

    def prune_network(self):
        """Remove erroneous nodes and edges post mutation."""
        done = False
        while not done:
            done = True
            for node in self.graph.nodes():
                if len(self.graph.in_edges(nbunch=[node])) == 0 and \
                        node not in self.input_node_names and \
                        node not in self.output_node_names:
                    self.graph.remove_node(node)
                    done = False

            for node in self.graph.nodes():
                if len(self.graph.out_edges(nbunch=[node])) == 0 and \
                        node not in self.input_node_names and \
                        node not in self.output_node_names:
                    self.graph.remove_node(node)
                    done = False

    def has_cycles(self):
        """Return True if the graph contains simple cycles (elementary circuits).
        A simple cycle is a closed path where no node appears twice, except that the first and last node are the same.
        """
        return sum(1 for _ in nx.simple_cycles(self.graph)) != 0

    def get_max_hidden_node_index(self):
        max_index = 0
        for input_node in nx.nodes(self.graph):
            if self.graph.node[input_node]["type"] == "hidden" and int(input_node) >= max_index:
                max_index = input_node + 1
        return max_index

    def new_edge_is_valid(self, node1, node2):
        if node1 == node2:
            return False
        if self.graph.node[node1]['type'] == "output":
            return False
        if self.graph.node[node2]['type'] == "input":
            return False
        if (node2, node1) in nx.edges(self.graph):
            return False
        if (node1, node2) in nx.edges(self.graph):
            return False
        return True


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    gaussian = lambda x: np.exp(-.5 * x**2)
    linear = lambda x: x
    square = lambda x: x**2
    neg_square = lambda x: -x**2
    neg = lambda x: -x
    activation_functions = [np.sin, np.tanh, gaussian, linear]
    inputs = ['x1', 'x2', 'b']
    outputs = ['w']

    fig, ax = plt.subplots(3, 1, figsize=(5, 10))

    for j in range(3):
        net = CPPN(inputs, outputs, activation_functions)
        for i in range(10):
            net.add_random_node()

        for i in range(10):
            net.add_random_edge()

        for i in range(5):
            net.mutate_function()

        for i in range(20):
            net.mutate_edge_weight()

        grid_size = 10
        matrix = np.zeros((grid_size, grid_size))
        for x1 in range(grid_size):
            for x2 in range(grid_size):
                input_vals = {'x1': x1, 'x2': x2, 'b': 1.0}
                output_dict = net.calculate_outputs(input_vals)
                matrix[x1, x2] = output_dict['w']

        ax[j].pcolormesh(matrix, cmap='Blues', vmin=-1, vmax=1)
        ax[j].set_aspect('equal')
    # plt.colorbar(ticks=np.linspace(-1,1,5, endpoint=True))
        # plt.savefig('gradient_04.png')
    plt.show()
