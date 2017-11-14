import numpy as np
import math
import random
import pyrosim

def normalize(np_array):
    magnitude = np.sqrt(np.sum(np.multiply(np_array, np_array)))

    return np_array/magnitude


def rotate_xy(old_orientation, radians):
    rotation_mat = np.matrix([[math.cos(radians), -math.sin(radians), 0],
                              [math.sin(radians), math.cos(radians), 0],
                              [0, 0, 1]])
    return np.dot(rotation_mat, old_orientation)

class Tree(object):
    """Symmetric binary tree"""

    def __init__(self,
                 my_pos,
                 my_orientation,
                 child_angle,
                 my_depth, max_depth,
                 rotation_angle,
                 my_id=0, parent_id=-1,
                 child_decay=.75,
                 rotation_decay=.75,
                 length=0.5, radius=0.03,
                 network=None):

        self.base = my_pos[:]
        self.orientation = np.ravel(normalize(my_orientation))
        self.tip = self.base + self.orientation*length
        self.position = (self.tip + self.base)/2.0

        self.child_angle = child_decay*child_angle
        self.depth = my_depth
        self.id = my_id
        self.parent_id = parent_id
        self.length = length
        self.radius = radius
        self.rotation_angle = abs(rotation_angle)
        self.base_rotation = self.rotation_angle

        self.max_depth = max_depth

        self.network = network

        # hacky and badddd
        if self.depth == 0:
            rotation_modier = 1.0
        else:
            rotation_modier = rotation_decay

        if self.depth != max_depth:
            self.children = [0]*2
            child_orientation = [rotate_xy(self.orientation, child_angle),
                                 rotate_xy(self.orientation, -child_angle)]
            child_id = [self.id+1, self.id+2**(max_depth-self.depth)]

            for i in range(2):
                self.children[i] = Tree(my_pos=self.tip,
                                        my_orientation=child_orientation[i],
                                        child_angle=self.child_angle,
                                        my_depth=self.depth+1,
                                        max_depth=max_depth,
                                        rotation_angle=rotation_modier*self.rotation_angle,
                                        child_decay=child_decay,
                                        rotation_decay=rotation_decay,
                                        my_id=child_id[i],
                                        parent_id=self.id,
                                        length=self.length,
                                        radius=self.radius)
        else:
            self.children = None

    def mutate(self, body_prob=.5, network_prob=0.):
        """Mutates the tree's Body and brain with given probabilities"""
        if (self.depth==0 and self.network):
            self.network.mutate(network_prob)

        if (np.random.rand()< body_prob):
            self.rotation_angle += np.random.normal(0, self.base_rotation/2.0)
            if self.rotation_angle > self.base_rotation:
                self.rotation_angle = self.base_rotation
            elif self.rotation_angle < 0:
                self.rotation_angle = 0

        if self.children:
            for child in self.children:
                child.mutate(body_prob, 0)

    def get_leaf_pos_and_dir(self):
        """Returns the position and direction of the leaves of the tree"""
        if self.children:
            pos_and_orientation = []
            for child in self.children:
                pos_and_orientation.extend(child.get_leaf_pos_and_dir())

            return pos_and_orientation
        else:
            return [{'pos': self.position, 'dir': self.orientation}]

    def get_node(self, node_id):
        """Returns the node with id node_id"""
        if self.id == node_id:
            return self
        elif self.children[1].id <= node_id:
            return self.children[1].get_node(node_id)
        else:
            return self.children[0].get_node(node_id)

    def get_num_nodes(self):
        """Returns number of nodes"""
        return 2**(self.max_depth+1) - 1

    
    def send_to_simulator(self, sim):
        """Sends robot to pyrosim simulator sim"""
        actual_id = sim.send_cylinder(self.position[0],
                                      self.position[1],
                                      self.position[2],
                                      self.orientation[0],
                                      self.orientation[1],
                                      self.orientation[2],
                                      length=self.length,
                                      radius=self.radius)

        assert actual_id == self.id, str(actual_id) + " " + str(self.id)
        # if root -> connect joint to world
        if (self.depth == 0):
            second_id = actual_id
            first_id = pyrosim.Simulator.WORLD
        else:
            first_id = self.parent_id
            second_id = actual_id
        # print(actual_id, -self.rotation_angle, self.rotation_angle)
        sim.send_hinge_joint(first_id, second_id,
                             x=self.base[0], y=self.base[1], z=self.base[2],
                             n1=0, n2=0, n3=1,
                             lo=-self.rotation_angle, hi=self.rotation_angle,
                             speed=2.0,
                             torque=1000)

        if self.children:
            for child in self.children:
                child.send_to_simulator(sim)
        else:
            sim.send_ray_sensor(actual_id,
                                x=self.tip[0], y=self.tip[1], z=self.tip[2],
                                r1=self.orientation[0],
                                r2=self.orientation[1],
                                r3=self.orientation[2],
                                max_distance=20.0)

        if (self.depth == 0 and self.network != None):
            self.network.send_to_simulator()

    def __repr__(self):
        out = '{} {}'.format(self.id, str(self.position))

        if self.children:
            for child in self.children:
                out = '{} ,[{}]'.format(out, repr(child))
        return out

if __name__ == '__main__':
    import pyrosim

    DEPTH = 2
    root = Tree(np.array([0, 0, .5]),
                np.array([0, 1, 0]),
                child_angle=math.pi/4.0,
                my_depth=0,
                max_depth=DEPTH,
                rotation_angle=math.pi/4.0,
                rotation_decay=.6,
                child_decay=.6
                )
    sim = pyrosim.Simulator(play_blind=False, play_paused=False, eval_time=200)

    root.send_to_simulator(sim)

    sim.send_bias_neuron()
    #for i in range(1):
    for i in range(root.get_num_nodes()):
        sim.send_motor_neuron(i)
        if i == 1 or i==2 or i==5:
            sim.send_synapse(0, i+1, +1.0)
        if i == 4 or i==3 or i==6:
            sim.send_synapse(0, i+1, 1.0)
        # if i <4:
        #     sim.send_synapse(0,i+1,-1.0)
        # else:
        #     sim.send_synapse(0,i+1,+1.0)
    
    sim.start()
    sim.wait_to_finish()
