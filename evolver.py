from __future__ import division, print_function
import copy

class Evolver(object):
    """Evolver base class

    Basic base class which maintains a populations genome, 
    id, and fitness. 

    Attributes
    ----------
    population      : dictionary
        The population of genomes. This base class provides 
        three standard keys to the dictionary: 'genome', 'my_id', and 
        'fitness'. New keys can be added for more complex evolutionary
        algorithms.

    genome_gen_func : function
        The genome generator function. When a new individual is born,
        It is created with this generator function.

    eval_func       : function
        Evaluation function. Should take in a list of genomes and output
        a fitness for each individual.

    mutation_func   : function
        Mutation function. Takes in a genome, mutates it, and returns it.
        Function must output the mutated genome.

    pop_size        : int
        The default size of the population.

    base_pop        : list of genomes
        A base starting population of genomes
    """

    def __init__(self, pop_size, genome_gen_func, eval_func, mutation_func, base_pop=None):
        self._generation = 0
        self.genome_gen_func = genome_gen_func
        self.mutation_func = mutation_func
        self.pop_size = pop_size
        self.eval_func = eval_func

        self._sorted = False
        self._next_id = 0

        self.population = []
        if base_pop:
            for indv in base_pop:
                self.add_genome_to_pop(base_pop)
        else:
            for i in range(pop_size):
                self.add_random_indv()

    def add_random_indv(self, genome_gen_func=None):
        """Adds a new individual to the population with fitness=None

        Parameters
        ----------
        genome_gen_func : function
            The generator for the new genome. Default is to use the 
            generator specified in the __init__() call.

        Returns
        -------
        bool 
            True if successful, False otherwise
        """
        assert genome_gen_func or self.genome_gen_func
        if not genome_gen_func:
            genome_gen_func = self.genome_gen_func

        self.population += [{'genome': genome_gen_func(),
                             'my_id': self._next_id, 'fitness': None}]

        self._next_id += 1

        return True

    def add_genome_to_pop(self, genome):
        """Adds a pre-specified genome to the population."""

        self.population += [{'genome': genome, 'my_id': self._next_id,
                             'fitness': None}]
        self._next_id += 1

        return True

    def age_population(self):
        for indv in self.population:
            indv['age'] += 1

    def copy_and_mutate(self, index, mutation_func=None):
        """Mutates the genome in the population at index"""
        assert mutation_func or self.mutation_func
        mutant = copy.deepcopy(self.population[index]['genome'])
        
        if not mutation_func:
            mutation_func = self.mutation_func

        mutant = mutation_func(mutant)

        return mutant

    def evaluate(self, eval_func=None):
        """Evaluates the un-evaluated genomes"""
        genomes_to_eval = []
        indexes = []

        if not eval_func:
            eval_func = self.eval_func

        for i, indv in enumerate(self.population):
            if (indv['fitness'] == None):
                genomes_to_eval += [indv['genome']]
                indexes += [i]

        fitness_vals = eval_func(genomes_to_eval)

        for i in range(len(indexes)):
            index = indexes[i]
            fitness = fitness_vals[i]
            self.population[index]['fitness'] = fitness

        return True

    def get_generation(self):
        return self._generation

    def set_generation(self, gen):
        self._generation = gen

    def sort_population(self, key=lambda k: k['fitness']):
        self.population.sort(key=key)
        self._sorted = True

        return True


class AFPO(Evolver):

    def __init__(self, pop_size, genome_gen_func, eval_func, mutation_func, base_pop=None):
        super(AFPO, self).__init__(pop_size, genome_gen_func, eval_func, mutation_func, base_pop)

    def add_random_indv(self, genome_gen_func=None):
        super(AFPO, self).add_random_indv(genome_gen_func)
        self.population[-1]['age'] = 0
        self.population[-1]['num_dominated_by'] = 0

    def add_genome_to_pop(self, genome, age=0):
        super(AFPO, self).add_genome_to_pop(genome)
        self.population[-1]['age'] = age
        self.population[-1]['num_dominated_by'] = 0

    def determine_dominated_by(self):
        """Determines how many other individuals dominate a genome"""

        # clear the num dominated by
        for indv in self.population:
            indv['num_dominated_by'] = 0

        # indv is dominated if older and poorer fitness than another
        for i in range(len(self.population)):
            indv1 = self.population[i]
            for j in range(i+1, len(self.population)):
                indv2 = self.population[j]

                if (indv1['fitness'] > indv2['fitness']):
                    if (indv1['age'] <= indv2['age']):
                        indv2['num_dominated_by'] += 1

                elif (indv2['fitness'] > indv1['fitness']):
                    if (indv2['age'] <= indv1['age']):
                        indv1['num_dominated_by'] += 1

                elif (indv1['fitness'] == indv2['fitness']):
                    if (indv1['age'] < indv2['age']):
                        indv2['num_dominated_by'] += 1
                    elif (indv1['age'] > indv2['age']):
                        indv1['num_dominated_by'] += 1
                    else:  # ages are equal
                        if (indv1['my_id'] > indv2['my_id']):
                            indv2['num_dominated_by'] += 1
                        else:
                            indv1['num_dominated_by'] += 1

    def evolve(self, num_gens=1000, fitness_threshold=None, checkpoint=None):
        """Evolves the population to the specified criteria"""
        assert num_gens != None or fitness_threshold != None, (
            'Must define number of '
            'generations to run or fitness '
            'threshold to achieve')
        best_indvs = []
        fitness_scores = []
        if num_gens:
            while(self._generation < num_gens):
                #print (self._generation)
                best = self.evolve_for_one_step()
                best_indvs += [best]
        else:
            best_fitness = float('-inf')
            while(best_fitness < fitness_threshold):
                best_indv = self.evolve_for_one_step()
                best_fitness = best_indv['fitness']
                best_indvs += [best_indv]

        return best_indvs

    def evolve_for_one_step(self):
        # increment generation
        self._generation += 1
        self._sorted = False

        # mutate pop
        self.mutate_population()

        # age population
        self.age_population()

        # inject new genome
        self.add_random_indv()

        # evaluate population
        self.evaluate()

        # count dominated
        self.determine_dominated_by()

        # kill back to population size
        self.remove_dominated()

        # print out
        self.print_out()

        # return best
        return self.population[0]

    def get_pareto_front(self):
        """returns the Pareto front of the population"""
        if not(self._sorted):
            self.sort_population()

        pareto_front = []
        non_dominated = True
        index = 0
        while(self.population[index]['num_dominated_by'] == 0):
            pareto_front += [self.population[index]]
            index += 1
        return pareto_front

    def mutate_population(self, mutation_func=None):
        pop_length = len(self.population)
        for i in range(pop_length):
            mutant = self.copy_and_mutate(i, mutation_func)
            self.add_genome_to_pop(mutant, age=self.population[i]['age'])

    def print_out(self):
        """prints out the current generation and best fitness"""
        pareto = len(self.get_pareto_front())
        best_fitness = self.population[0]['fitness']
        out_string = '{}: {}\n  {}: {},  {}: {}'.format(
            'gen', self._generation,
            'fitness', best_fitness,
            'Pareto size', pareto)

        print(out_string)

    def remove_dominated(self):
        """removes the dominated until population is reduced to orginal size"""

        if (self._sorted==False):
            self.sort_population()

        for i in range(len(self.population)-1, self.pop_size-1, -1):
            del self.population[i]

    def sort_population(self):
        """sorts by dominated by and fitness"""
        key = lambda k: (k['num_dominated_by'], -k['fitness'])
        super(AFPO, self).sort_population(key=key)

        return True


def sum_angles(tree):
    out = tree.rotation_angle

    if tree.children:
        for child in tree.children:
            out += sum_angles(child)

    return out 

def mutate_tree(tree):
    tree.mutate()
    return tree

def simple_eval(genomes):
    fitness_vals = []

    for genome in genomes:
        fitness_vals += [-sum_angles(genome)]
    return fitness_vals

if __name__ == '__main__':
    import treenome
    import tree_constants
    import environment
    import functools
    import numpy as np

    tree_generator = functools.partial(treenome.Tree, np.array([0, 0, .5]),
                                       np.array([0, 1, 0]),
                                       child_angle=np.pi/6.0,
                                       my_depth=0,
                                       max_depth=tree_constants.depth,
                                       rotation_angle=np.pi/6.0,
                                       )

    evolver = AFPO(100, tree_generator, eval_func=simple_eval, mutation_func=mutate_tree)

    indv = evolver.evolve(num_gens=5)[0]
    