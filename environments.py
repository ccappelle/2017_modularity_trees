from __future__ import division, print_function
import math
import treenome
import numpy as np
CYL_RADIUS_BASE = .4
CYL_RADIUS_INCR = 0.4


def send_to_simulator(sim, cyl_types, tree, distance, origin=.5):

    if isinstance(cyl_types, str):
        temp = [0]*len(cyl_types)
        for i,character in enumerate(cyl_types):
            temp[i] = int(character)
    cyl_types = temp
    pos_and_dir = tree.get_leaf_pos_and_dir()
    cyl_ids = [0]*len(cyl_types)

    for i, cyl in enumerate(cyl_types):
        pos = pos_and_dir[i]['pos']
        direction = pos_and_dir[i]['dir']
        x,y,z = pos + direction*distance

        radius = cyl_types[i]*CYL_RADIUS_INCR + CYL_RADIUS_BASE
        cyl_ids[i] = sim.send_cylinder(x, y, z,
                          length=2*z, radius=radius,
                          r1=0, r2=0, r3=1,
                          capped=False)
        sim.send_is_seen_sensor(cyl_ids[i])
    return cyl_ids

if __name__ == '__main__':
    import pyrosim
    DEPTH = 2
    for i in range(1):
        root = treenome.Tree(np.array([0, 0, .5]),
                np.array([0, 1, 0]),
                child_angle=math.pi/5.0,
                my_depth=0,
                max_depth=DEPTH,
                rotation_angle=math.pi/5.0,
                my_id=0,
                parent_id=-1,
                child_decay=.6,
                rotation_decay=.6
                )

        sim = pyrosim.Simulator(play_blind=False, play_paused=False, eval_time=200, xyz=[0, 1, 5],
                                hpr=[90, -90, 0], debug=True)

        root.send_to_simulator(sim)

        send_to_simulator(sim, '1211', root, 5.0)

        sim.start()

        sim.wait_to_finish()
