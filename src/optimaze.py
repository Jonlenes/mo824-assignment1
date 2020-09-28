import numpy as np
import gurobipy as gp
import scipy.sparse as sp
from gurobipy import GRB
from time import time
from read_instance import load_instance


def print_results(model, full=False):
    if full:
        for v in model.getVars():
            print("%s %g" % (v.varName, v.x))
    print("Obj: %g" % model.objVal)

def add_demand_const(model, y, D, n_P, n_F, n_J):
    for j in range(D.shape[0]):
        for p in range(D.shape[1]):
            indexes = []
            for f in range(n_F):
                indexes.append(np.ravel_multi_index((p, f, j), dims=(n_P, n_F, n_J)))
            model.addConstr(sum(y[indexes]) >= D[j, p], name=f"demand_{j*p}")

def build_model(instance):
    n_J, n_F, n_L, n_M, n_P, D, r, R, C, p, t = instance

    # Create a new model
    model = gp.Model("production")

    # Create variables
    x = model.addMVar(shape=n_P*n_L*n_F, vtype=GRB.INTEGER, name="x")
    y = model.addMVar(shape=n_P*n_F*n_J, vtype=GRB.INTEGER, name="y")
    z = model.addMVar(shape=n_L*n_F*n_M, vtype=GRB.INTEGER, name="z")

    # Set objective
    model.setObjective(p.reshape(-1) @ x + t.reshape(-1) @ y, GRB.MINIMIZE)

    add_demand_const(model, y, D, n_P, n_F, n_J)

    return model

def main():
    instance = load_instance('data/instancia-0.json')
    model = build_model(instance)
    
    # Optimize model
    print('Starting optimization')
    start_time = time()
    model.optimize()
    print('Spent time: {:.2f}s'.format(time() - start_time))

    print_results(model)


if __name__ == "__main__":
    main()





