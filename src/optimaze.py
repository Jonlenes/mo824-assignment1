import os
import sys
import numpy as np
import gurobipy as gp
import scipy.sparse as sp
import pandas as pd
from gurobipy import GRB
from time import time
from read_instance import load_instance, list_avaliable_instances
from tqdm import tqdm
from glob import glob


def print_results(model, x, y, z, factories, clients, machines, paper_type, mat_prima):
    print(f"Valor da função objetivo: {model.objVal}")
    print("\nQtd de Papeis \t\t Tipo de papel \t\t Máquina \t\t Fábrica ")

    for f in factories:
        for l in machines:
            for p in paper_type:
                if x[p,l,f].X > 0:
                    print(f"{x[p,l,f].X} \t\t\t\t {p+1} \t\t {l+1} \t\t {f+1}.")

    print("\n\nQtd de Papeis \t\t Tipo de papel \t\t Fábrica \t\t Cliente ")
    for f in factories:
        for j in clients:
            for p in paper_type:
                if y[p,f,j].X > 0:
                    print(f"{y[p,f,j].X} \t\t\t\t {p+1} \t\t {f+1} \t\t {j+1}.")

    print("\n\nQtd Mat. Prima \t\t Mat. Prima \t\t Máquina \t\t Fábrica ")
    for f in factories:
        for l in machines:
            for m in mat_prima:
                if z[m,l,f].X > 0:
                    print(f"{z[m,l,f].X} \t\t\t\t {m+1} \t\t {l+1} \t\t {f+1}.")


def build_model(instance, vtype=gp.GRB.INTEGER):
    n_J, n_F, n_L, n_M, n_P, D, r, R, C, p, t = instance

    # Create a new model
    model = gp.Model("production")
    model.setParam(gp.GRB.Param.OutputFlag, 0)

    # Generating the data
    paper_type = np.arange(n_P)
    machines = np.arange(n_L)
    factories = np.arange(n_F)
    clients = np.arange(n_J)
    raw_material = np.arange(n_M)

    # Naming the arrays
    paper_cost, trans_cost, capacity = p, t, C
    demands, unid_raw_material, unid_raw_mat_disp = D, r, R

    # Create variables
    # x_{p,l,f}
    x = model.addVars(paper_type, machines, factories, vtype=vtype, name="x")
    # y_{p,f,j}
    y = model.addVars(paper_type, factories, clients, vtype=vtype, name="y")
    # z_{m,l,f}
    z = model.addVars(raw_material, machines, factories, vtype=vtype, name="z")

    # Set objective: p @ x + t @ y
    model.setObjective(
        gp.quicksum(
            x[p, l, f] * paper_cost[p, l, f]
            for p in paper_type
            for l in machines
            for f in factories
        )
        + gp.quicksum(
            y[p, f, j] * trans_cost[p, f, j]
            for p in paper_type
            for f in factories
            for j in clients
        ),
        sense=gp.GRB.MINIMIZE,
    )

    # Grupo (1): Quantidade transportada <= quantidade produzida
    model.addConstrs(
        gp.quicksum(y[p, f, j] for j in clients)
        <= gp.quicksum(x[p, l, f] for l in machines)
        for p in paper_type
        for f in factories
    )

    # Grupo (2):  Quantidade produzida <= capacidade
    model.addConstrs(
        gp.quicksum(x[p, l, f] for p in paper_type) <= capacity[l, f]
        for l in machines
        for f in factories
    )

    # Grupo (3): Quantidade produzida >= demanda
    model.addConstrs(
        gp.quicksum(y[p, f, j] for f in factories) >= demands[j, p]
        for j in clients
        for p in paper_type
    )

    # Grupo (4): materia prima utilizada <= materia prima disponivel
    model.addConstrs(
        gp.quicksum(z[m, l, f] for l in machines) <= unid_raw_mat_disp[m, f]
        for m in raw_material
        for f in factories
    )

    # Grupo (5): Materia prima necessária <= materia prima utilizada
    model.addConstrs(
        gp.quicksum(x[p, l, f] * unid_raw_material[m, p, l] for p in paper_type)
        <= z[m, l, f]
        for m in raw_material
        for l in machines
        for f in factories
    )

    return model


def main(ins_folder):
    ins_filenames = list_avaliable_instances(os.path.join(ins_folder, "*.json"))
    results = pd.DataFrame(
        columns=["n_clients", "n_vars", "interger_cost", "interger_time", "relax_cost", "relax_time"]
    )
    vtypes = [gp.GRB.INTEGER, gp.GRB.CONTINUOUS]

    print("Starting experiments")
    for filename in tqdm(ins_filenames):
        instance = load_instance(filename)
        costs, times = [], []
        for vtype in vtypes:
            start_time = time()
            # Building the model
            model = build_model(instance, vtype)
            # Optimize model
            model.optimize()
            # Saving costs and times
            costs.append(model.objVal)
            times.append(time() - start_time)
        results = results.append(
            {
                "n_clients": instance[0],
                "n_vars": len(model.getVars()),
                "interger_cost": costs[0],
                "interger_time": round(times[0], 3),
                "relax_cost": costs[1],
                "relax_time": round(times[1], 3),
            },
            ignore_index=True,
        )
    results.to_csv("data/results.csv")

if __name__ == "__main__":
    ins_folder = "data"
    print_production = False
    if len(sys.argv) > 1:
        ins_folder = sys.argv[1]
    main(ins_folder)
