import os
import json
import numpy as np

def generate_instance(j):
    # Number of factories
    n_F = np.random.randint(j, 2*j)
    # Number of machines
    n_L = np.random.randint(5, 10)
    # Number of raw material types 
    n_M = np.random.randint(5, 10)
    # Number of paper types
    n_P = np.random.randint(5, 10)

    # demanda do cliente j em unidades de papel do tipo p;
    D = np.random.randint(10, 100, size=(j, n_P))
    
    # unidades de matéria-prima m necessárias para produzir
    # uma unidade de papel do tipo p na máquina l;
    r = np.random.randint(1, 10, size=(n_M, n_P, n_L))
    
    # unidades de matéria-prima m disponíveis na fábrica f;
    R = np.random.randint(100, 1000, size=(n_M, n_F))
    
    # capacidade disponível de produção, em unidades de papel, da máquina l na fábrica f;
    C = np.random.randint(10, 100, size=(n_L, n_F))
    
    # custo unitário de produção do papel tipo p utilizando a máquina l na fábrica f;
    p = np.random.randint(10, 100, size=(n_P, n_L, n_F))
    
    # custo unitário de transporte do papel tipo p partindo da fábrica f até o cliente j;
    t = np.random.randint(10, 20, size=(n_P, n_F, j))

    return [j, n_F, n_L, n_M, n_P, D, r, R, C, p, t]

def instance2json(ins):
    names = ['j', 'n_F', 'n_L', 'n_M', 'n_P', 'D', 'r', 'R', 'C', 'p', 't']
    dic_ins = {}
    for name, value in zip(names, ins):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dic_ins[name] = value
    return json.dumps(dic_ins)

def save_json(filename, str_json):
    with open(filename, 'w') as file:
        file.write(str_json)

def generate_and_save_all(save_folder='data'):
    os.makedirs(save_folder, exist_ok=True)

    # Amount of clients for each instance
    J = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    for index, j in enumerate(J):
       ins = generate_instance(j)
       json_ins = instance2json(ins)
       save_json(os.path.join(save_folder, f'instancia-{index}.json'), json_ins)

if __name__ == "__main__":
    generate_and_save_all()