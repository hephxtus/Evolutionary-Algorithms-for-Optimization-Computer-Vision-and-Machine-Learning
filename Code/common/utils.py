
import os
import random

from deap import gp

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

import pandas as pd

out_dir = "../Output/"
dataset_dir = "../Datasets/"


def get_dataset_path(dataset_name):
    # return absolute path to dataset
    return os.path.join(dataset_dir, dataset_name)
def get_output_path(question_name):
    return os.path.join(out_dir, question_name)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def output_plot(plot, question_name, plot_name):
    path = get_output_path(question_name)
    create_dir(path)
    # matplotlib.pyplot.savefig(os.path.join(path, plot_name))
    plot.savefig(os.path.join(path, plot_name) + ".jpg", bbox_inches='tight')


def save_features(features, question_name, file_name):
    path = get_output_path(question_name)
    create_dir(path)
    pd.DataFrame(features).to_csv(os.path.join(path, f"{file_name}.csv"), header=None, index=None)



def render(G, hofs, seed, question_name, toolbox, verbose=False, ):
    colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "grey"]

    for x, hof in enumerate(hofs):
        color = random.choice(colors)
        colors.remove(color)
        best_ind = toolbox.compile(expr=hof[0])

        if verbose:
            print(str(hof[0]))
        nodes, edges, labels = gp.graph(hof[0])
        base = G.number_of_nodes()
        G.add_nodes_from([base + n for n in nodes])
        G.add_edges_from([(base + ni, base + nj) for (ni, nj) in edges])
        # Add labels to nodes and split into a
        for n in nodes:
            G.get_node(base + n).attr["label"] = labels[n]
            G.get_node(base + n).attr["fillcolor"] = color
            # Shape
            # if is float Absolute hack job
            #check if label is a number
            if str(labels[n]).replace(".", "", 1).isdigit():
                G.get_node(base + n).attr["shape"] = "box"
            else:
                G.get_node(base + n).attr["shape"] = "ellipse"

        # Find nodes with no incoming edges
        if len(hofs) > 1:
            roots = []
            for n in nodes:
                if not any([e[1] == n for e in edges]):
                    roots.append(n)
            G.add_edge(x + 1, roots[0] + base)

    # G.layout(prog="dot")
    path = get_output_path(question_name)

    # Make sure output directory exists, create if not
    if not os.path.exists(path):
        os.makedirs(path)
    # add disconnected node with fitness and size
    G.get_node(0).attr["label"] = [f"f: {hof[0].fitness.values[0]} \n s: {hof[0].height}\n" for hof in hofs]
    G.draw(os.path.join(path, f"graph_{seed}.png"), prog="dot")
