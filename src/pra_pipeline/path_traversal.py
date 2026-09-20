#type: ignore
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SRC_DIR / "data"

for p in [str(SRC_DIR), str(DATA_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import data.load_graph as lg
import data.parse_qa as pq

def prob(g, head, rel):

    current_nodes = {head: 1.0}

    for r in rel:
        next_nodes = {}

        for node, p in current_nodes.items():
            if r in g[node]:
                neighbors = g[node][r]
                k = p / len(neighbors)

                for target in neighbors:
                    if target in next_nodes:
                        next_nodes[target] += k
                    else:
                        next_nodes[target] = k

        current_nodes = next_nodes

    return current_nodes

if __name__ == '__main__':
   
    g = lg.load_graph()

    head = 'Inception'
    path = ['starred_actors', 'inv_starred_actors', 'directed_by'] 

    print(prob(g, head, path))