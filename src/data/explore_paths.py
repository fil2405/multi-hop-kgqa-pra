#type: ignore
import config as c
import load_graph as lg
import parse_qa as pq

if __name__ == '__main__':

    g = lg.load_graph()

    print(g['Inception'])