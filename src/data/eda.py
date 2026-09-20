#type: ignore
import config as c
import load_graph as lg
import parse_qa as pq

#just for check dataset consistency between QA entities and KG nodes

if __name__ == '__main__':

    g = lg.load_graph()
    qa = pq.parser()

    match = 0

    for item in qa:
        topics = item['topic_entity']
        if topics in g:
            match += 1

    print(f"{match/len(qa) * 100:.2f}")
     

