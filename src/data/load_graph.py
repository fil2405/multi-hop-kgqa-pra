#type: ignore
import config as c

def load_graph():

    g = {}

    with open(c.PATH_KB, 'r', encoding='utf-8') as f:
        for row in f:
            row = row.strip()

            if row:
                head, pred, tail = row.split('|')
                
                #directed edge: head -> pred -> tail
                if head not in g:
                    g[head] = {} #dict
                if pred not in g[head]:
                    g[head][pred] = [] #list
                g[head][pred].append(tail)

                #inverse edge: tail -> inv_pred -> head
                inv_pred = f"inv_{pred}"
                if tail not in g:
                    g[tail] = {}
                if inv_pred not in g[tail]:
                    g[tail][inv_pred] = []
                g[tail][inv_pred].append(head)
            else:
                continue

    return g

if __name__ == '__main__':

    g = load_graph()
    print("Total entities: ", len(g))

    #test directed edge
    test_movie = "Flags of Our Fathers"
    print(f"\nPredicates for '{test_movie}':", g.get(test_movie))

    #test inverse edge
    test_director = "Clint Eastwood"
    print(f"\nPredicates for '{test_director}':", g.get(test_director))
