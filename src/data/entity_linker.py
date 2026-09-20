#type: ignore
import load_graph as lg
import parse_qa as pq

class EntityLinker:
    def __init__(self, graph):
        #map for case-insensitive
        self.graph = graph
        self.lower_map = {entity.lower(): entity for entity in graph.keys()}

    def link(self, raw_entity):
        if not raw_entity:
            return None
        
        #match
        if raw_entity in self.graph:
            return raw_entity
        
        #match case-insensitive (es. "chris noonan" -> "Chris Noonan")
        raw_lower = raw_entity.lower().strip()
        if raw_lower in self.lower_map:
            return self.lower_map[raw_lower]
        
        #no match
        return None

if __name__ == '__main__':
    #load data
    g = lg.load_graph()
    qa = pq.parser()
    
    linker = EntityLinker(g)
    
    #test
    matched = 0
    
    for item in qa:
        raw_ent = item["topic_entity"]
        resolved_ent = linker.link(raw_ent)
        if resolved_ent is not None:
            matched += 1

    print(f"Linking: {matched}/{len(qa)} ({matched/len(qa) * 100:.2f}%)")