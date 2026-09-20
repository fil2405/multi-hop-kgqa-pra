#type: ignore
import config as c
import re 

def parser(hop=1, split="train"):
    qa = []
    path = c.get_qa_path(hop, split)

    with open(path, 'r', encoding='utf-8') as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            question_text, answer_text = row.split('\t')

            #extract topic entity with regex
            entities = re.findall(r'\[(.*?)\]', question_text)
            topic_entity = entities[0] if entities else None

            answers = answer_text.split('|')

            qa.append({
                "question": question_text,
                "topic_entity": topic_entity,
                "answers": answers
            })

    return qa
        



if __name__ == '__main__':
    qa = parser()

    #test
    print("Total questions:", len(qa))
    print("\nFirst sample:\n", qa[0])