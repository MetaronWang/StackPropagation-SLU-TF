import random


def create_vocab(name):
    path = "./alphabet/" + name
    file = open(path + "_dict.txt", mode="r", encoding="utf8")
    lines = file.readlines()
    list = []
    dict = {}
    for i in lines:
        value, index = i.split("\t")
        list.append(value)
        dict[value] = int(index)
    return list, dict


def get_items(path, word_dict, intent_dict, slot_dict):
    file = open(path, mode="r", encoding="utf8")
    item = []
    lines = file.readlines()
    tmp_word = []
    tmp_slot = []
    for i in lines:
        if i == "\n":
            continue
        i = i.replace("\n", "", -1)
        line = i.split(" ")
        if (len(line)) == 1:
            item.append((tmp_word, [1 for j in range(len(tmp_word))], tmp_slot,
                         [intent_dict[line[0]] for j in range(len(tmp_word))]))
            tmp_slot = []
            tmp_word = []
        else:
            if line[0] in word_dict:
                tmp_word.append(word_dict[line[0]])
            else:
                tmp_word.append(1)
            tmp_slot.append(slot_dict[line[1]])
    return item


class DataProcessor:
    def __init__(self):
        self.intent_list, self.intent_dict = create_vocab("intent")
        self.word_list, self.word_dict = create_vocab("word")
        self.slot_list, self.slot_dict = create_vocab("slot")
        self.batch_size = 1
        self.data = {'train': get_items(
            "./data/snips/train.txt",
            self.word_dict,
            self.intent_dict,
            self.slot_dict,
        ), 'test': get_items(
            "./data/snips/test.txt",
            self.word_dict,
            self.intent_dict,
            self.slot_dict,
        ), 'dev': get_items(
            "./data/snips/dev.txt",
            self.word_dict,
            self.intent_dict,
            self.slot_dict,
        )}

    def get_data(self, name="train", batch_size=32):
        data = self.data[name]
        random.shuffle(data)
        word, seq, slot, intent = ([data[i][j] for i in range(len(data))]for j in range(4))
        batches = [(word[i:i + batch_size], seq[i:i + batch_size], slot[i:i + batch_size], intent[i:i + batch_size]) for i in
                   range(0, len(word), batch_size)]
        all_batch = []
        for batch in batches:
            max_len = max([len(item) for item in batch[0]])
            length = [len(item) for item in batch[0]]
            all_batch.append(
                ([[item + [0 for i in range(max_len - len(item))] for item in batch[j]] for j in range(len(batch))]+[length])
            )
        return all_batch


if __name__ == "__main__":
    # datasets = get_dataset(8, 73, 16, 10)
    data_processor = DataProcessor()
    data_processor.get_data('train')
