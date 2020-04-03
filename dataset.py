import os

import tensorflow as tf


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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


def getDate(path, batch_size, word_dict, intent_dict, slot_dict):
    file = open(path, mode="r", encoding="utf8")
    word = []
    seq = []
    slot = []
    intent = []
    lines = file.readlines()
    tmp_word = []
    tmp_slot = []
    for i in lines:
        if i == "\n":
            continue
        i = i.replace("\n", "", -1)
        line = i.split(" ")
        if (len(line)) == 1:
            word.append(tmp_word)
            slot.append(tmp_slot)
            intent.append([intent_dict[line[0]] for j in range(len(tmp_word))])
            seq.append([1 for k in range(0, len(tmp_word))])
            tmp_slot = []
            tmp_word = []
        else:
            if line[0] in word_dict:
                tmp_word.append(word_dict[line[0]])
            else:
                tmp_word.append(1)
            tmp_slot.append(slot_dict[line[1]])
    for i in range(len(word) // batch_size):
        max_len = max(
            [len(word[j]) for j in range(i * batch_size, (i + 1) * batch_size)]
        )
        for j in range(i * batch_size, (i + 1) * batch_size):
            while len(word[j]) < max_len:
                word[j].append(0)
                intent[j].append(0)
                slot[j].append(0)
                seq[j].append(0)

    if len(word) % batch_size > 0:
        max_len = max(
            [
                len(word[j])
                for j in range(len(word) - (len(word) % batch_size), len(word))
            ]
        )
        for j in range(len(word) - (len(word) % batch_size), len(word)):
            while len(word[j]) < max_len:
                word[j].append(0)
                intent[j].append(0)
                slot[j].append(0)
                seq[j].append(0)
        while len(word) % batch_size > 0:
            word.append(word[len(word) - 1])
            slot.append(slot[len(slot) - 1])
            intent.append(intent[len(intent) - 1])
            seq.append(seq[len(seq) - 1])
    return word, seq, intent, slot


def generator(data):
    def gen():
        for _ in data:
            yield _

    return gen


def generator_out(a, b):
    def gen():
        for x, y in zip(a, b):
            yield (x, y)

    return gen


def get_dataset(intent_num, slot_num, batch_size, padd_size):
    intent_list, intent_dict = create_vocab("intent")
    word_list, word_dict = create_vocab("word")
    slot_list, slot_dict = create_vocab("slot")
    train_word, train_seq, train_intent, train_slot = getDate(
        "./data/snips/train.txt",
        batch_size,
        word_dict,
        intent_dict,
        slot_dict,
    )
    word_dataset = tf.data.Dataset.from_generator(
        generator=generator(train_word), output_types=tf.int32, output_shapes=[None]
    )
    intent_dataset = tf.data.Dataset.from_generator(
        generator=generator(train_intent), output_types=tf.int32, output_shapes=[None]
    )
    slot_dataset = tf.data.Dataset.from_generator(
        generator=generator(train_slot), output_types=tf.int32, output_shapes=[None]
    )
    word_dataset = word_dataset.repeat().padded_batch(
        batch_size=batch_size, padded_shapes=[padd_size]
    )
    intent_dataset = intent_dataset.repeat().padded_batch(
        batch_size=batch_size, padded_shapes=[padd_size]
    )
    slot_dataset = slot_dataset.repeat().padded_batch(
        batch_size=batch_size, padded_shapes=[padd_size]
    )
    pred_dataset = slot_dataset.zip((slot_dataset, intent_dataset))
    dataset = tf.data.Dataset.zip((word_dataset, pred_dataset))
    # word_dataset = word_dataset.repeat().batch(batch_size)
    # pred_dataset = pred_dataset.repeat().batch(batch_size)
    return dataset


class DataProcessor:
    def __init__(self, batch_size):
        self.intent_list, self.intent_dict = create_vocab("intent")
        self.word_list, self.word_dict = create_vocab("word")
        self.slot_list, self.slot_dict = create_vocab("slot")
        self.batch_size = batch_size
        self.train_word, self.train_seq, self.train_intent, self.train_slot = getDate(
            "./data/snips/train.txt",
            batch_size,
            self.word_dict,
            self.intent_dict,
            self.slot_dict,
        )
        self.test_word, self.test_seq, self.test_intent, self.test_slot = getDate(
            "./data/snips/test.txt",
            batch_size,
            self.word_dict,
            self.intent_dict,
            self.slot_dict,
        )
        self.dev_word, self.dev_seq, self.dev_intent, self.dev_slot = getDate(
            "./data/snips/dev.txt",
            batch_size,
            self.word_dict,
            self.intent_dict,
            self.slot_dict,
        )
        self.train_end = False
        self.test_end = False
        self.dev_end = False
        self.train_idx = 0
        self.test_idx = 0
        self.dev_idx = 0

    def get_train_batch(self):
        def get_data():
            word = self.train_word[self.train_idx : self.train_idx + self.batch_size]
            seq = self.train_seq[self.train_idx : self.train_idx + self.batch_size]
            intent = self.train_intent[
                self.train_idx : self.train_idx + self.batch_size
            ]
            slot = self.train_slot[self.train_idx : self.train_idx + self.batch_size]
            seq_lens = [sum(seq[i]) for i in range(len(seq))]
            self.train_idx += self.batch_size
            if self.train_idx >= len(self.train_word):
                self.train_end = True
                self.train_idx = 0
            return word, slot, seq, seq_lens, intent

        return get_data

    def get_test_batch(self):
        def get_data():
            word = self.test_word[self.test_idx : self.test_idx + self.batch_size]
            seq = self.test_seq[self.test_idx : self.test_idx + self.batch_size]
            intent = self.test_intent[self.test_idx : self.test_idx + self.batch_size]
            slot = self.test_slot[self.test_idx : self.test_idx + self.batch_size]
            seq_lens = [sum(seq[i]) for i in range(len(seq))]
            self.test_idx += self.batch_size
            if self.test_idx >= len(self.test_word):
                self.test_end = True
                self.test_idx = 0
            return word, slot, seq, seq_lens, intent

        return get_data

    def get_dev_batch(self):
        def get_data():
            word = self.dev_word[self.dev_idx : self.dev_idx + self.batch_size]
            seq = self.dev_seq[self.dev_idx : self.dev_idx + self.batch_size]
            intent = self.dev_intent[self.dev_idx : self.dev_idx + self.batch_size]
            slot = self.dev_slot[self.dev_idx : self.dev_idx + self.batch_size]
            seq_lens = [sum(seq[i]) for i in range(len(seq))]
            self.dev_idx += self.batch_size
            if self.dev_idx >= len(self.dev_word):
                self.dev_end = True
                self.dev_idx = 0
            return word, slot, seq, seq_lens, intent

        return get_data

if __name__ == "__main__":
    # datasets = get_dataset(8, 73, 16, 10)
    data_processor = DataProcessor(16)
    test = data_processor.get_train_batch()()
    print(test[0])
    print(test[1])
    print(test[2])
    print(test[3])
    print(test[4])
