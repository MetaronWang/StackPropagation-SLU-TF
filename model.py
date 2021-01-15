import argparse
import logging
import math
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import graph_util
from tensorflow.contrib.rnn import DropoutWrapper, BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn, raw_rnn
from tensorflow.python.ops import array_ops
from random import random
from dataset import DataProcessor

parser = argparse.ArgumentParser()


def get_data_size(inputs):
    return array_ops.shape(inputs)[0], array_ops.shape(inputs)[1]


# Training parameters.
def init_args():
    parser.add_argument('--data_dir', '-dd', type=str, default='data/snips')
    parser.add_argument('--save_dir', '-sd', type=str, default='save')
    parser.add_argument("--random_state", '-rs', type=int, default=0)
    parser.add_argument('--num_epoch', '-ne', type=int, default=300)
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
    parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
    parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
    parser.add_argument('--intent_forcing_rate', '-ifr', type=float, default=0.9)
    parser.add_argument("--differentiable", "-d", action="store_true", default=False)
    parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.9)

    # model parameters.
    parser.add_argument('--num_words', '-nw', type=int, default=12150)
    parser.add_argument('--num_intents', '-ni', type=int, default=8)
    parser.add_argument('--num_slots', '-ns', type=int, default=73)
    parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
    parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
    parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=8)
    parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
    parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=64)
    parser.add_argument('--intent_decoder_hidden_dim', '-idhd', type=int, default=64)
    parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)
    parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)


def loop_fn_build(inputs, init_tensor, embedding_matrix, cell, batch_size, embedding_size, sequence_length, dense,
                  sentence_size):
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=sentence_size)
    # [B,T,S]-->[T,B,S]
    inputs_trans = tf.transpose(inputs, perm=[1, 0, 2])
    inputs_ta = inputs_ta.unstack(inputs_trans)

    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        if cell_output is None:  # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
            prev_intent = tf.squeeze(tf.tile(init_tensor, [1, batch_size, 1]), [0])
        else:
            next_cell_state = cell_state
            dense_layer = tf.einsum('jk,kl->jl', cell_output, dense)
            _, index = tf.math.top_k(dense_layer, k=1)
            prev_intent = tf.nn.embedding_lookup(embedding_matrix, tf.reshape(index, [-1]))
        elements_finished = (time >= sequence_length)
        finished = tf.reduce_all(elements_finished)
        next_input = tf.cond(finished,
                             lambda: tf.zeros([batch_size, embedding_size], dtype=tf.float32),
                             lambda: tf.concat([inputs_ta.read(time), prev_intent], 1))
        # next_input =  tf.concat([inputs_ta.read(time), prev_intent], 1)

        next_loop_state = None
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    return loop_fn


def build_decoder(input_tensor, force_tensor, cond_tensor, embedding_tensor, init_tensor, batch_size, keep_prob,
                  decoder_cell, args, dense, sentence_size, sequence_length, embedding_dim):
    def force():
        force_embed = tf.nn.embedding_lookup(embedding_tensor, force_tensor[:, :-1])
        decoder_input = tf.concat(
            [input_tensor, tf.concat([tf.tile(init_tensor, [batch_size, 1, 1]), force_embed], 1)], 2)
        decoder_input = tf.nn.dropout(decoder_input, keep_prob=keep_prob)
        force_decoder, _ = dynamic_rnn(decoder_cell, decoder_input,
                                       sequence_length=sequence_length, dtype=tf.float32)
        return force_decoder

    def non_force():
        outputs, final_state, _ = raw_rnn(
            decoder_cell,
            loop_fn_build(input_tensor,
                          init_tensor,
                          embedding_tensor,
                          decoder_cell,
                          batch_size,
                          embedding_dim + args.encoder_hidden_dim + args.attention_output_dim,
                          sequence_length,
                          dense,
                          sentence_size
                          )
        )
        non_force_decoder = tf.transpose(outputs.stack(), [1, 0, 2])
        return non_force_decoder

    if force_tensor is not None:
        # decoder = force()
        decoder = tf.cond(cond_tensor, lambda: force(), lambda: non_force(), name="Condition_Input_Feeding")
    else:
        decoder = non_force()
    return decoder


def create_model(args, text, sequence_length, keep_prob, num_word, num_slot, num_intent, force_intent=None,
                 force_slot=None, intent_feeding=None, slot_feeding=None):
    batch_size, sentence_size = get_data_size(text)
    embedding_encoder = tf.get_variable(
        "embedding", [num_word, args.word_embedding_dim])
    # Look up embedding:
    #   encoder_inputs: [max_time, batch_size]
    #   encoder_emb_inp: [max_time, batch_size, embedding_size]
    word_tensor = tf.nn.embedding_lookup(
        embedding_encoder, text)
    word_tensor = tf.nn.dropout(word_tensor, keep_prob=keep_prob)
    with tf.variable_scope('lstm-encoder'):
        cell_fw = BasicLSTMCell(args.encoder_hidden_dim // 2)
        cell_bw = BasicLSTMCell(args.encoder_hidden_dim // 2)
        cell_fw = DropoutWrapper(cell_fw, input_keep_prob=keep_prob,
                                 output_keep_prob=keep_prob)
        cell_bw = DropoutWrapper(cell_bw, input_keep_prob=keep_prob,
                                 output_keep_prob=keep_prob)
        lstm_encoder, _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, word_tensor,
                                                    sequence_length=sequence_length, dtype=tf.float32)
        lstm_encoder = tf.concat([lstm_encoder[0], lstm_encoder[1]], 2)
    with tf.variable_scope('self-attention'):
        q_m = tf.get_variable(
            "query_dense", [args.word_embedding_dim, args.attention_hidden_dim])
        k_m = tf.get_variable(
            "key_dense", [args.word_embedding_dim, args.attention_hidden_dim])
        v_m = tf.get_variable(
            "value_dense", [args.word_embedding_dim, args.attention_output_dim])
        q = tf.einsum('ijk,kl->ijl', word_tensor, q_m)
        k = tf.einsum('ijk,kl->ijl', word_tensor, k_m)
        v = tf.einsum('ijk,kl->ijl', word_tensor, v_m)
        score = tf.nn.softmax(tf.matmul(q, k, transpose_b=True)) / math.sqrt(args.attention_hidden_dim)
        attention_tensor = tf.matmul(score, v)
        attention_tensor = tf.nn.dropout(attention_tensor, keep_prob=keep_prob)
    encoder = tf.concat([lstm_encoder, attention_tensor], 2)
    # encoder = tf.nn.dropout(encoder, keep_prob=keep_prob)

    with tf.variable_scope('decoder'):
        with tf.variable_scope('intent_decoder'):
            embedding_intent_decoder = tf.get_variable(
                "intent_embedding", [num_intent, args.intent_embedding_dim])
            intent_init_tensor = tf.get_variable(
                "intent_init", [1, 1, args.intent_embedding_dim], initializer=tf.random_normal_initializer()
            )
            intent_decoder_cell = BasicLSTMCell(args.intent_decoder_hidden_dim)
            intent_decoder_cell = DropoutWrapper(intent_decoder_cell, input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob)
            intent_dense = tf.get_variable("intent_dense", [args.intent_decoder_hidden_dim, num_intent])
            intent_decoder = build_decoder(encoder, force_intent, intent_feeding, embedding_intent_decoder,
                                           intent_init_tensor, batch_size, keep_prob, intent_decoder_cell,
                                           args, intent_dense, sentence_size, sequence_length,
                                           args.intent_embedding_dim)

            intent_pred = tf.einsum('ijk,kl->ijl', intent_decoder, intent_dense)
        slot_combine = tf.concat([encoder, intent_pred], 2, name='slotInput')
        # slot_combine = tf.nn.dropout(slot_combine, keep_prob=keep_prob, name='slotInput')
        with tf.variable_scope('slot_decoder'):
            embedding_slot_decoder = tf.get_variable(
                "slot_embedding", [num_slot, args.slot_embedding_dim])
            slot_init_tensor = tf.get_variable(
                "slot_init", [1, 1, args.slot_embedding_dim], initializer=tf.random_normal_initializer()
            )
            slot_decoder_cell = BasicLSTMCell(args.slot_decoder_hidden_dim)
            slot_decoder_cell = DropoutWrapper(slot_decoder_cell, input_keep_prob=keep_prob,
                                               output_keep_prob=keep_prob)
            slot_dense = tf.get_variable("slot_dense", [args.slot_decoder_hidden_dim, num_slot])
            slot_decoder = build_decoder(slot_combine, force_slot, slot_feeding, embedding_slot_decoder,
                                         slot_init_tensor, batch_size, keep_prob, slot_decoder_cell,
                                         args, slot_dense, sentence_size, sequence_length,
                                         args.slot_embedding_dim + num_intent)
            slot_pred = tf.einsum('ijk,kl->ijl', slot_decoder, slot_dense)
    return [slot_pred, intent_pred]


def io_operator(args):
    input_data = tf.placeholder(tf.int32, [None, None], name='inputs')
    sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    slots = tf.placeholder(tf.int32, [None, None], name='slots')
    word_weights = tf.placeholder(tf.float32, [None, None], name='slot_weights')
    intent = tf.placeholder(tf.int32, [None, None], name='intent')
    intent_feeding = tf.placeholder(tf.bool, (), name='intent_feeding')
    slot_feeding = tf.placeholder(tf.bool, (), name='slot_feeding')
    with tf.variable_scope('model'):
        training_outputs = create_model(args, input_data, sequence_length, 1 - args.dropout_rate, args.num_words,
                                        args.num_slots, args.num_intents, force_intent=intent, force_slot=slots,
                                        intent_feeding=intent_feeding, slot_feeding=slot_feeding)

    slots_shape = tf.shape(slots)
    slot_outputs = training_outputs[0]
    with tf.variable_scope('slot_loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slots, logits=slot_outputs)
        cross_entropy = tf.reshape(cross_entropy, slots_shape)
        slot_loss = tf.reduce_sum(cross_entropy * word_weights, 1)

    intent_output = training_outputs[1]
    with tf.variable_scope('intent_loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent, logits=intent_output)
        cross_entropy = tf.reshape(cross_entropy, slots_shape)
        intent_loss = tf.reduce_sum(cross_entropy * word_weights, 1)

    loss = intent_loss + slot_loss
    params = tf.trainable_variables()
    regularization_cost = args.l2_penalty * tf.reduce_sum([tf.nn.l2_loss(v) for v in params])
    loss += regularization_cost

    opt = tf.train.AdamOptimizer()

    gradients = tf.gradients(loss, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
    gradient_norm = norm
    update_slot = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
    training_outputs = [global_step, slot_loss, intent_loss,
                        update_slot, gradient_norm]

    with tf.variable_scope('model', reuse=True):
        inference_outputs = create_model(args, input_data, sequence_length, 1, 12150, 73, 8)

    inference_slot_output = tf.nn.softmax(inference_outputs[0], name='slot_output')
    inference_intent_output = tf.nn.softmax(inference_outputs[1], name='intent_output')

    inference_outputs = [inference_intent_output, inference_slot_output]
    inference_inputs = [input_data, sequence_length]
    placeholder = (input_data, sequence_length, global_step, slots, word_weights, intent, intent_feeding, slot_feeding)
    return placeholder, inference_outputs, training_outputs


def main():
    init_args()
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # saver = tf.train.Saver()
    data_processor = DataProcessor()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        placeholder, inference_outputs, training_outputs = io_operator(args)
        input_data, sequence_length, global_step, slots, word_weights, intent, intent_feeding, slot_feeding = placeholder
        sess.run(tf.global_variables_initializer())
        logging.info('Training Start')

        def valid(name="test"):
            intent_right_num, semantic_right_num, all_num = 0, 0, 0
            for test_batch in data_processor.get_data(name, batch_size=10000):
                test_feed_dict = {input_data.name: test_batch[0], sequence_length.name: test_batch[4]}
                test_ret = sess.run(inference_outputs, test_feed_dict)
                pred_intent = [np.argmax(i[0]) for i in test_ret[0]]
                correct_intent = [i[0] for i in test_batch[3]]
                pred_slot = [[np.argmax(i) for i in s] for s in test_ret[1]]
                pred_slot = [pred_slot[i][:test_batch[4][i]] for i in range(len(pred_slot))]
                correct_slot = [test_batch[2][i][:test_batch[4][i]] for i in range(len(test_batch[2]))]
                for i in range(len(test_batch[0])):
                    all_num += 1
                    if pred_intent[i] == correct_intent[i]:
                        intent_right_num += 1
                        if pred_slot[i] == correct_slot[i]:
                            semantic_right_num += 1

            logging.info('intent accuracy: ' + str(intent_right_num / all_num * 100))
            logging.info('semantic accuracy(intent, slots are all correct): ' + str(semantic_right_num / all_num * 100))

        for num in range(args.num_epoch):
            slots_loss, intents_loss = 0, 0
            train_data = data_processor.get_data("train", batch_size=args.batch_size)
            for batch in train_data:
                feed_dict = {input_data.name: batch[0], slots.name: batch[2], word_weights.name: batch[1],
                             sequence_length.name: batch[4], intent.name: batch[3],
                             intent_feeding.name: random() > args.intent_forcing_rate,
                             slot_feeding.name: random() > args.slot_forcing_rate}
                ret = sess.run(training_outputs, feed_dict)
                slots_loss += np.mean(ret[1])
                intents_loss += np.mean(ret[2])
            slots_loss, intents_loss = slots_loss / len(train_data), intents_loss / len(train_data)

            logging.info('Epoch: {}'.format(num))
            logging.info('Test:')
            valid('test')
            logging.info('Valid:')
            valid('dev')
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            [
                "slot_output",
                "intent_output"
            ],
        )

        # # 写入序列化的 PB 文件
        LOGDIR = './logdir'
        train_writer = tf.summary.FileWriter(LOGDIR)
        train_writer.add_graph(sess.graph)
        train_writer.flush()
        train_writer.close()
        with tf.gfile.FastGFile("model/model.pb", mode="wb") as f:
            f.write(constant_graph.SerializeToString())


if __name__ == '__main__':
    main()
