import argparse
import logging
import math

import tensorflow as tf
import numpy as np
from tensorflow import variable_scope
from tensorflow.python.ops import rnn_cell_impl, embedding_ops
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.framework import graph_util
import os
from dataset import DataProcessor
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops.rnn import _transpose_batch_time, _best_effort_input_batch_size
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.python.ops import array_ops

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


def create_model(args, text, sequence_length, keep_prob, num_word, num_slot, num_intent, force_intent=None,
                 force_slot=None):
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
        cell_fw = tf.contrib.rnn.BasicLSTMCell(args.encoder_hidden_dim // 2)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(args.encoder_hidden_dim // 2)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob,
                                                output_keep_prob=keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob,
                                                output_keep_prob=keep_prob)
        lstm_encoder, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, word_tensor,
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
            intent_decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(args.intent_decoder_hidden_dim)
            intent_decoder_cell = tf.contrib.rnn.DropoutWrapper(intent_decoder_cell, input_keep_prob=keep_prob,
                                                                output_keep_prob=keep_prob)
            intent_dense = tf.get_variable("intent_dense", [args.intent_decoder_hidden_dim, num_intent])
            if force_intent is not None:
                # Inference
                force_intent_embed = tf.nn.embedding_lookup(embedding_intent_decoder, force_intent[:, :-1])
                intent_input = tf.concat(
                    [encoder, tf.concat([tf.tile(intent_init_tensor, [batch_size, 1, 1]), force_intent_embed], 1)], 2)
                intent_input = tf.nn.dropout(intent_input, keep_prob=keep_prob)
                intent_decoder, _ = tf.nn.dynamic_rnn(intent_decoder_cell, intent_input,
                                                      sequence_length=sequence_length, dtype=tf.float32)
                # intent_decoder = tf.nn.dropout(intent_decoder, keep_prob=keep_prob)
            else:
                outputs_intent, final_state, _ = tf.nn.raw_rnn(
                    intent_decoder_cell,
                    loop_fn_build(encoder,
                                  intent_init_tensor,
                                  embedding_intent_decoder,
                                  intent_decoder_cell,
                                  batch_size,
                                  args.intent_embedding_dim + args.encoder_hidden_dim + args.attention_output_dim,
                                  sequence_length,
                                  intent_dense,
                                  sentence_size
                                  )
                )
                intent_decoder = tf.transpose(outputs_intent.stack(), [1, 0, 2])
            intent_pred = tf.einsum('ijk,kl->ijl', intent_decoder, intent_dense)
        slot_combine = tf.concat([encoder, intent_pred], 2,  name='slotInput')
        # slot_combine = tf.nn.dropout(slot_combine, keep_prob=keep_prob, name='slotInput')
        with tf.variable_scope('slot_decoder'):
            embedding_slot_decoder = tf.get_variable(
                "slot_embedding", [num_slot, args.slot_embedding_dim])
            slot_init_tensor = tf.get_variable(
                "slot_init", [1, 1, args.slot_embedding_dim], initializer=tf.random_normal_initializer()
            )
            slot_decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(args.slot_decoder_hidden_dim)
            slot_decoder_cell = tf.contrib.rnn.DropoutWrapper(slot_decoder_cell, input_keep_prob=keep_prob,
                                                              output_keep_prob=keep_prob)
            slot_dense = tf.get_variable("slot_dense", [args.slot_decoder_hidden_dim, num_slot])
            if force_slot is not None:
                # Inference
                force_slot_embed = tf.nn.embedding_lookup(embedding_slot_decoder, force_slot[:, :-1])
                slot_input = tf.concat(
                    [slot_combine, tf.concat([tf.tile(slot_init_tensor, [batch_size, 1, 1]), force_slot_embed], 1)], 2)
                slot_input = tf.nn.dropout(slot_input, keep_prob=keep_prob)
                slot_decoder, _ = tf.nn.dynamic_rnn(slot_decoder_cell, slot_input,
                                                    sequence_length=sequence_length, dtype=tf.float32)
                # slot_decoder = tf.nn.dropout(slot_decoder, keep_prob=keep_prob)
            else:
                outputs_slot, final_state, _ = tf.nn.raw_rnn(
                    slot_decoder_cell,
                    loop_fn_build(slot_combine,
                                  slot_init_tensor,
                                  embedding_slot_decoder,
                                  slot_decoder_cell,
                                  batch_size,
                                  args.slot_embedding_dim + args.encoder_hidden_dim + args.attention_output_dim + num_intent,
                                  sequence_length,
                                  slot_dense,
                                  sentence_size
                                  )
                )
                slot_decoder = tf.transpose(outputs_slot.stack(), [1, 0, 2])
            slot_pred = tf.einsum('ijk,kl->ijl', slot_decoder, slot_dense)
    # slot_out = tf.nn.softmax(slot_pred, 2)
    # intent_out = tf.nn.softmax(intent_pred, 2)
    return [slot_pred, intent_pred]


print(tf.__version__)
init_args()
args = parser.parse_args()
input_data = tf.placeholder(tf.int32, [None, None], name='inputs')
sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
global_step = tf.Variable(0, trainable=False, name='global_step')
slots = tf.placeholder(tf.int32, [None, None], name='slots')
word_weights = tf.placeholder(tf.float32, [None, None], name='slot_weights')
intent = tf.placeholder(tf.int32, [None, None], name='intent')
with tf.variable_scope('model'):
    training_outputs = create_model(args, input_data, sequence_length, 1 - args.dropout_rate, 12150, 73, 8,
                                    force_intent=intent, force_slot=slots
                                    )

slots_shape = tf.shape(slots)
slot_outputs = training_outputs[0]
with tf.variable_scope('slot_loss'):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slots, logits=slot_outputs)
    crossent = tf.reshape(crossent, slots_shape)
    slot_loss = tf.reduce_sum(crossent * word_weights, 1)
    # # total_size = tf.reduce_sum(word_weights, 1)
    # # total_size += 1e-12
    # slot_loss = slot_loss

intent_output = training_outputs[1]
with tf.variable_scope('intent_loss'):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent, logits=intent_output)
    crossent = tf.reshape(crossent, slots_shape)
    intent_loss = tf.reduce_sum(crossent * word_weights, 1)
    # total_size = tf.reduce_sum(word_weights, 1)
    # total_size += 1e-12
    # intent_loss = intent_loss / total_size
loss = intent_loss + slot_loss
params = tf.trainable_variables()
regularization_cost = args.l2_penalty * tf.reduce_sum([tf.nn.l2_loss(v) for v in params])
loss += regularization_cost

opt = tf.train.AdamOptimizer()
# intent_params = []
# slot_params = []
# for p in params:
#     if not 'slot_' in p.name:
#         intent_params.append(p)
#     if not 'intent_' in p.name:
#         slot_params.append(p)
gradients = tf.gradients(loss, params)
# gradients_intent = tf.gradients(intent_loss, intent_params)
clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
# clipped_gradients_intent, norm_intent = tf.clip_by_global_norm(gradients_intent, 5.0)

gradient_norm = norm
# gradient_norm_intent = norm_intent
update_slot = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
# update_intent = opt.apply_gradients(zip(clipped_gradients_intent, intent_params), global_step=global_step)

training_outputs = [global_step, slot_loss, intent_loss,
                    update_slot, gradient_norm]  # update_intent, update_slot, gradient_norm_intent,
# gradient_norm_slot]

with tf.variable_scope('model', reuse=True):
    inference_outputs = create_model(args, input_data, sequence_length, 1, 12150, 73, 8)

inference_slot_output = tf.nn.softmax(inference_outputs[0], name='slot_output')
inference_intent_output = tf.nn.softmax(inference_outputs[1], name='intent_output')

inference_outputs = [inference_intent_output, inference_slot_output]
inference_inputs = [input_data, sequence_length]

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    saver = tf.train.Saver()
    data_processor = DataProcessor(args.batch_size)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, tf.train.latest_checkpoint(args.save_dir))
        logging.info('Training Start')
        epochs = 0
        loss = 0.0
        line = 0
        num_loss = 0
        step = 0
        no_improve = 0

        # variables to store highest values among epochs, only use 'valid_err' for now
        valid_slot = 0
        test_slot = 0
        valid_intent = 0
        test_intent = 0
        valid_err = 0
        test_err = 0
        slots_loss = 0.0
        intents_loss = 0.0


        def Valid(dataset, type):
            pred_intents = []
            correct_intents = []
            slot_outputs = []
            correct_slots = []
            input_words = []

            while True:
                in_data, slot_data, slot_weight, length, intents = dataset()
                feed_dict = {input_data.name: in_data, sequence_length.name: length}
                ret = sess.run(inference_outputs, feed_dict)
                for i in ret[0]:
                    pred_intents.append(np.argmax(i[0]))
                for i in intents:
                    correct_intents.append(i[0])
                pred_slots = ret[1].reshape((len(slot_data), len(slot_data[0]), -1))
                for p, t, i, l in zip(pred_slots, slot_data, in_data, length):
                    p = np.argmax(p, 1)
                    tmp_pred = []
                    tmp_correct = []
                    tmp_input = []
                    for j in range(l):
                        tmp_pred.append(data_processor.slot_list[p[j]])
                        tmp_correct.append(data_processor.slot_list[t[j]])
                        tmp_input.append(data_processor.word_list[i[j]])

                    slot_outputs.append(tmp_pred)
                    correct_slots.append(tmp_correct)
                    input_words.append(tmp_input)

                if type == 1:
                    if data_processor.test_end:
                        data_processor.test_end = False
                        break
                elif type == 2:
                    if data_processor.dev_end:
                        data_processor.dev_end = False
                        break
                elif type == 3:
                    if data_processor.train_end:
                        data_processor.train_end = False
                        break

            pred_intents = np.array(pred_intents)
            correct_intents = np.array(correct_intents)
            accuracy = (pred_intents == correct_intents)
            semantic_error = accuracy
            accuracy = accuracy.astype(float)
            accuracy = np.mean(accuracy) * 100.0

            index = 0
            for t, p in zip(correct_slots, slot_outputs):
                # Process Semantic Error
                if len(t) != len(p):
                    raise ValueError('Error!!')

                for j in range(len(t)):
                    if p[j] != t[j]:
                        semantic_error[index] = False
                        break
                index += 1
            semantic_error = semantic_error.astype(float)
            semantic_error = np.mean(semantic_error) * 100.0

            # f1, precision, recall = computeF1Score(correct_slots, slot_outputs)
            # logging.info('slot f1: ' + str(f1))
            logging.info('intent accuracy: ' + str(accuracy))
            logging.info('semantic error(intent, slots are all correct): ' + str(semantic_error))


        while True:
            # break
            in_data, slot_data, slot_weight, length, intents = data_processor.get_train_batch()()
            feed_dict = {input_data.name: in_data, slots.name: slot_data, word_weights.name: slot_weight,
                         sequence_length.name: length, intent.name: intents}
            ret = sess.run(training_outputs, feed_dict)
            slots_loss += np.mean(ret[1])
            intents_loss += np.mean(ret[2])

            line += args.batch_size
            step = ret[0]
            num_loss += 1
            # print(str(step))
            if data_processor.train_end:
                data_processor.train_end = False
                line = 0
                epochs += 1
                logging.info('Step: ' + str(step))
                logging.info('Epochs: ' + str(epochs))
                logging.info('Slot Loss: ' + str(slots_loss))
                logging.info('Intent Loss: ' + str(intents_loss))
                num_loss = 0
                slots_loss = 0.0
                intents_loss = 0.0
                save_path = os.path.join(args.save_dir, '_step_' + str(step) + '.ckpt')
                saver.save(sess, save_path)
                logging.info('Test:')
                Valid(data_processor.get_test_batch(), 1)
                logging.info('Valid:')
                Valid(data_processor.get_dev_batch(), 2)
                if epochs >= args.num_epoch:
                    break
        logging.info('Test:')
        Valid(data_processor.get_test_batch(), 1)
        logging.info('Valid:')
        Valid(data_processor.get_dev_batch(), 2)
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            [
                "slot_output",
                "intent_output"
            ],
        )

        # # 写入序列化的 PB 文件
        LOGDIR = '../logdir'
        train_writer = tf.summary.FileWriter(LOGDIR)
        train_writer.add_graph(sess.graph)
        train_writer.flush()
        train_writer.close()
        with tf.gfile.FastGFile("pb/model.pb", mode="wb") as f:
            f.write(constant_graph.SerializeToString())
