# encoding=utf-8

import tensorflow as tf
from tensorflow.contrib import crf


class CRF(object):
    def __init__(self, embedded_chars, droupout_rate,seq_length,
    	num_labels , labels, lengths, is_training):
        
        self.droupout_rate = droupout_rate
        
        
        self.embedded_chars = embedded_chars
        
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths

        self.is_training = is_training

    def add_crf_layer(self):
        
        if self.is_training:
            # lstm input dropout rate set 0.5 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.droupout_rate)
        # project
        logits = self.project_layer(self.embedded_chars)
        # crf
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return (loss, logits, trans, pred_ids)


    def project_layer(self, embedded_chars, name=None):
        
        hidden_state = self.embedded_chars.get_shape()[-1]
        with tf.variable_scope("project" if not name else name):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[hidden_state, self.num_labels],
                                    dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.2))

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                embeddeding = tf.reshape(self.embedded_chars,[-1, hidden_state])
                pred = tf.nn.xw_plus_b(embeddeding, W, b)
                logtits_=tf.reshape(pred, [-1, self.seq_length, self.num_labels],name='output')
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])


    def crf_layer(self, logits):
        
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=tf.truncated_normal_initializer(stddev=0.2))
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.labels,
                transition_params=trans,
                sequence_lengths=self.lengths)
            return tf.reduce_mean(-log_likelihood), trans
