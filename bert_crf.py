
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from tensorflow.python.ops import math_ops
import pickle
from crf_layer import CRF
import numpy as np
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", 'data',
    "The input datadir.",)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("do_train", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False,"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 2, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")



flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        #self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file,'r',encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split('\t')[0]
                label = line.strip().split('\t')[-1]
                if len(contends) == 0 :
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            #print(lines)
            # exit()
            return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "test")


    def get_labels(self):
        return ["O", "B-PER", "I-PER", "E-PER","B-ORG", "I-ORG","E-ORG", "B-LOC", "I-LOC", "E-LOC","B-PRO", "I-PRO", "E-PRO","S-LOC", 
        "S-PER","S-PRO", "S-ORG","X","[CLS]","[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode):
    label_map = {}
    for (i, label) in enumerate(label_list,1):
        label_map[label] = i
    with open('./output_c/label2id.pkl','wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    #print(textlist)
    tokens = []
    labels = []
    # print(textlist)
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        # print(token)
        tokens.extend(token)
        label_1 = labellist[i]
        # print(label_1)
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
        # print(tokens, labels)
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        #label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    #assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        #label_mask = label_mask
    )
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer,mode=None):
    
    #print(len(len(examples)))
    feature_dict=[]
    for (ex_index, example) in enumerate(examples):
        #print('ex_index',ex_index)
        
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode)
        feature_dict.append(feature)
        
        #features["label_mask"] = create_int_feature(feature.label_mask)
    return feature_dict

#===================
#转化为GPU调用
class model_fn(object):
    def __init__(self,bert_config,
        init_checkpoint,
        num_labels,
        learning_rate,
        seq_length,
        num_train_steps,
        num_warmup_steps,
            
        use_one_hot_embeddings):
        self.input_ids=tf.placeholder(tf.int32,shape=[None,seq_length],name='input_ids')
        self.input_mask=tf.placeholder(tf.int32,shape=[None,seq_length],name='input_mask')
        self.segment_ids=tf.placeholder(tf.int32,shape=[None,seq_length],name='segment_ids')
        self.label_ids=tf.placeholder(tf.int32,shape=[None,seq_length],name='label_ids')
        self.is_training=tf.placeholder(tf.bool,shape=[],name='is_train')
        self.global_step = tf.Variable(0, trainable=False)
       

        #===============================
        model=modeling.BertModel(config=bert_config,
            is_training=False,input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)
        #============================
        #
        self.tvars=tf.trainable_variables()
        (self.assignment_map,_)=modeling.get_assignment_map_from_checkpoint(self.tvars,init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint,self.assignment_map)
        embedding=model.get_sequence_output()
        hidden_size=embedding.shape[-1].value
        print(hidden_size)
        used=tf.sign(tf.abs(self.input_ids))
        length=tf.reduce_sum(used,reduction_indices=1)
        
        crf=CRF(embedded_chars=embedding,
            droupout_rate=0.9,
            seq_length=FLAGS.max_seq_length,

            num_labels=num_labels,
            labels=self.label_ids,
            lengths=length,
            is_training=True)
        self.total_loss,self.logits,self.trans,self.predictions=crf.add_crf_layer()
        print(',self.total_loss',self.total_loss)




        with tf.variable_scope('loss'):

            
            # ===========================
            # 设置不同的学习率
            all_variables = tf.trainable_variables()
            bert_variable = [x for x in all_variables if 'bert' in x.name]
            other_variable = [x for x in all_variables if 'bert' not in x.name]
            other_optimizer = tf.train.AdamOptimizer(0.001)
            other_op = other_optimizer.minimize(self.total_loss, var_list=other_variable)

            train_op = optimization.create_optimizer(self.total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            self.train_op=tf.group(other_op,train_op)
           






def main(_):
    #tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()
    label_dict={}
    for i in range(len(label_list)):
        label_dict[i+1]=label_list[i]

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    

    #======================
    word_dict={}
    for word in tokenizer.vocab.keys():
        word_dict[int(tokenizer.vocab[word])]=word



    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        print('############################')
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        print('^^^^^^^^^^^^^^^^^train_examples')
        print(len(train_examples))


        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


    if FLAGS.do_train:
        train_feature=filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer)
       
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        #===============================
        #===========================
        num_example=len(train_feature)
        print('num_example',num_example)
        all_input_ids=[]
        all_input_mask=[]
        all_segment_ids=[]
        all_label_ids=[]


        for feature in train_feature:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_ids)

        #=====================
        #
        all_input_ids=np.array(all_input_ids)
        all_input_mask=np.array(all_input_mask)
        all_segment_ids=np.array(all_segment_ids)
        all_label_ids=np.array(all_label_ids)

        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as sess:
            model=model_fn(bert_config=bert_config,init_checkpoint=FLAGS.init_checkpoint,
                num_labels=len(label_list)+1,learning_rate=FLAGS.learning_rate,
                seq_length=FLAGS.max_seq_length,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                use_one_hot_embeddings=False)
            batch_size=FLAGS.train_batch_size

            sess.run(tf.global_variables_initializer())
            saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=5)
            sess.run(tf.local_variables_initializer())
            ckpt=tf.train.get_checkpoint_state('model')

          

            np.savetxt('new.csv',model.trans.eval(),delimiter=',')

            #===============================
            #
            # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            #     print('mode_path %s' %ckpt.model_checkpoint_path)
            #     saver.restore(sess,ckpt.model_checkpoint_path)
            for i in range(int(FLAGS.num_train_epochs)):
                print('$$$$$$$$$$$$$$$$$$')
                print('i',i)
                num=np.arange(num_example)
                np.random.shuffle(num)
                temp_all_input_ids=all_input_ids[num]
                temp_all_input_mask=all_input_mask[num]
                temp_all_sgment_ids=all_segment_ids[num]
                temp_all_label_ids=all_label_ids[num]

                for start,end in zip(range(0,num_example,batch_size),range(batch_size,num_example,batch_size)):
                    print('epochs')
                    # print(temp_all_input_ids[start:end])
                    # print(np.shape(temp_all_input_ids[start:end]))
                    # print(np.shape(temp_all_input_mask[start:end]))
                    # print(np.shape(temp_all_sgment_ids[start:end]))
                    # print(np.shape(temp_all_label_ids[start:end]))
                    
                    feed={model.input_ids:np.array(temp_all_input_ids[start:end]),
                          model.input_mask:np.array(temp_all_input_mask[start:end]),
                          model.segment_ids:np.array(temp_all_sgment_ids[start:end]),
                          model.label_ids:np.array(temp_all_label_ids[start:end]),
                          model.is_training:True}
                    print('******************')
                    #============================
                    #传入优化器,计算loss
                    loss,_=sess.run([model.total_loss,model.train_op],feed)
                    print(loss)

                    
            checkpoint_path=os.path.join('model','model.ckpt-382')
            saver.save(sess,checkpoint_path)


            #========================
            #pb file
           
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["project/logits/output"])
            with tf.gfile.FastGFile('bert_ner.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())
    
            # #==============================
            # #验证集准确率
            if FLAGS.do_eval:
                eval_examples=processor.get_dev_examples(FLAGS.data_dir)
                eval_file=os.path.join(FLAGS.output_dir,'eval.tf_record')

                eval_feature=filed_based_convert_examples_to_features(eval_examples,label_list,FLAGS.max_seq_length,tokenizer,eval_file)
                tf.logging('***********************evaluation')

                test_all_input_ids=[]
                test_all_input_mask=[]
                test_all_sgment_ids=[]
                test_all_label_ids=[]
                test_num_examples=len(eval_feature)

                for feature in eval_feature:
                    test_all_label_ids.append(feature.input_ids)
                    test_all_input_mask.append(feature.input_mask)
                    test_all_sgment_ids.append(feature.segment_ids)
                    test_all_label_ids.append(feature.label_ids)
                f_w=open('result.txt','w',encoding='utf-8')
                for start,end in zip(range(0,test_num_examples,batch_size),range(batch_size,test_num_examples,batch_size)):
                    print('epochs')
                    feed={model.input_ids:test_all_input_ids[start:end],
                          model.input_mask:test_all_input_mask[start:end],
                          model.segment_ids:test_all_sgment_ids[start:end],
                          model.label_ids:test_all_label_ids[start:end],model.is_training:False
                          }

                    loss,pre=sess.run([model.loss,model.predictions],feed)
                    #========================
                    #
                    input_acc=test_all_input_ids[start:end]
                    label_acc=test_all_label_ids[start:end]
                    for i in range(8):
                        pre_line=[label_dict[id] for id in pre[i] if id!=0]
                        y_label=[label_dict[id] for id in label_acc[i] if id!=0]
                        test_line=[word_dict[id] for id in input_acc[i] if id!=0]

                        for j in range(len(pre_line)):
                            if pre_line[j]=='[CLS]':
                                continue
                            elif pre_line[j]=='[SEP]':
                                break
                            else:
                                f_w.write(test_line[j]+'\t')
                                f_w.write(y_label[j]+'\t')
                                f_w.write(pre_line[j]+'\n')
                        f_w.write('\n')




        
    

        
if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
