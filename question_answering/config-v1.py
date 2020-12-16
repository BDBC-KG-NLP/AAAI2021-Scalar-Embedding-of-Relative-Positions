import os
import tensorflow as tf

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''
import sys
from prepro import prepro
from main import train, test, demo


home = os.getcwd()
target_dir = "data-v1"

if not os.path.exists(target_dir):
  os.makedirs(target_dir)
        
train_file = os.path.join(home, "datasets", "squad", "train-v1.1.json")
dev_file = os.path.join(home, "datasets", "squad", "dev-v1.1.json")
test_file = os.path.join(home, "datasets", "squad", "dev-v1.1.json")
glove_word_file = os.path.join(home, "datasets", "glove", "glove.840B.300d.txt")

dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
word_dictionary = os.path.join(target_dir, "word_dictionary.json")
char_dictionary = os.path.join(target_dir, "char_dictionary.json")
train_record_file = os.path.join(target_dir, "train.tfrecords")
  

flags = tf.flags
flags.DEFINE_string("model", "Raw", "[Raw, T5, TPE]")
flags.DEFINE_string("mode", "train", "Running mode train/debug/test")

flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")
flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("word_dictionary", word_dictionary, "Word dictionary")
flags.DEFINE_string("char_dictionary", char_dictionary, "Character dictionary")


flags.DEFINE_string("event_log_dir", "event_log_dir", "Directory for tf event")
flags.DEFINE_string("save_dir", "save_dir", "Directory for saving model")
flags.DEFINE_string("answer_file", "answer_file", "Out file for answer")


flags.DEFINE_integer("glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 30, "Limit length for answers")
flags.DEFINE_integer("test_para_limit", 400, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 50, "Limit length for question in test file")
#to adjust the soft-t5
flags.DEFINE_integer("fixed_c_maxlen", 400, "Limit length for question in test file")
flags.DEFINE_integer("fixed_q_maxlen", 50, "Limit length for question in test file")

flags.DEFINE_integer("char_limit", 16, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("num_steps", 200000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.1, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
flags.DEFINE_integer("hidden", 128, "Hidden size")
flags.DEFINE_integer("num_heads",1, "Number of heads in self attention")
#we try to large the training time...
flags.DEFINE_integer("early_stop", 10, "Checkpoints for early stop")

# Extensions (Uncomment corresponding code in download.sh to download the required data)
glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding source file")
flags.DEFINE_boolean("pretrained_char", False, "Whether to use pretrained character embedding")

fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")
flags.DEFINE_boolean("fasttext", False, "Whether to use fasttext")

flags.DEFINE_float("bucket_slop_min", 1.0, "buceket_slop_t1")
flags.DEFINE_float("bucket_slop_max", 10.0, "buceket_slop_t1")

flags.DEFINE_float("t5_num_buckets", 32, "buceket_slop_t1")
flags.DEFINE_float("t5_max_distance", 128, "buceket_slop_t1")

flags.DEFINE_integer('l1_width', 100, 'mlp l1 width')
flags.DEFINE_integer('l2_width', 40, 'mlp l2 width')
flags.DEFINE_float('stddev', 0.01, 'initialization stddev')

flags.DEFINE_string("soft_t5_activation", "relu", "[relu, sigmoid]")
flags.DEFINE_float("trail", 0, "we run five trails for each model")


def main(_):
  config = flags.FLAGS
  #srun -p sugon --gres=gpu:1 python config-v1.py prepro Raw 2 128 50
  
  
  mode=sys.argv[1]
  model_name = sys.argv[2]
  num_heads=sys.argv[3]
  
  train_dir = 'train-v1'
  
  if model_name in ['Soft_T5']:
    
    fixed_c_maxlen=sys.argv[4]
    learning_rate=sys.argv[5]
    
    bucket_slop_min=float(sys.argv[6])
    bucket_slop_max=float(sys.argv[7])
    
    l1_width = int(sys.argv[8])
    l2_width = int(sys.argv[9])
    stddev=float(sys.argv[10])
    
    soft_t5_activation=sys.argv[11]
    trail = sys.argv[12]
    
    dir_name = os.path.join(train_dir, "_".join([model_name, 
                                                 str(num_heads),
                                                 str(fixed_c_maxlen),
                                                 str(learning_rate),
                                                 str(bucket_slop_min),
                                                 str(bucket_slop_max),
                                                 str(l1_width),
                                                 str(l2_width),
                                                 str(stddev),
                                                 soft_t5_activation,
                                                 trail
                                                 ]
                                                ))
  elif model_name in ['Soft_T5_Nob']:
    fixed_c_maxlen=sys.argv[4]
    learning_rate=sys.argv[5]
    
    soft_t5_activation=sys.argv[6]
    trail = sys.argv[7]
    
    dir_name = os.path.join(train_dir, "_".join([model_name, 
                                                 str(num_heads),
                                                 str(fixed_c_maxlen),
                                                 str(learning_rate),
                                                 soft_t5_activation,
                                                 trail
                                                 ]
                                                ))
  elif model_name in ['T5','T5_Nob']:
    t5_num_buckets=int(sys.argv[4])
    t5_max_distance=int(sys.argv[5])
    
    trail = sys.argv[6]
    
    dir_name = os.path.join(train_dir, 
                            "_".join([model_name, 
                                      str(num_heads),
                                      str(t5_num_buckets),
                                      str(t5_max_distance),
                                      trail
                                      ])
                            )
  else:
    trail = sys.argv[4]
    dir_name = os.path.join(train_dir, 
                            "_".join([model_name, 
                                      str(num_heads),
                                      trail
                                      ])
                            )
    
  if not os.path.exists(train_dir):
    os.mkdir(train_dir)
  
  if not os.path.exists(dir_name):
    os.mkdir(dir_name)
  
  event_log_dir = os.path.join(dir_name, "event")
  save_dir = os.path.join(dir_name, "model")
  answer_dir = os.path.join(dir_name, "answer")
  answer_file = os.path.join(answer_dir, "answer.json")
  
  if not os.path.exists(event_log_dir):
    os.makedirs(event_log_dir)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)
  
  config.mode=mode
  config.model=model_name
  config.num_heads=int(num_heads)
  config.trail = trail
  
  if config.model in ['Soft_T5','Soft_T5_TPE']:
    config.fixed_c_maxlen=int(fixed_c_maxlen)
    config.learning_rate=float(learning_rate)
    
    config.bucket_slop_min=bucket_slop_min
    config.bucket_slop_max=bucket_slop_max
    config.soft_t5_activation = soft_t5_activation
    
    config.l1_width=l1_width
    config.l2_width=l2_width
    config.stddev=stddev
  
  if config.model in ['Soft_T5_Nob']:
    config.fixed_c_maxlen=int(fixed_c_maxlen)
    config.learning_rate=float(learning_rate)
    
    config.soft_t5_activation = soft_t5_activation
  
  if config.model in ['T5','T5_TPE','T5_Nob']:
    config.t5_num_buckets=t5_num_buckets
    config.t5_max_distance=t5_max_distance
    
  config.event_log_dir=event_log_dir
  config.save_dir=save_dir
  config.answer_file = answer_file
  
  if config.mode == "train":
    train(config)
  elif config.mode == "prepro":
    prepro(config)
  elif config.mode == "debug":
    config.num_steps = 2
    config.val_num_batches = 1
    config.checkpoint = 1
    config.period = 1
    train(config)
  elif config.mode == "test":
    test(config)
  elif config.mode == "demo":
    demo(config)
  else:
    print("Unknown mode")
    exit(0)

if __name__ == "__main__":
    tf.app.run()
