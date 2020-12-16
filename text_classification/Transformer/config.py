import tensorflow as tf
# Model Hyperparameters
flags = tf.app.flags

flags.DEFINE_integer("embedding_dim", 128,
                     "Dimensionality of character embedding (default: 384)")
flags.DEFINE_integer("num_hidden_layers", 1,
                     "Multi-head attention layer")
flags.DEFINE_integer("num_attention_heads", 1,
                     "attention head")
flags.DEFINE_integer("training_nums", 20000,
                     "training_nums")
flags.DEFINE_string("filter_sizes", "3,4,5",
                    "Comma-separated filter sizes (default: '3,4,5')")
#flags.DEFINE_string("filter_sizes", "5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 128,
                     "Number of filters per filter size (default: 128)")
flags.DEFINE_integer(
    "hidden_num", 100, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5,
                   "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0,
                   "L2 regularizaion lambda (default: 0.0)")
flags.DEFINE_float("learning_rate", 0.0001, "learn rate( default: 0.0)")
flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
flags.DEFINE_string("loss", "point_wise", "loss function (default:point_wise)")
flags.DEFINE_integer('extend_feature_dim', 10, 'overlap_feature_dim')
# Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_boolean(
    "trainable", True, "is embedding trainable? (default: False)")
flags.DEFINE_integer(
    "num_epochs", 40, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 500,
                     "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 500,
                     "Save model after this many steps (default: 100)")
flags.DEFINE_boolean('overlap_needed', False, "is overlap used")
flags.DEFINE_boolean('position_needed', False, 'is position used')
flags.DEFINE_boolean('is_training', True, 'is position used')
flags.DEFINE_boolean('dns', 'False', 'whether use dns or not')
flags.DEFINE_string('data', 'mr', 'data set')
flags.DEFINE_float('sample_train', 1, 'sampe my train data')
flags.DEFINE_boolean(
    'fresh', True, 'wheather recalculate the embedding or overlap default is True')
flags.DEFINE_string('pooling', 'max', 'pooling strategy')
flags.DEFINE_boolean('clean', False, 'whether clean the data')
# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True,
                     "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False,
                     "Log placement of ops on devices")
# data_helper para
flags.DEFINE_boolean('isEnglish', True, 'whether data is english')

flags.DEFINE_string(
    "trail", '1', "For every model, we run for 5 times")
flags.DEFINE_string(
    "model_name", 'Non_PE', "Non_PE, PE_concat, PE_reduce")

flags.DEFINE_integer("n_fold", 10, "the number of Cross-validation for mr,subj,cr,mpqa")

flags.DEFINE_integer("t5_bucket", 32, "the number of bucket  for T5 relative positon")

flags.DEFINE_string("transformer_ret_pooling","last","mean, last or sum")

flags.DEFINE_boolean("is_Embedding_Needed",True, "whether to use the pre-trained word embedding")

#bucket_slop_min
flags.DEFINE_float('bucket_slop_min', 1.0, 'bucket_slop_min')
flags.DEFINE_float('bucket_slop_max', 10.0, 'bucket_slop_max')
flags.DEFINE_integer('l1_width', 100, 'mlp l1 width')
flags.DEFINE_integer('l2_width', 40, 'mlp l2 width')
flags.DEFINE_float('stddev', 0.01, 'initialization stddev')