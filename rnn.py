import tensorflow as tf
import math

class CustomRNNLayer(tf.keras.layers.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim, num_field,
                 hidden_dim, output_dim=2, scope_name="custom_rnn_layer",** kwargs):
        super(CustomRNNLayer, self).__init__(**kwargs)
        self.sparse_feature_number = sparse_feature_number  # 稀疏特征总数
        self.sparse_feature_dim = sparse_feature_dim        # 每个稀疏特征的嵌入维度
        self.num_field = num_field                          # 特征域数量
        self.hidden_dim = hidden_dim                        # RNN隐藏层维度
        self.output_dim = output_dim                        # 输出维度，固定为2
        self.scope_name = scope_name                        # 变量作用域
        
        # 初始化嵌入层和RNN权重
        self._init_embedding()
        self._init_rnn_weights()

    def _init_embedding(self):
        """初始化嵌入层，与ESCMLayer保持一致"""
        # 嵌入权重：[sparse_feature_number, sparse_feature_dim]
        self.embedding_weights = self.add_weight(
            name="SparseFeatFactors",
            shape=[self.sparse_feature_number, self.sparse_feature_dim],
            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        )

    def _init_rnn_weights(self):
        """初始化RNN相关权重"""
        self.feature_size = self.num_field * self.sparse_feature_dim
        
        # 输入到隐藏层的权重
        self.w_ih = self.add_weight(
            shape=(self.feature_size, self.hidden_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            name='input_to_hidden'
        )
        
        # 隐藏层到隐藏层的权重
        self.w_hh = self.add_weight(
            shape=(self.hidden_dim, self.hidden_dim),
            initializer=tf.keras.initializers.Orthogonal(),
            name='hidden_to_hidden'
        )
        
        # 隐藏层到输出层的权重
        self.w_ho = self.add_weight(
            shape=(self.hidden_dim, self.output_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            name='hidden_to_output'
        )
        
        # 偏置项
        self.b_h = self.add_weight(
            shape=(self.hidden_dim,),
            initializer=tf.keras.initializers.Zeros(),
            name='hidden_bias'
        )
        
        self.b_o = self.add_weight(
            shape=(self.output_dim,),
            initializer=tf.keras.initializers.Zeros(),
            name='output_bias'
        )

    def call(self, inputs, training=None):
        """
        前向传播计算，输入处理方式与ESCMLayer保持一致
        
        参数:
            inputs: 输入列表，每个元素是一个特征域的索引张量
            training: 是否处于训练模式
        
        返回:
            outputs: 输出张量，形状为 (batch_size, 2, output_dim)，表示长度为2的序列
        """
        # 嵌入处理，与ESCMLayer保持一致
        emb_list = []
        for data in inputs:  
            feat_emb = tf.nn.embedding_lookup(self.embedding_weights, data)
            feat_emb_sum = tf.reduce_sum(feat_emb, axis=1)
            emb_list.append(feat_emb_sum)
        
        # 拼接嵌入向量，得到与ESCMLayer相同的输入特征
        concat_emb = tf.concat(emb_list, axis=1)  # 形状: (batch_size, feature_size)
        
        # 初始化隐藏状态
        batch_size = tf.shape(concat_emb)[0]
        h = tf.zeros((batch_size, self.hidden_dim))
        
        
        
        # 第一个时间步：使用输入特征
        h = tf.tanh(tf.matmul(concat_emb, self.w_ih) + tf.matmul(h, self.w_hh) + self.b_h)
        o1 = tf.matmul(h, self.w_ho) + self.b_o
        out1 = tf.nn.softmax(o1)
        
        
        # 第二个时间步：继续计算，生成第二个输出
        # 这里可以根据需要调整，比如是否使用新输入或仅使用隐藏状态
       
        h = tf.tanh(tf.matmul(concat_emb, self.w_ih) + tf.matmul(h, self.w_hh) + self.b_h)
        o2 = tf.matmul(h, self.w_ho) + self.b_o
        out2 = tf.nn.softmax(o2)

        ctr_prop_one = tf.slice(out1, [0, 1], [-1, 1])
        cvr_prop_one = tf.slice(out2, [0, 1], [-1, 1])
        
        return ctr_prop_one,cvr_prop_one

    def get_config(self):
        """获取层的配置信息"""
        config = super(CustomRNNLayer, self).get_config()
        config.update({
            'sparse_feature_number': self.sparse_feature_number,
            'sparse_feature_dim': self.sparse_feature_dim,
            'num_field': self.num_field,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'scope_name': self.scope_name
        })
        return config
