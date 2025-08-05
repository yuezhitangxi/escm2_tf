import tensorflow as tf
import math

class ESCMLayer(tf.keras.layers.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim, num_field,
                 ctr_layer_sizes, cvr_layer_sizes, expert_num, expert_size,
                 tower_size, counterfact_mode, feature_size, scope_name="escm_layer", **kwargs):
        super(ESCMLayer, self).__init__(** kwargs)  # 调用父类构造函数
        self.sparse_feature_number = sparse_feature_number  # 稀疏特征总数
        self.sparse_feature_dim = sparse_feature_dim        # 每个稀疏特征的嵌入维度
        self.num_field = num_field                          # 特征域数量
        self.expert_num = expert_num                        # 专家数量
        self.expert_size = expert_size                      # 专家输出维度
        self.tower_size = tower_size                        # 塔层维度
        self.counterfact_mode = counterfact_mode            # 反事实模式（DR或其他）
        self.feature_size = feature_size                    # 输入特征总维度
        self.scope_name = scope_name                        # 变量作用域
        
        # 根据模式确定gate数量（DR模式需要3个gate，其他模式2个）
        self.gate_num = 3 if counterfact_mode == "DR" else 2

        # 初始化所有变量（使用Keras层的变量管理）
        self._init_embedding()
        self._init_experts()
        self._init_gates()
        self._init_towers()
    


    def _init_embedding(self):
        """初始化嵌入层（使用Keras的add_weight）"""
        # 嵌入权重：[sparse_feature_number, sparse_feature_dim]
        self.embedding_weights = self.add_weight(
            name="SparseFeatFactors",
            shape=[self.sparse_feature_number, self.sparse_feature_dim],
            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        )

    def _init_experts(self):
        """初始化专家层（每个专家是一个线性层）"""
        self.experts = []
        for i in range(self.expert_num):
            # 专家层权重：[feature_size, expert_size]
            w = self.add_weight(
                name=f"expert_{i}_w",
                shape=[self.feature_size, self.expert_size],
                initializer=tf.keras.initializers.GlorotUniform()  # Xavier均匀初始化
            )
            # 专家层偏置：[expert_size]
            b = self.add_weight(
                name=f"expert_{i}_b",
                shape=[self.expert_size],
                initializer=tf.constant_initializer(0.1)
            )
            self.experts.append((w, b))

    def _init_gates(self):
        """初始化门控层（每个gate是一个线性层）"""
        self.gates = []
        for i in range(self.gate_num):
            # 门控层权重：[feature_size, expert_num]
            w = self.add_weight(
                name=f"gate_{i}_w",
                shape=[self.feature_size, self.expert_num],
                initializer=tf.keras.initializers.GlorotUniform()
            )
            # 门控层偏置：[expert_num]
            b = self.add_weight(
                name=f"gate_{i}_b",
                shape=[self.expert_num],
                initializer=tf.constant_initializer(0.1)
            )
            self.gates.append((w, b))

    def _init_towers(self):
        """初始化塔层（每个tower包含两层线性层）"""
        self.towers = []  # 每个元素是 (tower_w, tower_b, out_w, out_b)
        for i in range(self.gate_num):
            # 塔层第一层权重：[expert_size, tower_size]
            tower_w = self.add_weight(
                name=f"tower_{i}_w",
                shape=[self.expert_size, self.tower_size],
                initializer=tf.keras.initializers.GlorotUniform()
            )
            tower_b = self.add_weight(
                name=f"tower_{i}_b",
                shape=[self.tower_size],
                initializer=tf.constant_initializer(0.1)
            )
            # 塔层输出层权重：[tower_size, 2]（二分类）
            out_w = self.add_weight(
                name=f"tower_out_{i}_w",
                shape=[self.tower_size, 2],
                initializer=tf.keras.initializers.GlorotUniform()
            )
            out_b = self.add_weight(
                name=f"tower_out_{i}_b",
                shape=[2],
                initializer=tf.constant_initializer(0.1)
            )
            self.towers.append((tower_w, tower_b, out_w, out_b))

    def call(self, inputs, training=None):
        emb_list = []
        for data in inputs:  
            feat_emb = tf.nn.embedding_lookup(self.embedding_weights, data)
            feat_emb_sum = tf.reduce_sum(feat_emb, axis=1)
            emb_list.append(feat_emb_sum)
        

        concat_emb = tf.concat(emb_list, axis=1)

        expert_outputs = []
        for i in range(self.expert_num):
            w, b = self.experts[i]
            linear_out = tf.matmul(concat_emb, w) + b
            expert_out = tf.nn.relu(linear_out)
            expert_outputs.append(expert_out)
        expert_concat = tf.concat(expert_outputs, axis=1)
        expert_concat = tf.reshape(expert_concat, [-1, self.expert_num, self.expert_size])

        
        output_layers = []
        for i in range(self.gate_num):
            gate_w, gate_b = self.gates[i]
            cur_gate_linear = tf.matmul(concat_emb, gate_w) + gate_b
            cur_gate = tf.nn.softmax(cur_gate_linear)
            cur_gate = tf.reshape(cur_gate, [-1, self.expert_num, 1])
            cur_gate_expert = tf.multiply(expert_concat, cur_gate)
            cur_gate_expert = tf.reduce_sum(cur_gate_expert, axis=1)
            
           
            # 塔层计算（第一层）
            tower_w, tower_b, out_w, out_b = self.towers[i]
            tower_out = tf.nn.relu(tf.matmul(cur_gate_expert, tower_w) + tower_b)  # [batch_size, tower_size]
            
            # 塔层输出（第二层，二分类logits）
            logits = tf.matmul(tower_out, out_w) + out_b  # [batch_size, 2]
            # 计算概率
            out = tf.nn.softmax(logits)
            out = clipped_out = tf.clip_by_value(out, clip_value_min=1e-15, clip_value_max=1.0 - 1e-15)
            output_layers.append(out)
        
        
        ctr_out = output_layers[0]
        cvr_out = output_layers[1]

        ctr_prop_one = tf.slice(ctr_out, [0, 1], [-1, 1])
        cvr_prop_one = tf.slice(cvr_out, [0, 1], [-1, 1])
        ctcvr_prop_one = tf.multiply(ctr_prop_one, cvr_prop_one)
        ctcvr_prop = tf.concat([1 - ctcvr_prop_one, ctcvr_prop_one], axis=1)

        out_list = []
        out_list.append(ctr_out)
        out_list.append(ctr_prop_one)
        out_list.append(cvr_out)
        out_list.append(cvr_prop_one)
        out_list.append(ctcvr_prop)
        out_list.append(ctcvr_prop_one)


        if self.counterfact_mode == "DR":
            imp_out = output_layers[2]
            imp_prop_one = tf.slice(imp_out,[0, 1], [-1, 1])
            out_list.append(imp_prop_one)

        return out_list
