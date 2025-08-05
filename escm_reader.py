import numpy as np
import tensorflow as tf
from collections import defaultdict

class RecDataset:
    def __init__(self, file_list, config):
        self.config = config
        self.file_list = file_list
        self.init()

        self.total_samples = self._count_total_samples()
        self.dataset = tf.data.Dataset.from_generator(
            self._generate_data,
            output_signature=self._get_output_signature()
        )
        
        
        self.batch_size = self.config.get("runner", {}).get("batch_size", 32)
        self.dataset = self.dataset.batch(self.batch_size)
        
       
        self.total_steps_per_epoch = (self.total_samples + self.batch_size - 1) // self.batch_size
        self.iterator = iter(self.dataset)

    def init(self):
        all_field_id = [
            '101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124',
            '125', '126', '127', '128', '129', '205', '206', '207', '210',
            '216', '508', '509', '702', '853', '301'
        ]
        self.all_field_id_dict = defaultdict(int)
        self.max_len = self.config.get("hyper_parameters", {}).get("max_len", 3)
        for i, field_id in enumerate(all_field_id):
            self.all_field_id_dict[field_id] = [False, i]
        self.padding = 0
        self.num_fields = len(all_field_id)
    
    def reset(self):
        """重置数据集迭代器，用于每个epoch开始时"""
        self.iterator = iter(self.dataset)

    def _count_total_samples(self):
        """统计所有文件的总样本数（行数）"""
        total = 0
        for file in self.file_list:
            with open(file, "r") as f:
                for _ in f:
                    total += 1
        return total

    def _generate_data(self):
        """生成器：确保遍历所有样本后终止，不无限循环"""
        for file in self.file_list:
            with open(file, "r") as rf:
                for line in rf: 
                    features = line.strip().split(',')
                    if len(features) < 5:  
                        continue
                    
                    try:
                        ctr = int(features[1])
                        ctcvr = int(features[2])
                    except (IndexError, ValueError):
                        continue  # 跳过格式错误的行
                    
                    output_list = []
                    # 初始化所有字段的特征列表
                    field_features = [[] for _ in range(self.num_fields)]
                    
                    # 解析特征
                    for elem in features[4:]:
                        if ':' not in elem:
                            continue
                        field_id, feat_id = elem.strip().split(':', 1)  # 限制分割次数，避免异常
                        if field_id not in self.all_field_id_dict:
                            continue
                        _, index = self.all_field_id_dict[field_id]
                        try:
                            field_features[index].append(int(feat_id))
                        except ValueError:
                            continue  # 跳过非整数的特征ID
                    
                    # 处理每个字段的特征（截断或补全）
                    for feat in field_features:
                        if len(feat) > self.max_len:
                            processed = feat[:self.max_len]
                        else:
                            processed = feat + [self.padding] * (self.max_len - len(feat))
                        output_list.append(np.array(processed, dtype='int64'))
                    
                    # 添加标签
                    output_list.append(np.array([ctr], dtype='int64'))
                    output_list.append(np.array([ctcvr], dtype='int64'))
                    
                    yield tuple(output_list)
        # 所有样本处理完毕后，生成器自动终止

    def _get_output_signature(self):
        signature = []
        # 特征部分签名
        for _ in range(self.num_fields):
            signature.append(tf.TensorSpec(shape=(self.max_len,), dtype=tf.int64))
        # 标签部分签名
        signature.append(tf.TensorSpec(shape=(1,), dtype=tf.int64))  # ctr标签
        signature.append(tf.TensorSpec(shape=(1,), dtype=tf.int64))  # ctcvr标签
        return tuple(signature)

    def get_next(self):
        """获取下一个批次，数据耗尽时抛出StopIteration"""
        return next(self.iterator)  # 不再自动重置，让训练循环处理终止

    def __str__(self):
        """自定义对象打印信息"""
        return (
            f"RecDataset(\n"
            f"  file_list: {self.file_list}\n"
            f"  total_samples: {self.total_samples}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  total_steps_per_epoch: {self.total_steps_per_epoch}\n"
            f"  num_fields: {self.num_fields}\n"
            f")"
        )    
    