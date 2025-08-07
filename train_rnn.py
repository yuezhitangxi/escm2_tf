import tensorflow as tf
import numpy as np
import math
import json
import random

from collections import defaultdict
from escm_reader import RecDataset  
from rnn import CustomRNNLayer


def set_random_seed(seed=42):
    
    random.seed(seed)   
    np.random.seed(seed)
    tf.random.set_seed(seed)

class RNNTrainer:
    """CustomRNNLayer模型训练器，封装训练和评估逻辑"""
    def __init__(self, config):
        self.config = config
        self._init_hyper_parameters()
        self._init_metrics()
        self._build_model()
        
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

    def _init_hyper_parameters(self):
        """初始化超参数"""
        
        self.max_len = self.config.get("hyper_parameters", {}).get("max_len", 3)
        self.sparse_feature_number = self.config.get("hyper_parameters", {})["sparse_feature_number"]
        self.sparse_feature_dim = self.config.get("hyper_parameters", {})["sparse_feature_dim"]
        self.num_field = self.config.get("hyper_parameters", {})["num_field"]
        self.feature_size = self.config.get("hyper_parameters", {})["feature_size"]
        self.hidden_dim = self.config.get("hyper_parameters", {})["hidden_dim"]
        self.output_dim = self.config.get("hyper_parameters", {})["output_dim"]
        
        self.learning_rate = self.config.get("hyper_parameters", {}).get("optimizer", {}).get("learning_rate", 0.001)
      

    def _init_metrics(self):
        """初始化评估指标（显式指定参数确保兼容性）"""
        
        self.auc_ctr = tf.keras.metrics.AUC(
            num_thresholds=200,
            curve='ROC',
            summation_method='interpolation',
            name='auc_ctr'
        )
        self.auc_cvr = tf.keras.metrics.AUC(
            num_thresholds=200,
            curve='ROC',
            summation_method='interpolation',
            name='auc_cvr'
        )
        self.auc_ctcvr = tf.keras.metrics.AUC(
            num_thresholds=200,
            curve='ROC',
            summation_method='interpolation',
            name='auc_ctcvr'
        )

    def _build_model(self):
        """构建模型"""
        
        inputs = [
            tf.keras.Input(shape=(self.max_len,), dtype=tf.int64, name=f"field_{i}")
            for i in range(self.num_field)
        ]
        
        self.rnn_layer = CustomRNNLayer(
            sparse_feature_number=self.sparse_feature_number,
            sparse_feature_dim=self.sparse_feature_dim,
            num_field=self.num_field,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )
        
        
        outputs = self.rnn_layer(inputs)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)


   
    def train_step(self, batch_data):
        """单步训练"""
        features = list(batch_data[:-2])
        ctr_clk = batch_data[-2]
        ctcvr_buy = batch_data[-1]
        
        
        ctr_clk_float = tf.cast(ctr_clk, dtype=tf.float32)
        ctcvr_buy_float = tf.cast(ctcvr_buy, dtype=tf.float32)

       
        
        with tf.GradientTape() as tape:

            ctr_prop_one, cvr_prop_one = self.model(features, training=True)
            ctcvr_prop_one = ctr_prop_one * cvr_prop_one

            ctr_num = tf.reduce_sum(ctr_clk_float, axis=0)
            O = ctr_clk_float  # 观测指示变量(是否点击)

            
            loss_ctr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=ctr_clk_float,
                logits=tf.math.log(ctr_prop_one / (1-ctr_prop_one))  
            ))
            
            
            loss_ctcvr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=ctcvr_buy_float,
                logits=tf.math.log(ctcvr_prop_one / (1 - ctcvr_prop_one))
            ))
            

            
            total_loss = loss_ctr + loss_ctcvr
        
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    

        # 过滤出“点击过的样本”（仅这些样本参与CVR评估）
        click_mask = tf.equal(ctr_clk_float, 1.0)  # 掩码：点击的样本为True
        cvr_true = tf.boolean_mask(ctcvr_buy_float, click_mask)  # 仅保留点击样本的真实标签
        cvr_pred = tf.boolean_mask(cvr_prop_one, click_mask)  # 仅保留点击样本的预测值

        # 更新AUC指标
        self.auc_ctr.update_state(ctr_clk_float, ctr_prop_one)
        self.auc_ctcvr.update_state(ctcvr_buy_float, ctcvr_prop_one)

        # 仅用点击样本更新CVR的AUC
        if tf.size(cvr_true) > 0:  # 避免空样本导致的错误
            self.auc_cvr.update_state(cvr_true, cvr_pred)

        
        return {
            'total_loss': total_loss,
            'loss_ctr': loss_ctr,
            'loss_ctcvr': loss_ctcvr
        }

    def evaluate(self, dataset):
        """评估模型"""
        
        try:
            self.auc_ctr.reset_states()
            self.auc_cvr.reset_states()
            self.auc_ctcvr.reset_states()
        except AttributeError:
            self._init_metrics()
        
        for batch_data in dataset.dataset:
            
            features = list(batch_data[:-2])
            ctr_clk = batch_data[-2]
            ctcvr_buy = batch_data[-1]
            
            ctr_clk_float = tf.cast(ctr_clk, dtype=tf.float32)
            ctcvr_buy_float = tf.cast(ctcvr_buy, dtype=tf.float32)


            
            
            ctr_prop_one,cvr_prop_one = self.model(features, training=False)
            ctcvr_prop_one = ctr_prop_one * cvr_prop_one

            # 过滤点击样本
            click_mask = tf.equal(ctr_clk_float, 1.0)
            cvr_true = tf.boolean_mask(ctcvr_buy_float, click_mask)
            cvr_pred = tf.boolean_mask(cvr_prop_one, click_mask)
            


            
            self.auc_ctr.update_state(ctr_clk_float, ctr_prop_one)
            self.auc_ctcvr.update_state(ctcvr_buy_float, ctcvr_prop_one)

            if tf.size(cvr_true) > 0:
                self.auc_cvr.update_state(cvr_true, cvr_pred)
        
        return {
            'auc_ctr': self.auc_ctr.result().numpy(),
            'auc_cvr': self.auc_cvr.result().numpy(),
            'auc_ctcvr': self.auc_ctcvr.result().numpy()
        }



def main():
    set_random_seed(seed=42)
    # 读取配置文件
    with open("rnn_config.json", "r") as f:
        config = json.load(f)

    
    train_dataset = RecDataset(config["data"]["train_files"], config)
    test_dataset = RecDataset(config["data"]["test_files"], config)
    
   
    trainer = RNNTrainer(config)
    epochs = config["runner"]["epochs"]
    log_steps = config["runner"]["log_steps"]
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        try:
            trainer.auc_ctr.reset_states()
            trainer.auc_cvr.reset_states()
            trainer.auc_ctcvr.reset_states()
        except AttributeError:
            trainer._init_metrics()  

        train_dataset.reset()
        
        total_loss = 0.0
        step = 0
        
        
        try:
            while True:
                batch_data = train_dataset.get_next()
                losses = trainer.train_step(batch_data)
                
                total_loss += losses['total_loss']
                step += 1
                
                
                if step % log_steps == 0:
                    avg_loss = total_loss / step
                    print(f"Step {step} - "
                      f"平均损失: {avg_loss.numpy().item():.4f} - "  
                      f"auc_ctr: {trainer.auc_ctr.result().numpy().item():.4f} - "
                      f"auc_cvr: {trainer.auc_cvr.result().numpy().item():.4f} - "
                      f"auc_ctcvr: {trainer.auc_ctcvr.result().numpy().item():.4f}")
        except StopIteration:
            pass
        
        # 计算 epoch 平均损失
        avg_epoch_loss = total_loss / step
        print(f"\nEpoch {epoch + 1} 训练完成 - "
              f"平均损失: {avg_epoch_loss:.4f} - "
              f"最终AUC_ctr: {trainer.auc_ctr.result():.4f} - "
              f"AUC_cvr: {trainer.auc_cvr.result():.4f} - "
              f"AUC_ctcvr: {trainer.auc_ctcvr.result():.4f}")
        
        # 在测试集上评估
        eval_metrics = trainer.evaluate(test_dataset)
        print(f"测试集指标 - "
              f"AUC_ctr: {eval_metrics['auc_ctr']:.4f} - "
              f"AUC_cvr: {eval_metrics['auc_cvr']:.4f} - "
              f"AUC_ctcvr: {eval_metrics['auc_ctcvr']:.4f}")
    
   
    trainer.model.save("escm_model.keras")
    print("\n模型已保存至 escm_model 目录")


if __name__ == "__main__":
    main()