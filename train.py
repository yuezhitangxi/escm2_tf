import tensorflow as tf
import numpy as np
import math
import json

from collections import defaultdict
from escm_reader import RecDataset  
from net import ESCMLayer




class ESCMTrainer:
    """ESCMLayer模型训练器，封装训练和评估逻辑"""
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
        
        
        self.global_w = self.config.get("hyper_parameters", {}).get("global_w", 1.0)
        self.counterfactual_w = self.config.get("hyper_parameters", {}).get("counterfactual_w", 0.01)
        
        
        self.sparse_feature_number = self.config.get("hyper_parameters", {})["sparse_feature_number"]
        self.sparse_feature_dim = self.config.get("hyper_parameters", {})["sparse_feature_dim"]
        self.num_field = self.config.get("hyper_parameters", {})["num_field"]
        self.feature_size = self.config.get("hyper_parameters", {})["feature_size"]
        
        
        self.ctr_fc_sizes = self.config.get("hyper_parameters", {}).get("ctr_fc_sizes", [])
        self.cvr_fc_sizes = self.config.get("hyper_parameters", {}).get("cvr_fc_sizes", [])
        self.expert_num = self.config.get("hyper_parameters", {})["expert_num"]
        self.expert_size = self.config.get("hyper_parameters", {})["expert_size"]
        self.tower_size = self.config.get("hyper_parameters", {})["tower_size"]
        
        
        self.learning_rate = self.config.get("hyper_parameters", {}).get("optimizer", {}).get("learning_rate", 0.001)
        self.counterfact_mode = self.config.get("runner", {})["counterfact_mode"]

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
        
        self.escm_layer = ESCMLayer(
            sparse_feature_number=self.sparse_feature_number,
            sparse_feature_dim=self.sparse_feature_dim,
            num_field=self.num_field,
            ctr_layer_sizes=self.ctr_fc_sizes,
            cvr_layer_sizes=self.cvr_fc_sizes,
            expert_num=self.expert_num,
            expert_size=self.expert_size,
            tower_size=self.tower_size,
            counterfact_mode=self.counterfact_mode,
            feature_size=self.feature_size
        )
        
        
        outputs = self.escm_layer(inputs)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def counterfact_ipw(self, loss_cvr, ctr_num, O, ctr_out_one):
        """反事实IPW损失计算"""
        # 计算倾向得分PS
        PS = tf.multiply(
            ctr_out_one, 
            tf.cast(ctr_num, dtype=tf.float32)
        )
        
        # 防止除零操作
        min_v = tf.fill(tf.shape(PS), 0.000001)
        PS = tf.maximum(PS, min_v)
        
        # 计算IPS权重
        IPS = tf.math.reciprocal(PS)
        batch_shape = tf.fill(tf.shape(O), 1)
        batch_size = tf.reduce_sum(tf.cast(batch_shape, dtype=tf.float32))
        
        # 截断IPS范围
        IPS = tf.clip_by_value(IPS, -15, 15)
        IPS = tf.multiply(IPS, batch_size)
        IPS = tf.stop_gradient(IPS)  # 停止梯度传播
        
        # 计算加权损失
        loss_cvr = tf.multiply(loss_cvr, IPS)
        loss_cvr = tf.multiply(loss_cvr, O)
        
        return tf.reduce_mean(loss_cvr)

    def counterfact_dr(self, loss_cvr, O, ctr_out_one, imp_out):
        """反事实DR损失计算"""
        # 计算误差项 e = loss_cvr - imp_out
        e = tf.subtract(loss_cvr, imp_out)
        
        # 防止除零操作
        min_v = tf.fill(tf.shape(ctr_out_one), 0.000001)
        ctr_out_one = tf.maximum(ctr_out_one, min_v)
        
        # 计算IPS权重
        IPS = tf.divide(tf.cast(O, dtype=tf.float32), ctr_out_one)
        IPS = tf.clip_by_value(IPS, -15, 15)
        IPS = tf.stop_gradient(IPS)
        
        # 计算DR误差项
        loss_error_second = tf.multiply(e, IPS)
        loss_error = tf.add(imp_out, loss_error_second)
        
        # 计算DR正则项
        loss_imp = tf.square(e)
        loss_imp = tf.multiply(loss_imp, IPS)
        
        # 总DR损失
        loss_dr = tf.add(loss_error, loss_imp)
        
        return tf.reduce_mean(loss_dr)

   
    def train_step(self, batch_data):
        """单步训练"""
        
        features = list(batch_data[:-2])
        ctr_clk = batch_data[-2]
        ctcvr_buy = batch_data[-1]
        
        
        ctr_clk_float = tf.cast(ctr_clk, dtype=tf.float32)
        ctcvr_buy_float = tf.cast(ctcvr_buy, dtype=tf.float32)

       
        
        with tf.GradientTape() as tape:

            out_list = self.model(features, training=True)
            (ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one) = out_list[0:6]

            
            
            ctr_num = tf.reduce_sum(ctr_clk_float, axis=0)
            O = ctr_clk_float  # 观测指示变量(是否点击)

            
            loss_ctr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=ctr_clk_float,
                logits=tf.math.log(ctr_out_one / (1-ctr_out_one))  
            ))
            
            
            
            loss_cvr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=ctcvr_buy_float,
                logits=tf.math.log(cvr_out_one / (1 - cvr_out_one))  
            ))
            
            # 根据反事实模式调整CVR损失
            if self.counterfact_mode == "DR":  
                loss_cvr = self.counterfact_dr(loss_cvr, O, ctr_out_one, out_list[6])
            else:
                loss_cvr = self.counterfact_ipw(loss_cvr, ctr_num, O, ctr_out_one)
            
            
            loss_ctcvr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=ctcvr_buy_float,
                logits=tf.math.log(ctcvr_prop_one / (1 - ctcvr_prop_one))
            ))
            

            
            total_loss = loss_ctr + loss_cvr * self.counterfactual_w + loss_ctcvr * self.global_w
        
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # 更新AUC指标
        self.auc_ctr.update_state(ctr_clk_float, ctr_out_one)
        self.auc_cvr.update_state(ctcvr_buy_float, cvr_out_one)
        self.auc_ctcvr.update_state(ctcvr_buy_float, ctcvr_prop_one)

        
        return {
            'total_loss': total_loss,
            'loss_ctr': loss_ctr,
            'loss_cvr': loss_cvr,
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
            
            out_list = self.model(features, training=False)
            
            ctr_out_one = out_list[1]
            cvr_out_one = out_list[3]
            ctcvr_prop_one = out_list[5]

            
            self.auc_ctr.update_state(ctr_clk_float, ctr_out_one)
            self.auc_cvr.update_state(ctcvr_buy_float, cvr_out_one)
            self.auc_ctcvr.update_state(ctcvr_buy_float, ctcvr_prop_one)
        
        return {
            'auc_ctr': self.auc_ctr.result().numpy(),
            'auc_cvr': self.auc_cvr.result().numpy(),
            'auc_ctcvr': self.auc_ctcvr.result().numpy()
        }



def main():
    
    # 读取配置文件
    with open("config.json", "r") as f:
        config = json.load(f)

    
    train_dataset = RecDataset(config["data"]["train_files"], config)
    test_dataset = RecDataset(config["data"]["test_files"], config)
    
   
    trainer = ESCMTrainer(config)
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
        
        # # 在测试集上评估
        # eval_metrics = trainer.evaluate(test_dataset)
        # print(f"测试集指标 - "
        #       f"AUC_ctr: {eval_metrics['auc_ctr']:.4f} - "
        #       f"AUC_cvr: {eval_metrics['auc_cvr']:.4f} - "
        #       f"AUC_ctcvr: {eval_metrics['auc_ctcvr']:.4f}")
    
   
    trainer.model.save("escm_model.keras")
    print("\n模型已保存至 escm_model 目录")


if __name__ == "__main__":
    main()