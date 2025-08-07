# escm2_tf

## 依赖包
tensorflow2.x

## 下载数据集
1.确认当前所在目录为escm2_tf<br/>
2.ali-ccp目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的ali-ccp全量数据集，并解压到指定文件夹
``` bash
cd ali-ccp
sh run.sh
```
3.切回模型目录,执行命令运行全量数据
```bash
cd .. # 切回模型
python train.py #训练escm2
python train_rnn.py #训练rnn
```
