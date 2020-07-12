# Model Parallel For ResNet50
## dataset 说明
--dataset imagenette 为了让加速实验，在一开始会把所有数据读入内存[ImagenetteDataset](data/loader.py)，这样 load 数据不会成为瓶颈。多开几个 load worker 也能解除 load 数据的瓶颈，但如果到硬盘 io 就没法解除，而且还有别的任务要读硬盘，所以干脆内存。GPU 等待数据读取而空闲的时间不好测量，必须去掉。

因为主题是测速，所以随便找了个 ImageNet 的子集 [Imagenette](https://github.com/fastai/imagenette)
另外有一个问题是，在 validate 的时候会有大量时间拷贝，GPU 跑不满。所以所有测速实验都开 --no-validate 让测速更纯粹。

--dataset random 随机产生 8*8 小图，用来跑通 data flow，检查代码。不用来测速。（其实调大图片也可以用来测速）

一开始没有 sync，一查发现 pytorch 操作 cuda 全部异步，测量 forward 不准。对于 data para 还是开着 async。

## 实验结果
