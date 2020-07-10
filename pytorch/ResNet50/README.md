# Model Parallel For ResNet50
--dataset imagenette 为了让加速实验，在一开始会把所有数据读入内存[ImagenetteDataset](data/loader.py)，这样 load 数据不会成为瓶颈。多开几个 load worker 也能解除 load 数据的瓶颈，但如果到硬盘 io 就没法解除，而且还有别的任务要读硬盘，所以干脆内存。GPU 等待数据读取而空闲的时间不好测量，必须去掉。

因为主题是测速，所以随便找了个 ImageNet 的子集 [Imagenette](https://github.com/fastai/imagenette)
另外有一个问题是，在 validate 的时候会有大量时间拷贝，GPU 跑不满。所以所有测速实验都开 --no-validate 让测速更纯粹。

model parallel 的效果好像没那么好，不知道是不是因为另外一个巨大实验占了 device to device 的带宽。明天起来再跑跑看，并做做 model share 的对照。我猜更有可能是分析的时候想太好了，考虑 backward 可能就不是那么回事了。观察到 forward 减少的很多，但是 backward 时间反而增加了，很奇怪，按理来说应该 backward 也更快呀。倒是不用 pipeline backward 还快一点。

找了三篇相关论文，去读一读。