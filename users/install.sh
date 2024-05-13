## 创建软链接

# exp -> /data3/litianhao/workdir/fairmot/
# models -> /data3/litianhao/checkpoints/fairmot/

ln -s /data3/litianhao/workdir/fairmot/ exp
ln -s /data3/litianhao/checkpoints/fairmot/ models


## 安装子库
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh