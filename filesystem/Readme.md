#文件系统若干技巧
## sles11sp3使用ext4
```text
加载驱动：
modprobe jbd2
modprobe crc16
insmod /lib/modules/2.6.32.59-0.7-default/weak-updates/extra/ext4.ko

编辑 /etc/mke2fs.conf
[defaults]
  base_features = sparse_super,filetype,resize_inode,dir_index,ext_attr
  blocksize = 2048
  inode_size = 128
  inode_ratio = 4096
[fs_types]
  ext4 = {
    features = extents,flex_bg,uninit_bg,dir_nlink,extra_isize
    inode_size = 128
    inode_ratio = 4096
  }
  
/sys/block/sdm/queue/read_ahead_kb = 16
/sys/block/sdm/queue/scheduler = deadline
/sys/block/sdm/queue/max_sectors_kb = 128
/sys/block/sdm/queue/iosched/fifo_batch = 64
/sys/block/sdm/queue/iosched/read_expire = 200
/sys/block/sdm/device/queue_depth = 32
/sys/fs/ext4/sdm4/mb_stream_req = 16

创建文件系统：
mkfs.ext4 -N 500000 -I 128 -b 4096 -E lazy_itable_init=1 -O dir_index,uninit_bg,^has_journal -G 4 /dev/sdb

挂载分区：
mount -t ext4 -o data=writeback,errors=remount-ro,nodiratime,noatime,norelatime,nobh,barrier=0,inode_readahead_blks=32,orlov /dev/sdb /home/disk

对于 sles11sp3，需要vi /etc/modprobe.d/ext4，添加一条：options ext4 rw = 1
```
