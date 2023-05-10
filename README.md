# moziai

moziai是在墨子推演平台上运行强化学习的python组件，目的是方便大家在墨子平台开发强化学习算法，墨子平台提供了与强化学习的python代码进行交互的界面，以获取状态空间并执行动作。用户可以基于墨子AI开发包开展深度学习、机器学习、对抗博弈、多智能体、行为树等多种模式的人工智能研究，以期在战术战法研究、效能评估、智能蓝军、规则学习与策略更新等多个应用领域产生突破性成果。

## 快速入门指南

### 获取moziai

#### windows

获取moziai软件包最简单的方法是从gitee克隆到本地：

>git clone [https://gitee.com/hs-defense/moziai.git](https://gitee.com/hs-defense/moziai.git)
python代码的运行版本为python3.6.9，python依赖包可以参考代码仓库中的requirements.txt文件，通过 pip install -r requirement.txt安装相关的环境依赖。

第二种方法 使用 pip install 安装moziai，会自动下载python依赖。由于moziai主要是为用户提供案例demo和与“墨子”推演系统的接口，所以pip安装后，可以到 ./Lib/site-packages下把mozi_ai_sdk,mozi_simu_sdk,mozi_utils三个文件夹复制到桌面或任意位置，然后把该环境的python设为moziai项目的python 解释器。

>pip install mozi-ai -i [https://pypi.org/simple/](https://pypi.org/simple/) 
推荐使用第一种方法获取moziai，针对第一种方法，也可到[https://www.hs-defense.com/col.jsp?id=105](https://www.hs-defense.com/col.jsp?id=105) mozi·AI开发包（windows）下载离线的python开发环境安装包和安装流程文档。

#### linux

环境要求：centos 7.6，其他版本未测试，docker>=19.0

Linux版本的“墨子”推演系统是一个docker镜像，安装比较麻烦。可以到[https://www.hs-defense.com/col.jsp?id=105](https://www.hs-defense.com/col.jsp?id=105) 页面中间下载：**全国兵棋推演大赛专项赛智能体开发平台**离线安装包，下载的压缩包中有安装流程文档。


### 获取墨子推演平台

 硬件环境：CPU i5及以上，显卡GTX 960 及以上（兼容cuda8.0以上的显卡驱动），内存8G。

 获取墨子推演平台，可以到华戍防务官网支持中心[https://www.hs-defense.com/col.jsp?id=105](https://www.hs-defense.com/col.jsp?id=105)下载, 其中包括windows个人版和linux版本，linux版本参考上边离线下载。

### moziai运行测试

参考码云上“墨子AI开发环境安装手册—anaconda” 安装完python开发环境后，进入到mozi_ai_sdk案例文件bt_test， 运行main_versus.py文件。在墨子推演平台能够可视化的看到代码运行的效果。 

其他案例与bt_test案例相似，且在每个案例文件夹中，带有readme介绍如何启动案例。


