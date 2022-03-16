# AutoAction

## Deploy

### 下载必要文件

运行script文件夹下的download\_models和download\_ncnn\_sdk两个脚本。download\_ncnn\_sdk只需要执行一次，download\_models在模型文件出现修改时需要重新执行。

```bash
cd script
./download_models.sh
./download_ncnn_sdk.sh
```

### 编译contextactionlibrary包

安卓应用位于android/DataCollection中，运行需要动态加载contextactionlibrary库的dex二进制包。编译二进制包分为两步，首先生成classes.jar，再将classes.jar转化为classes.dex。classes.jar由android studio生成，在左侧的Project栏中选择ContextActionLibrary库，再点击菜单栏Build下的Make Module 'DataCollection.contextactionlibrary'。编译得到的结果位于contextactionlibrary/build/outputs/aar/contextactionlibrary-debug.aar。

执行DataCollection下的aar2dex.sh脚本可以将aar包中的classes.jar提取转化为classes.dex，并放到后端的指定位置供前端下载使用。

执行aar2dex.sh需要配置D8\_PATH这个环境变量，即d8所在的路径。在Mac下一般位于~/Library/Android/sdk/build-tools/{sdk-version}/d8。

```bash
export D8_PATH={YOUR_PATH} # Recommend adding this line to .bashrc
./aar2dex.sh
```

### 运行后端

```bash
cd backend/server/src
python3 main.py
```

### 网络配置

手机和后端所在的电脑应该在同一个局域网内，在DataCollection的local.properties中加入字段web.server，值为后端的ip。

示例：

```bash
web.server="http://192.168.31.186:60010"
```

### 测试情境和动作

## Roadmap

- [ ] 前端编辑动作类型
- [ ] 前端可视化样本
- [ ] 前端可视化训练结果
- [ ] 后端维护数据
- [ ] 后端训练自动化流程
- [ ] 自动生成apk
- [ ] Docker

## Reference

[Build](http://developer.android.com/studio/build/building-cmdline)
