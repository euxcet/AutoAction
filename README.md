# AutoAction

## Deploy

下载[数据](https://cloud.tsinghua.edu.cn/f/9c334b37d3914b1a95d8/?dl=1)，保存到backend/server/data文件夹内。其中的classes.dex文件由classes.jar通过d8编译得到。

Mac

```bash
~/Library/Android/sdk/build-tools/{sdk-version}/d8 classes.jar
```

classes.jar由android studio生成，点击菜单栏Build下的Make Module 'DataCollection.contextactionlibrary'。编译得到的结果位于contextactionlibrary/build/outputs/aar/contextactionlibrary-debug.aar，将文件的后缀名改为zip，解压后文件夹内的classes.jar即为所需文件。

运行后端。

```bash
cd backend/server/src
python3 main.py
```

将datacollection.utils.NetworkUtils里的ROOT\_URL改成后端的ip。

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
