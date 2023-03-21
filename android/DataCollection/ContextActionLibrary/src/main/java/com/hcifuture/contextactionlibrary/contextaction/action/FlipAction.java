package com.hcifuture.contextactionlibrary.contextaction.action;


import static android.os.VibrationEffect.createOneShot;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.StrictMath.abs;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;

import com.hcifuture.contextactionlibrary.sensor.collector.sync.LogCollector;
import com.hcifuture.contextactionlibrary.sensor.data.NonIMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;
import com.hcifuture.contextactionlibrary.status.Heart;
import com.hcifuture.shared.communicate.config.ActionConfig;
import com.hcifuture.shared.communicate.listener.ActionListener;
import com.hcifuture.shared.communicate.listener.RequestListener;
import com.hcifuture.shared.communicate.result.ActionResult;

import java.util.Deque;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class FlipAction extends BaseAction {

    //判断自然动作所需的变量
    boolean flag1,flag2; //翻转的第1步和第2步的角度判断
    boolean gyro_flag1,gyro_flag2; //角速度对flip进行判断
    long flag1_id,gyro_id1,gyro_id2,flag2_id; //两次稳态的时间
    double pitch;
    float gx,gy,gz; //陀螺仪在x,y,z上的
    boolean postive; //第一次是否为正数
    float max_gy,max_gx; //最大的gy的值，避免出现gx过大的情况
    long gx_id; //最大gx的下标
    private float[] values, r, gravity, geomagnetic;
    boolean success_flag;
    long value_timestamp;

    //避免误触的一些参数
    //1. gx必须满足先大后小
    float min_gx_check,max_gx_check;
    long min_gx_id,max_gx_id;
    float min_gy_check,max_gy_check;
    long min_gy_id,max_gy_id;
    long time_interval = 2 * 10000; //统计这段时间内的最大和最小值
    //2. 判断是否连续增大后减小


    //3. 判断识别成功后是否在一段时间内稳定了
    boolean check_stable_flag;
    long check_stable_id;
    boolean stable_flag1_gx,stable_flag1_gy; //是否在一定时间内减为0了
    int stable_thres1 = 3000;
    int stable_thres2 = 500;
    int stable_num = 30, stay_num = 120;
    long stable_min_id_gx,stable_min_id_gy; //稳定的第一个stable id

    boolean stable_long_gx, stable_long_gy; //稳定后维持的时间

//    Deque all_gx = new LinkedList();
//    Deque all_gy = new LinkedList();

    Deque<Float> gx_before = new LinkedList<Float>();
    Deque<Float> gy_before = new LinkedList<Float>();
    Deque<Long> time = new LinkedList<Long>();

    private Vibrator vibrator;

    private String TAG = "FLIP";

    private void vibrate(){
        VibrationEffect vibe = null;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            vibe = createOneShot(100,10);
            vibrator.vibrate(vibe);
        }
    }

    public FlipAction(Context context, ActionConfig config, RequestListener requestListener, List<ActionListener> actionListener, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList, LogCollector FlipLogCollector) {
        super(context, config, requestListener, actionListener, scheduledExecutorService, futureList);
        logCollector = FlipLogCollector;
        reset();
    }

    //对变量进行初始化
    private void reset(){
        values = new float[3];//用来保存最终的结果
        gravity = new float[3];//用来保存加速度传感器的值
        r = new float[9];//
        geomagnetic = new float[3];//用来保存地磁传感器的值
        flag1 = false; flag2 = false;
        gyro_flag1 = false; gyro_flag2 = false;
        gyro_id1 = -1; gyro_id2 = -1;
        flag1_id = -1; flag2_id = -1;
        max_gy = 0; max_gx = 0;
        gx_id = -1;
        check_stable_flag = false;
        max_gx_check = -1; min_gx_check = 1000;
        max_gy_check = -1; min_gy_check = 1000;
        gx_before.clear();
        gy_before.clear();
        time.clear();
        stable_long_gy = false; stable_long_gx = false;
        stable_flag1_gx = false; stable_flag1_gy = false;
        stable_min_id_gx = -1; stable_min_id_gy = -1;
        check_stable_id = -1;
        min_gx_id = -1; max_gx_id = -1;
        min_gy_id = -1; max_gy_id = -1;
        success_flag = false;
//        logCollector = new LogCollector(mContext, CollectorManager.CollectorType.Log, scheduledExecutorService, futureList, TAG, 800);
    }

    private boolean check_gy(){
        if(max_gx>max_gy+100 && gx_id> Long.min(gyro_id1,flag1_id) && gx_id< Long.max(gyro_id2,flag2_id)){
            Log.i(TAG,"getnum"+"-------gy不满足条件");
            return false;
        }
        Log.i(TAG,"getnum"+"-------gy满足条件");
        return true;
    }

    private void double_check(){
        Log.i(TAG,"getnum"+"------开始进行二次检查------"+System.currentTimeMillis());
        //检查是否是先大后小
        if(max_gx_id>min_gx_id){ //假如是先小后大， 则说明gx不对
            Log.i(TAG,"getnum:"+"-------gx不满足条件"+max_gx_id+" "+max_gx_check+" "+min_gx_id+" "+min_gx_check);
        }
        else{
            Log.i(TAG,"getnum:"+"-------gx满足条件"+max_gx_id+" "+min_gx_id);
        }
//        if(max_gx_id>min_gx_id || !check_gy() || !analysis()){
        if(!check_gy() || !analysis()){
            reset();
            Log.i(TAG,"getnum"+"========二次检查不满足条件------"+System.currentTimeMillis());
            return;
        }

        check_stable_flag = true;
        check_stable_id = System.currentTimeMillis();
        Log.i(TAG,"getnum:"+"-------二次检查全部满足条件---------"+max_gx_id+" "+min_gx_id);
        return;
    }

    //分析gx和gy 是否满足！！连续！！上升和下降的条件
    private boolean analysis(){
        //遍历时间 然后根据时间给迭代器的位置给 get_num, 从而获取连续上升、下降的数量
        Iterator<Long> iterator = time.iterator();
        Iterator<Float> gx_iterator = gx_before.iterator();
        Iterator<Float> gy_iterator = gy_before.iterator();

        Long t;

        Log.i(TAG,"getnum"+"===size=== "+ gx_before.size()+" "+gy_before.size()+" "+time.size());
        Log.i(TAG,"getnum"+"actual_time:"+max_gx_id +" "+min_gx_id+" "+max_gy_id+" "+min_gy_id);
        Log.i(TAG,"getnum"+"time_interval:"+abs(max_gx_id-min_gx_id)+" "+abs(max_gy_id-min_gy_id));


        // 得到迭代器之后，就得break掉，要是一直遍历 就会到最后一个 迭代器就不对了！！
        //T 所以要分开找迭代器 即找到了之后就break掉 ！！

        //只需要找到第一个的的 gx 和 t 对应的 iterator
        //t用来遍历查看后一个 极值是多少
        int thres_x = (int) (max_gx_check-min_gx_check); //最大值减最小值
        int thres_y = (int) (max_gy_check-min_gy_check); //最大值减最小值
        float st1_num_x = -1; // 第1阶段的连续的x的数量
        float r2_x = -1; //第2阶段的比例
//        int st2_num_x = -1; // 第2阶段的连续的x的数量
        float st1_num_y = -1; // 第1阶段的连续的x的数量
        float r2_y = -1;
//        int st2_num_y = -1; // 第2阶段的连续的x的数量

        float[] st1_x = new float[3]; //第一阶段x的值
        float[] st2_x = new float[3]; //第二阶段x的值


        float[] st1_y = new float[3]; //第二阶段x的值
        float[] st2_y = new float[3]; //第二阶段x的值

        //查找gx的最大值
        while (iterator.hasNext() && gx_iterator.hasNext()){
            t = iterator.next();
            gx_iterator.next();
//            gx_num ++;
            if(max_gx_id < min_gx_id){
                //去寻找max_gx_id
                if (t >= max_gx_id) {
                    st1_x = get_num(gx_before.iterator(),time.iterator(),thres_x,true,max_gx_id,true);
                    st2_x = get_num(gx_iterator,iterator,thres_x,false,min_gx_id,false);
//                    st1_num_x = get_num(gx_before.iterator(),time.iterator(),thres_x,true,max_gx_id,true);
//                    r2_x = get_num(gx_iterator,iterator,thres_x,false,min_gx_id,false);
                    Log.i(TAG,"getnum"+"situation1");
                    break;
                }
            }
            else{
                if(t == min_gx_id){
                    st1_x = get_num(gx_before.iterator(),time.iterator(),thres_x,false,min_gx_id,true);
                    st2_x = get_num(gx_iterator,iterator,thres_x,true,max_gx_id,false);
//                    st1_num_x = get_num(gx_before.iterator(),time.iterator(),thres_x,false,min_gx_id,true);
//                    r2_x = get_num(gx_iterator,iterator,thres_x,true,max_gx_id,false);
                    Log.i(TAG,"getnum"+"situation2");
                    break;
                }
            }
        }

        if(!(st1_x[0]>30 && st2_x[0]>0.75)) {
            Log.i(TAG,"getnum"+"===x不满足一般条件 "+st1_x[0]+" "+st2_x[0]);
            if(!(st1_x[1]<30 && st1_x[0]>st1_x[1]*0.6 && st2_x[0]>0.75)){
                Log.i(TAG,"getnum"+"===x不满足过快条件 "+st1_x[1]+" "+st1_x[0]+" "+st2_x[0]);
                if(!(st2_x[1]>400 && st1_x[1]>400 && st1_x[0]>30 && st2_x[0]>0.6 && st2_x[2]==3)){
                    Log.i(TAG,"getnum"+"===x不满足过慢条件"+st2_x[1]+" "+st1_x[1]+" "+st1_x[0]+" "+st2_x[0]+" "+st2_x[2]);
                    return false;
                }
            }
        }

        iterator = time.iterator();  //重新赋值到开始
        //查找gy的极值
        boolean first_time = true;
        while (iterator.hasNext() && gy_iterator.hasNext()){
            t = iterator.next();
            gy_iterator.next();
            if(first_time){
                Log.i(TAG,"getnum"+"第一次的时间："+t);
                first_time = false;
            }
//            gy_num ++;
            if(max_gy_id<min_gy_id){
                //去寻找max_gy_id
                if (t == max_gy_id) {
                    st1_y = get_num(gy_before.iterator(),time.iterator(),thres_y,true,max_gy_id,true);
                    st2_y = get_num(gy_iterator,iterator,thres_y,false,min_gy_id,false);
//                    st1_num_y = get_num(gy_before.iterator(),time.iterator(),thres_y,true,max_gy_id,true);
//                    r2_y = get_num(gy_iterator,iterator,thres_y,false,min_gy_id,false);
                    Log.i(TAG,"getnum"+"situation3");
                    break;
                }
            }
            else{
                if(t == min_gy_id){
                    st1_y = get_num(gy_before.iterator(),time.iterator(),thres_y,false,min_gy_id,true);
                    st2_y = get_num(gy_iterator,iterator,thres_y,true,max_gy_id,false);
//                    st1_num_y = get_num(gy_before.iterator(),time.iterator(),thres_y,false,min_gy_id,true);
//                    r2_y = get_num(gy_iterator,iterator,thres_y,true,max_gy_id,false);
                    Log.i(TAG,"getnum"+"situation4");
                    break;
                }
            }
        }

        double thres_st2_y = 0.75;
        if(st2_y[2]==3){
            thres_st2_y = 0.7;
        }

        if(!(st1_y[0]>25 && st2_y[0]>thres_st2_y)) {
            Log.i(TAG,"getnum"+"===y不满足一般条件 "+st1_y[0]+" "+st2_y[0]);
            if(!(st1_y[1]<30 && st1_y[0]>st1_y[1]*0.7 && st2_y[0]>thres_st2_y)){
                Log.i(TAG,"getnum"+"===y不满足过快条件 "+st1_y[1]+" "+st1_y[0]+" "+st2_y[0]);
                if(!(st2_y[1]>400 && st1_y[1]>400 && st1_y[0]>30 && st2_y[0]>thres_st2_y-0.05 && st2_y[2]==3)){
                    Log.i(TAG,"getnum"+"===y不满足过慢条件"+st2_y[1]+" "+st1_y[1]+" "+st1_y[0]+" "+st2_y[0]+" "+st2_y[2]);
                    return false;
                }
            }
        }

        Log.i(TAG,"getnum"+"stage:"+st1_num_x+" "+r2_x+" "+st1_num_y+" "+r2_y);

        //1。 正常情况， 前一个用数量 后一个用比例 （因为前一个是小波动，后一个是连续的大波动，用比例比较好！！！）
        //2。 很慢的情况，比例会不对，但是基本是连续的数目，改小所需的比例！
        //3。 很快的情况，数量会不对，可以改用比例？
//        if (!(st1_num_x > 30 && r2_x > 0.8 && st1_num_y > 20 && r2_y > 0.75)) {
//            Log.i(TAG,"getnum"+"-------analysis不满足条件："+st1_num_x+" "+r2_x+" "+st1_num_y+" "+r2_y);
//            return false;
//        }
        Log.i(TAG,"getnum"+"-------analysis满足条件");
        return true;
    }

    //order代表是正序遍历还是反序
    //st1代表是stage1还是stage2，stage2的话返回比例就好了
    private float[] get_num(Iterator<Float> it1, Iterator<Long> it, int thres, boolean order, long time, boolean st1){
        float decrease_num = 0;
        float increase_num = 0;

        Float before_value = null;
        Float now_value;
        int tolerance_increase = 3;
        int tolerance_decrease = 3;
        float cnt = 0; //两个间隔间的数量
        //看连续增加的数量，感觉看时间应该更合理一点。但之前python用的就不是时间间隔。
        //直接用个数来看？感觉迭代器不是很准？？

        //当总数增加时，tolerance的数量也应该增加

        //从it开始，到it的值为time
        while(it.hasNext() && it1.hasNext()) {
            long t = it.next();
            cnt++;
            if (t == time) {
                Log.i("getnum:","-break_time:"+cnt);
                break;
            }
            if (before_value == null) {
                if (it1.hasNext()) {
                    before_value = it1.next();
                    continue;
                } else {
                    Log.i(TAG,"getnum"+ "初始化就失败了");
                    break;
                }
            }
            now_value = it1.next();
//                Log.i(TAG,"getnum"+ String.valueOf(now_value));
            if (now_value > before_value) {
                increase_num += 1;
            } else {
                if (now_value > before_value - 0.04 * thres) {
                    increase_num += 0;
                } else if (now_value > before_value - 0.1 * thres) {
                    if (tolerance_increase > 0) {
                        tolerance_increase -= 1;
                    } else {
                        Log.i(TAG,"getnum"+ "increase 没有容忍度了");
                        increase_num = 1; //重新开始计数
                        tolerance_increase = 3;
                    }
                } else {
                    Log.i(TAG,"getnum"+ "increase 突变地太离谱了"+String.valueOf((float)((now_value-before_value)/thres)));
                    increase_num = 1; //重新开始计数
                    tolerance_increase = 3;
                }
            }
            if (before_value > now_value) {
                decrease_num += 1;
            } else {
                if (before_value > now_value - 0.02 * thres) {
                    decrease_num += 0;
                } else if (before_value > now_value - 0.05 * thres) {
                    if (tolerance_decrease > 0) {
                        tolerance_decrease -= 1;
                    } else {
                        Log.i(TAG,"getnum"+ "decrease 没有容忍度了");
                        decrease_num = 1; //重新开始计数
                        tolerance_decrease = 3;
                    }
                } else {
                    Log.i(TAG,"getnum"+ "decrease  突变地太离谱了");
                    decrease_num = 1; //重新开始计数
                    tolerance_decrease = 3;
                }
            }
            before_value = now_value;
        }
        Log.i(TAG,"getnum"+"all_num:"+increase_num+" "+decrease_num+" "+cnt+" "+tolerance_decrease+" "+tolerance_increase);

        if(order)
            if(st1)
                return new float[]{increase_num,cnt,tolerance_increase};
            else {
                float temp = increase_num/cnt;
                return new float[]{temp,cnt,tolerance_increase};
//                return temp;
            }
        else
        if(st1)
            return new float[]{decrease_num,cnt,tolerance_decrease};
        else {
            float temp = decrease_num/cnt;
            return new float[]{temp,cnt,tolerance_decrease};
        }
    }

    private void check_stable(){
        if(!stable_flag1_gx) { //先减小到一定的值
            if (System.currentTimeMillis() - check_stable_id > stable_thres1) {
                Log.i(TAG,"getnum"+"gx太久没成功");
                reset();
            }
            else if (abs(gx)<stable_num){
                stable_min_id_gx = System.currentTimeMillis();
                stable_flag1_gx = true;
                Log.i("getnum:","gx足够小了");
            }
        }
        else{
            if(System.currentTimeMillis()-stable_min_id_gx>stable_thres2){
                stable_long_gx = true;
                Log.i(TAG,"getnum"+"gx足够长");
            }
            if(abs(gx)>=stay_num){
                Log.i(TAG,"getnum"+"gx有剧烈抖动"+gx);
                stable_flag1_gx = false;
//                reset();
            }
        }
        if(!stable_flag1_gy) {
            if (System.currentTimeMillis() - check_stable_id > stable_thres1) {
                Log.i("getnum:","gy太久没成功");
                reset(); //时间太长没成功 重置
            }
            else if (abs(gy)<stable_num){
                Log.i("getnum:","gy足够小了");
                stable_min_id_gy = System.currentTimeMillis();
                stable_flag1_gy = true;
            }
        }
        else{
            if(System.currentTimeMillis()-stable_min_id_gy>stable_thres2){
                Log.i("getnum:","gy足够长");
                stable_long_gy = true;
            }
            if(abs(gy)>=stay_num){
                Log.i("getnum:","gy有剧烈抖动"+gy);
                stable_flag1_gy = false;
//                reset();
            }
        }
        if(stable_long_gx && stable_long_gy){
            Success();
        }
    }

    private void Success(){
        //判断是否在一段时间内保持稳定了
        //识别成功时，注册listener
        success_flag = true;
        vibrate();
//        Log.i(TAG,"===========翻转识别成功============");
    }

    @Override
    public synchronized void start() {
        isStarted = true;
        vibrator=(Vibrator)mContext.getSystemService(Context.VIBRATOR_SERVICE);
        reset();
    }

    @Override
    public synchronized void stop() {
        isStarted = false;
    }

    @Override
    public void onIMUSensorEvent(SingleIMUData data) {
        Heart.getInstance().newActionAliveEvent(getConfig().getAction(), data.getTimestamp());
        int type = data.getType();
        switch (type) {
            case Sensor.TYPE_ACCELEROMETER:
                gravity[0] = data.getValues().get(0);
                gravity[1] = data.getValues().get(1);
                gravity[2] = data.getValues().get(2);
                getValue(); //更新方位角
                if(abs(gravity[2])>55){
                    Log.i(TAG,"角速度过大"+abs(gravity[2]));
                    Log.i(TAG,"getnum"+"角速度过大"+abs(gravity[2]));
                    reset();
                }
                break;
            case Sensor.TYPE_MAGNETIC_FIELD:
                geomagnetic[0] = data.getValues().get(0);
                geomagnetic[1] = data.getValues().get(1);
                geomagnetic[2] = data.getValues().get(2);
                getValue();
                break;
            case Sensor.TYPE_GYROSCOPE:
                //需要将弧度转为角度
                gx = (float) Math.toDegrees(data.getValues().get(0));
                gy = (float) Math.toDegrees(data.getValues().get(1));
                gz = (float) Math.toDegrees(data.getValues().get(2));

                long time_now = System.currentTimeMillis();
                long time_thres = 20000; //收集2s的数据



                if (check_stable_flag) { //已经识别成功了 检查稳定性
                    check_stable();
                } else {
                    //把超出时间的数据给弄出去
                    while(!time.isEmpty()){
                        //以前的时间间隔太远了
                        if(abs(time.getFirst()-time_now)>time_thres) {
//                            Log.i("remove:", String.valueOf(time.getFirst())+" "+time_now);
                            time.removeFirst();
                            gx_before.removeFirst();
                            gy_before.removeFirst();
                        }
                        else{
                            break;
                        }
                    }
                    gx_before.add(gx); gy_before.add(gy); time.add(time_now);
                    //当前时间间隔下找最大的gx
                    if (gx > max_gx_check || time_now - max_gx_id > time_interval) {
                        max_gx_check = gx;
                        max_gx_id = time_now;
                    }
                    //最小的gx
                    if (gx < min_gx_check || time_now - min_gx_id > time_interval) {
                        min_gx_check = gx;
                        min_gx_id = time_now;
                    }
                    //当前时间间隔下 找最大的gy
                    if (gy > max_gy_check || time_now - max_gy_id > time_interval) {
                        max_gy_check = gy;
                        max_gy_id = time_now;
                    }
                    //最小的gx
                    if (gy < min_gy_check || time_now - min_gy_id > time_interval) {
                        min_gy_check = gy;
                        min_gy_id = time_now;
                    }

                    if (abs(gy) > max_gy + 100) {
                        max_gx = abs(gx);
                        gx_id = time_now;
                    }

                    if (abs(gy) > 160) {
                        max_gy = Float.max(max_gy, abs(gy));
                        if (!gyro_flag1) {
                            //第一次开始的地方 重新计算？
                            postive = gy > 0 ? true : false;
                            gyro_flag1 = true;
                            gyro_id1 = System.currentTimeMillis();
                            Log.i(TAG, "角速度1:第一次大于200:" + "gy: " + gy + " gx: " + gx + "gz:" + gz + "ID：" + gyro_id1);
                        } else if (!gyro_flag2) {
                            Log.i(TAG, "角速度2:第二次大于200" + "gy: " + gy + " gx: " + gx + "gz:" + gz + "ID：" + gyro_id1);
                            if (postive) {
                                if (gy < 0 && flag1) {
                                    if (System.currentTimeMillis() - gyro_id1 < 20000) {
                                        gyro_flag2 = true;
                                        gyro_id2 = System.currentTimeMillis();
                                        Log.i(TAG, "角速度2:第二次为正" + gyro_id2);
                                        Log.i(TAG, "-------------------角速度ok");
                                        if (flag2 && (gyro_id2 - flag2_id) < 100) {
                                            Log.i(TAG, "角速度3:角度也满足条件！");
                                            double_check(); //进行二次检查
//                                        Success();
                                        }
                                    }
                                } else {
                                    Log.i(TAG, "角速度4:距离上次200的时间太长");
                                    gyro_id1 = System.currentTimeMillis();
                                    postive = false;
                                }
                            } else {
                                Log.i(TAG, "角速度5:更新上一次为正");
                                gyro_id1 = System.currentTimeMillis();
                                postive = true;
                            }
                        } else {
                            if (gy > 0) {
                                if (System.currentTimeMillis() - gyro_id1 < 20000) {
                                    gyro_flag2 = true;
                                    gyro_id2 = System.currentTimeMillis();
                                    Log.i(TAG, "角速度2:第二次为负");
                                    Log.i(TAG, "-------------------角速度ok");
                                    if (flag2 && (gyro_id2 - flag2_id) < 3000) {
                                        Log.i(TAG, "角速度3:角度也满足条件！");
                                        double_check();
                                        //  Success();

                                    }
                                } else {
                                    Log.i(TAG, "角速度4:距离上次200的时间太长");
                                    gyro_id1 = System.currentTimeMillis();
                                    postive = true;
                                }
                            } else {
                                Log.i(TAG, "角速度5:更新上一次为负");
                                gyro_id1 = System.currentTimeMillis();
                                postive = false;
                            }
                        }
                    } else {
                        if (((postive && gy < 0) || (!postive && gy > 0)) && (System.currentTimeMillis() - gyro_id1 < 30000)) {
                            gyro_id2 = System.currentTimeMillis();
//                            Log.i(TAG, "角速度7：更新gyro2" + gy + "id2:" + gyro_id2);
                        } else {
                            //假如gyro_flag1和2 都满足了，就要考虑更新gyro_flag1和2
                            //可能是用户第一次没识别出来，要通过第二次来识别，假如这时不更新，只能等到时长太长再更新了！
                            postive = gy > 0 ? true : false;
                            gyro_flag1 = true;
                            gyro_id1 = System.currentTimeMillis();
//                            Log.i(TAG, "角速度6:第三次大于200" + "postive:" + postive + "gy:" + gy);
                            gyro_flag2 = false;
                        }
                    }
                }
                if (gyro_flag1 && gyro_flag2) {
                    if (System.currentTimeMillis() - gyro_id2 > 10000) {
                        Log.i(TAG, "角速度6:时间太长了 角速度的两个flag更新为false");
                        gyro_flag1 = false;
                        gyro_flag2 = false;
                    }
                }
                break;
            default:
                break;
        }

    }

    @Override
    public void onNonIMUSensorEvent(NonIMUData data) {

    }

    @Override
    public void onExternalEvent(Bundle bundle) {

    }



    //获取方位角数据
    public void getValue() {
        // r从这里返回
        if(gravity!=null && geomagnetic!=null){
            if(SensorManager.getRotationMatrix(r, null, gravity, geomagnetic)) {
                //values从这里返回
                SensorManager.getOrientation(r, values);
                //提取数据
                double azimuth = Math.toDegrees(values[0]);
                if (azimuth < 0) {
                    azimuth = azimuth + 360;
                }
                pitch = Math.toDegrees(values[1]);
                double roll = Math.toDegrees(values[2]);
//                Log.i(TAG,"roll: "+Math.floor(roll)+"ptich: "+Math.floor(pitch));

                if(pitch>45){
                    reset();
                    Log.i("getnum:","pitch太大了，重置"+pitch);
                    Log.i("FLIP:","pitch太大了，重置"+pitch);
                }

                if(!flag1) {
                    if(gyro_flag1) {
                        if (pitch > -60 && pitch < 35) { //保证是平面 , <35是因为当速度很快时 就会让pitch很大
                            //TODO: 快的时候，很多次 roll不在阈值范围里面
//                        Log.i(TAG,"角度0:满足pitch条件" + roll);
                            if ((roll < -100 && roll > -180) || (roll > 100 && roll < 180)) { //保证翻转了180度
                                flag1 = true;
                                flag1_id = System.currentTimeMillis();
                                Log.i(TAG, "角度1:翻转到下面了"+(flag1_id-gyro_id1)+" "+flag1_id+" "+gyro_id1);
                            }
//                        else{
//                            Log.i(TAG,"roll的角度不满足:"+roll);
//                        }
                        }
                    }
                }
                //flag
                else{
//                    if(!(pitch>-65 && pitch<35)) {
//                        flag1 = false;
//                        Log.i(TAG,"角度2:太竖直了 失败:"+pitch);
////                        flip_cnt = 0;
//                    }
                    if(roll<30 && roll>-30 && pitch > -60 && pitch < 15){
                        flag1 = false;
                        Log.i(TAG,"角度3:返回平面了");
                        if(System.currentTimeMillis() - flag1_id<20000) {
                            Log.i(TAG,"角度4:时间满足");
                            //检查gyro满足条件没 , 只要gyro_id2满足条件就行， 其实不用检查gyro_flag2
                            if((System.currentTimeMillis()-gyro_id2)<10000) {
                                Log.i(TAG,"角度5:角速度也满足条件啦！");
                                double_check();
//                                      Success();
//                                }
                            }
                            else{
                                Log.i(TAG,"角度6:角速度未满足条件"+(System.currentTimeMillis()-gyro_id2)+"f1:"+gyro_flag1+"f2:"+gyro_flag2);
                                Log.i(TAG,"-------------------角度ok");
                                flag2 = true;
                                flag2_id = System.currentTimeMillis();
                            }
                        }
                    }
//                    else{
//                        Log.i(TAG,"roll: "+Math.floor(roll)+"ptich: "+Math.floor(pitch));
//                    }
                }
            }
        }
    }


    /*
    @Override
    public void onProximitySensorChanged(SensorEvent event) {
    }
     */

    @Override
    public synchronized void getAction() {
        if (!isStarted)
            return;
        if (success_flag) {
            Log.i(TAG,"识别成功了");
            reset();
            for (ActionListener listener: actionListener) {
                listener.onAction(new ActionResult(TAG));
            }
        }
    }

    @Override
    public String getName() {
        return "FlipAction";
    }

}
