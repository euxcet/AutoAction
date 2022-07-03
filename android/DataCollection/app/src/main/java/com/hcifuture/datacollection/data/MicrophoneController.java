package com.hcifuture.datacollection.data;

import android.content.Context;
import android.media.MediaRecorder;

import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;

/**
 * The controller for managing audio data.
 */
public class MicrophoneController {
    private Context mContext;
    // use system MediaRecorder to acquire and save data
    private MediaRecorder mMediaRecorder;
    private File saveFile;

    public MicrophoneController(Context context) {
        this.mContext = context;
    }

    public void start(File audioFile) {
        this.saveFile = audioFile;
        try {
            mMediaRecorder = new MediaRecorder();
            mMediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            mMediaRecorder.setAudioChannels(2);
            mMediaRecorder.setAudioSamplingRate(44100);
            mMediaRecorder.setAudioEncodingBitRate(16 * 44100);
            mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            mMediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            mMediaRecorder.setOutputFile(audioFile);
            mMediaRecorder.prepare();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void stop() {
        if (mMediaRecorder != null) {
            mMediaRecorder.stop();
            mMediaRecorder.release();
            mMediaRecorder = null;
            /*
            MediaScannerConnection.scanFile(mContext,
                    new String[] {file.getAbsolutePath(), audioFile.getAbsolutePath()},
                    null, null);
             */
        }
    }

    public void upload(String taskListId, String taskId, String subtaskId,
                       String recordId, long timestamp) {
        if (saveFile != null) {
            NetworkUtils.uploadRecordFile(mContext, saveFile,
                    TaskListBean.FILE_TYPE.AUDIO.ordinal(), taskListId, taskId, subtaskId,
                    recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                }
            });
        }
    }
}
