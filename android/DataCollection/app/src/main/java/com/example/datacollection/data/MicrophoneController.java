package com.example.datacollection.data;

import android.content.Context;
import android.media.MediaRecorder;
import android.media.MediaScannerConnection;

import com.example.datacollection.TaskList;
import com.example.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;

public class MicrophoneController {
    private Context mContext;
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

    public void upload(String taskListId, String taskId, String subtaskId, String recordId, long timestamp) {
        if (saveFile != null) {
            NetworkUtils.uploadRecordFile(mContext, saveFile, TaskList.FILE_TYPE.AUDIO.ordinal(), taskListId, taskId, subtaskId, recordId, timestamp, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                }
            });
        }
    }
}
