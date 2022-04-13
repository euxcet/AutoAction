package com.hcifuture.contextactionlibrary.collect.trigger;

public class TriggerConfig {
    private int audioLength;
    public TriggerConfig() {
    }

    public int getAudioLength() {
        return audioLength;
    }

    public TriggerConfig setAudioLength(int audioLength) {
        this.audioLength = audioLength;
        return this;
    }
}
