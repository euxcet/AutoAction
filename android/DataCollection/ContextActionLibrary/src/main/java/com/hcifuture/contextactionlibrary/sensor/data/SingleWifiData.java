package com.hcifuture.contextactionlibrary.sensor.data;

public class SingleWifiData extends Data {
    private String ssid;
    private String bssid;
    private String capabilities;
    private int level;
    private int frequency;
    private long timestamp;

    private int channelWidth;
    private int centerFreq0;
    private int centerFreq1;

    private boolean connected;

    public SingleWifiData() {
        this.ssid = "";
        this.bssid = "";
        this.capabilities = "";
        this.level = 0;
        this.frequency = 0;
        this.timestamp = 0;
        this.channelWidth = 0;
        this.centerFreq0 = 0;
        this.centerFreq1 = 0;
        this.connected = false;
    }

    public SingleWifiData(String ssid, String bssid,
                          String capabilities,
                          int level,
                          int frequency,
                          long timestamp,
                          int channelWidth,
                          int centerFreq0, int centerFreq1,
                          boolean connected) {
        setSsid(ssid);
        setBssid(bssid);
        setCapabilities(capabilities);
        this.level = level;
        this.frequency = frequency;
        this.timestamp = timestamp;
        this.channelWidth = channelWidth;
        this.centerFreq0 = centerFreq0;
        this.centerFreq1 = centerFreq1;
        this.connected = connected;
    }

    public int getCenterFreq0() {
        return centerFreq0;
    }

    public int getCenterFreq1() {
        return centerFreq1;
    }

    public int getChannelWidth() {
        return channelWidth;
    }

    public int getFrequency() {
        return frequency;
    }

    public int getLevel() {
        return level;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public String getBssid() {
        return bssid;
    }

    public String getCapabilities() {
        return capabilities;
    }

    public String getSsid() {
        return ssid;
    }

    public boolean getConnected() { return connected; }

    public void setBssid(String bssid) {
        if (bssid == null) {
            this.bssid = "";
        } else {
            this.bssid = bssid;
        }
    }

    public void setCapabilities(String capabilities) {
        if (capabilities == null) {
            this.capabilities = "";
        } else {
            this.capabilities = capabilities;
        }
    }

    public void setCenterFreq0(int centerFreq0) {
        this.centerFreq0 = centerFreq0;
    }

    public void setCenterFreq1(int centerFreq1) {
        this.centerFreq1 = centerFreq1;
    }

    public void setChannelWidth(int channelWidth) {
        this.channelWidth = channelWidth;
    }

    public void setFrequency(int frequency) {
        this.frequency = frequency;
    }

    public void setLevel(int level) {
        this.level = level;
    }

    public void setSsid(String ssid) {
        if (ssid == null) {
            this.ssid = "";
        } else {
            this.ssid = ssid;
        }
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public void setConnected(boolean connected) {
        this.connected = connected;
    }

    @Override
    public DataType dataType() {
        return DataType.SingleWifiData;
    }
}
