package com.hcifuture.datacollection.ui;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;

/**
 * Stores some information used in an alert dialog.
 */
public class NormalAlertDialog {
    private Context mContext;
    private AlertDialog.Builder builder;
    private AlertDialog dialog;

    public NormalAlertDialog(Context context, String title, String message) {
        mContext = context;
        builder = new AlertDialog.Builder(context);
        builder.setTitle(title);
        builder.setMessage(message);
    }

    public void setPositiveButton(String positiveText, DialogInterface.OnClickListener positiveListener) {
        builder.setPositiveButton(positiveText, positiveListener);
    }

    public void setNegativeButton(String negativeText, DialogInterface.OnClickListener negativeListener) {
        builder.setNegativeButton(negativeText, negativeListener);
    }

    public void create() {
        dialog = builder.create();
    }

    public void show() {
        if (dialog != null) {
            dialog.show();
        }
    }

    public void dismiss() {
        if (dialog != null) {
            dialog.dismiss();
        }
    }
}
