package com.signnav.live

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager

/**
 * Foreground service that keeps the stream, the servers and speech alive with the screen
 * off or the phone in a pocket (adapted from the DAT CameraAccess sample's StreamingService).
 */
class LiveService : Service() {
  companion object {
    private const val CHANNEL_ID = "signnav_live"
    private const val NOTIFICATION_ID = 1
    private const val ACTION_STOP = "com.signnav.live.STOP"

    fun start(context: Context) {
      context.startForegroundService(Intent(context, LiveService::class.java))
    }

    fun stop(context: Context) {
      // Through onStartCommand, so startForeground() always runs first (see the sample).
      context.startForegroundService(
          Intent(context, LiveService::class.java).apply { action = ACTION_STOP })
    }
  }

  private var wakeLock: PowerManager.WakeLock? = null

  override fun onBind(intent: Intent?): IBinder? = null

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    try {
      startForeground(NOTIFICATION_ID, notification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE)
    } catch (e: Exception) {
      AppLog.e("could not enter the foreground: ${e.message}", e)
      stopSelf()
      return START_NOT_STICKY
    }
    if (intent?.action == ACTION_STOP) {
      wakeLock?.let { if (it.isHeld) it.release() }
      wakeLock = null
      stopForeground(STOP_FOREGROUND_REMOVE)
      stopSelf()
      return START_NOT_STICKY
    }
    if (wakeLock == null) {
      wakeLock = (getSystemService(Context.POWER_SERVICE) as PowerManager)
          .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "SignNavLive::stream")
          .apply { acquire(3 * 60 * 60 * 1000L) }   // backstop: 3 h
    }
    return START_STICKY
  }

  override fun onDestroy() {
    wakeLock?.let { if (it.isHeld) it.release() }
    super.onDestroy()
  }

  private fun notification(): Notification {
    val nm = getSystemService(NotificationManager::class.java)
    nm.createNotificationChannel(
        NotificationChannel(CHANNEL_ID, "Live streaming", NotificationManager.IMPORTANCE_LOW))
    val open = PendingIntent.getActivity(
        this, 0,
        Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
    return Notification.Builder(this, CHANNEL_ID)
        .setContentTitle("Sign-nav live")
        .setContentText("Streaming the glasses to the PC")
        .setSmallIcon(android.R.drawable.ic_menu_compass)
        .setOngoing(true)
        .setContentIntent(open)
        .build()
  }
}
