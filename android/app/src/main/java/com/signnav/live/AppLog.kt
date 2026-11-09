package com.signnav.live

import android.util.Log

/** Logcat (tag "SignNavLive") plus the last lines for the status screen. */
object AppLog {
  private const val TAG = "SignNavLive"
  private const val KEEP = 40
  private val lines = ArrayDeque<String>()

  fun i(msg: String) {
    Log.i(TAG, msg)
    add(msg)
  }

  fun e(msg: String, t: Throwable? = null) {
    Log.e(TAG, msg, t)
    add("ERROR $msg")
  }

  private fun add(msg: String) {
    val stamp = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US).format(java.util.Date())
    synchronized(lines) {
      lines.addLast("$stamp $msg")
      while (lines.size > KEEP) lines.removeFirst()
    }
  }

  fun recent(n: Int = 15): List<String> = synchronized(lines) { lines.takeLast(n) }
}
