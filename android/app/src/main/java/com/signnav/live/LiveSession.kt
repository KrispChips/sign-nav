package com.signnav.live

import android.content.Context
import com.meta.wearable.dat.camera.types.VideoQuality
import java.net.Inet4Address
import java.net.NetworkInterface
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

/**
 * Everything that runs during a live test: the glasses session, the three servers the PC
 * connects to (docs/LIVE.md), and their stats. A process-wide singleton so it survives the
 * activity being recreated; [LiveService] keeps the process alive with the screen off.
 */
object LiveSession {
  const val MJPEG_PORT = 8080
  const val SPEECH_PORT = 9002
  const val YAW_PORT = 9003

  val frames = FrameStore()
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
  private var glasses: Glasses? = null
  private var mjpeg: MjpegServer? = null
  private var yawServer: YawServer? = null
  var speech: SpeechServer? = null
    private set

  @Volatile var running = false
    private set
  @Volatile var invertYaw = false
  @Volatile var yawDeg: Double? = null
    private set

  fun start(context: Context, quality: VideoQuality, streamFps: Int) {
    if (running) return
    running = true
    val app = context.applicationContext
    LiveService.start(app)
    frames.clear()
    if (speech == null) speech = SpeechServer(app, SPEECH_PORT)
    Thread {
      try {
        mjpeg = MjpegServer(frames, MJPEG_PORT).also { it.start() }
        speech?.start()
        yawServer = YawServer(YAW_PORT).also { it.start() }
        AppLog.i("servers up: :$MJPEG_PORT frames, :$SPEECH_PORT speech, :$YAW_PORT yaw")
      } catch (e: Exception) {
        AppLog.e("could not open the servers' ports: ${e.message}", e)
      }
    }.start()
    glasses = Glasses(scope, frames) { y ->
      val v = if (invertYaw) -y else y
      yawDeg = v
      yawServer?.publish(v)
    }.also { it.start(quality, streamFps) }
  }

  fun stop(context: Context) {
    if (!running) return
    running = false
    glasses?.stop()
    glasses = null
    mjpeg?.stop()
    mjpeg = null
    yawServer?.stop()
    yawServer = null
    speech?.stop() // keeps TTS alive for "test speech"; sockets closed
    yawDeg = null
    LiveService.stop(context.applicationContext)
    AppLog.i("stopped")
  }

  fun resetYaw() = glasses?.resetYaw()

  fun status(): Map<String, String> {
    val g = glasses
    return linkedMapOf(
        "session" to (g?.sessionState ?: "-"),
        "stream" to (g?.streamState ?: "-"),
        "motion" to (g?.motionState ?: "-"),
        "stream size" to (g?.streamSize ?: "-"),
        "PC frames" to "${mjpeg?.clients?.size ?: 0} client(s)",
        "PC speech" to "${speech?.clients?.size ?: 0} client(s)",
        "PC yaw" to "${yawServer?.clients?.size ?: 0} client(s)",
    )
  }

  /** IPv4 addresses (Wi-Fi, and Tailscale's 100.x on its VPN interface). */
  fun addresses(): List<String> =
      runCatching {
            NetworkInterface.getNetworkInterfaces().toList()
                .filter { it.isUp && !it.isLoopback }
                .flatMap { nif ->
                  nif.inetAddresses.toList().filterIsInstance<Inet4Address>()
                      .map { "${it.hostAddress} (${nif.name})" }
                }
          }
          .getOrDefault(emptyList())
}
