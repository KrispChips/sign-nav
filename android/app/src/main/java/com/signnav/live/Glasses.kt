package com.signnav.live

import com.meta.wearable.dat.camera.Camera
import com.meta.wearable.dat.camera.addCamera
import com.meta.wearable.dat.camera.types.StreamConfiguration
import com.meta.wearable.dat.camera.types.VideoFrame
import com.meta.wearable.dat.camera.types.VideoQuality
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.AutoDeviceSelector
import com.meta.wearable.dat.core.session.DeviceSession
import com.meta.wearable.dat.core.session.DeviceSessionState
import com.meta.wearable.dat.motion.Motion
import com.meta.wearable.dat.motion.addMotion
import com.meta.wearable.dat.motion.types.MotionConfiguration
import com.meta.wearable.dat.motion.types.MotionSample
import com.meta.wearable.dat.motion.types.MotionSamplingRate
import com.meta.wearable.dat.motion.types.MotionSource
import kotlin.math.PI
import kotlin.math.sqrt
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

/**
 * One DAT device session with two capabilities (Meta DAT 1.0.1, patterns from the
 * CameraAccess and BirdSpotter samples):
 * - camera: a compressed HEVC stream (keeps flowing with the phone locked), decoded by
 *   [HevcDecoder] into [FrameStore];
 * - motion (experimental): IMU samples turned into head yaw for [onYaw]. If the glasses or
 *   the SDK refuse it, streaming carries on without yaw.
 */
class Glasses(
    private val scope: CoroutineScope,
    private val frames: FrameStore,
    private val onYaw: (Double) -> Unit,
) {
  @Volatile var sessionState = "-"
  @Volatile var streamState = "-"
  @Volatile var motionState = "-"
  @Volatile var streamSize = "-"

  private var session: DeviceSession? = null
  private var camera: Camera? = null
  private var motion: Motion? = null
  private var decoder: HevcDecoder? = null
  private val jobs = mutableListOf<Job>()
  private val yaw = YawEstimator()
  private var quality = VideoQuality.HIGH
  private var fps = 24

  fun start(quality: VideoQuality, fps: Int) {
    if (session != null) return
    this.quality = quality
    this.fps = fps
    Wearables.createSession(AutoDeviceSelector())
        .onSuccess { s ->
          session = s
          jobs += scope.launch {
            s.errors.collect { e -> AppLog.e("session: ${e.description}") }
          }
          jobs += scope.launch {
            s.state.collect { st ->
              sessionState = st.toString()
              AppLog.i("session $st")
              // Capabilities can only be added once the session has STARTED.
              if (st == DeviceSessionState.STARTED) {
                if (camera == null) attachCamera(s)
                if (motion == null) attachMotion(s)
              }
            }
          }
          s.start() // fire-and-forget; progress arrives through s.state / s.errors
        }
        .onFailure { e, _ -> AppLog.e("could not create a session: ${e.description}") }
  }

  private fun attachCamera(s: DeviceSession) {
    s.addCamera(
            StreamConfiguration(videoQuality = quality, frameRate = fps, compressVideo = true),
        )
        .onSuccess { cam ->
          camera = cam
          val stream = cam.stream
          jobs += scope.launch(Dispatchers.Default) { stream.videoStream.collect { onFrame(it) } }
          jobs += scope.launch {
            stream.state.collect { st ->
              streamState = st.toString()
              AppLog.i("stream $st")
            }
          }
          jobs += scope.launch {
            stream.errorStream.collect { e -> AppLog.e("stream: ${e.description}") }
          }
          stream.start().onFailure { e, _ -> AppLog.e("stream would not start: ${e.description}") }
          AppLog.i("camera added: $quality @ $fps fps (compressed HEVC)")
        }
        .onFailure { e, _ -> AppLog.e("camera would not attach: ${e.description}") }
  }

  private fun onFrame(frame: VideoFrame) {
    if (!frame.isCompressed) return
    val buf = frame.buffer
    val bytes = ByteArray(buf.remaining())
    val pos = buf.position()
    buf.get(bytes)
    buf.position(pos)
    val dec = decoder ?: HevcDecoder(frames::offer).also {
      it.start(frame.width, frame.height)
      decoder = it
      streamSize = "${frame.width}x${frame.height}"
      AppLog.i("first frame: ${frame.width}x${frame.height}")
    }
    dec.decodeFrame(bytes, frame.presentationTimeUs)
  }

  private fun attachMotion(s: DeviceSession) {
    s.addMotion(MotionConfiguration(samplingRate = MotionSamplingRate.HZ_30))
        .onSuccess { m ->
          motion = m
          jobs += scope.launch(Dispatchers.IO) { m.samples.collect { onMotion(it) } }
          jobs += scope.launch {
            m.state.collect { st ->
              motionState = st.toString()
              AppLog.i("motion $st")
            }
          }
          jobs += scope.launch {
            m.errors.collect { e -> if (e != null) AppLog.e("motion: ${e.description}") }
          }
          m.start()
        }
        .onFailure { e, _ ->
          motionState = "unavailable"
          AppLog.e("motion unavailable (no yaw; streaming continues): ${e.description}")
        }
  }

  private var motionSourceLogged = false

  private fun onMotion(sample: MotionSample) {
    if (!motionSourceLogged) {
      motionSourceLogged = true
      AppLog.i("motion: first sample from ${sample.source}, gyro " +
          (if (sample.gyroscope == null) "none" else "yes") + ", accel " +
          (if (sample.accelerometer == null) "none" else "yes"))
    }
    if (sample.source != MotionSource.GLASSES) return
    val a = sample.accelerometer
    val g = sample.gyroscope
    val y = yaw.update(
        a?.let { doubleArrayOf(it.x.toDouble(), it.y.toDouble(), it.z.toDouble()) },
        g?.let { doubleArrayOf(it.x.toDouble(), it.y.toDouble(), it.z.toDouble()) },
        sample.timestampNs,
    ) ?: return
    onYaw(y)
  }

  fun resetYaw() = yaw.reset()

  fun stop() {
    jobs.forEach { it.cancel() }
    jobs.clear()
    runCatching { motion?.stop() }
    runCatching { camera?.stop() }
    runCatching { session?.stop() }
    synchronized(this) {
      decoder?.stop()
      decoder = null
    }
    motion = null
    camera = null
    session = null
    sessionState = "-"
    streamState = "-"
    motionState = "-"
  }
}

/**
 * Head yaw from the glasses' IMU without knowing its axis convention (the SDK doesn't
 * document it): gravity comes from the low-passed accelerometer, and the gyroscope's rate
 * about that vertical axis is integrated. At rest an accelerometer reads +g pointing UP, and
 * a positive rate about up is a turn to the LEFT (right-hand rule), so yaw = -integral.
 * It drifts slowly; sign-nav only uses it relative to its own smoothed heading, which follows.
 * If turning right makes the number go down, tick "invert yaw" (or --gyro-sign -1 on the PC).
 */
class YawEstimator {
  private val g = DoubleArray(3)
  private var hasG = false
  private var lastNs = 0L
  @Volatile var yawDeg = 0.0
    private set

  fun reset() {
    yawDeg = 0.0
  }

  fun update(acc: DoubleArray?, gyro: DoubleArray?, tNs: Long): Double? {
    if (acc != null) {
      if (!hasG) {
        acc.copyInto(g)
        hasG = true
      } else {
        for (i in 0..2) g[i] = 0.95 * g[i] + 0.05 * acc[i]
      }
    }
    if (gyro == null || !hasG) return null
    val n = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
    if (n < 1e-3) return null
    val wUp = (gyro[0] * g[0] + gyro[1] * g[1] + gyro[2] * g[2]) / n
    if (lastNs != 0L) {
      val dt = ((tNs - lastNs) / 1e9).coerceIn(0.0, 0.2)
      yawDeg -= wUp * dt * 180.0 / PI
    }
    lastNs = tNs
    return yawDeg
  }
}
