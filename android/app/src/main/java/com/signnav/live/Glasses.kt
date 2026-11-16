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
import com.meta.wearable.dat.inputs.Inputs
import com.meta.wearable.dat.inputs.addInputs
import com.meta.wearable.dat.inputs.types.InputEvent
import com.meta.wearable.dat.inputs.types.InputSource
import com.meta.wearable.dat.inputs.types.InputsConfiguration
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
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * One DAT device session with two capabilities (Meta DAT 1.0.1, patterns from the
 * CameraAccess and BirdSpotter samples):
 * - camera: a compressed HEVC stream (keeps flowing with the phone locked), decoded by
 *   [HevcDecoder] into [FrameStore];
 * - motion (experimental): IMU samples turned into head yaw for [onYaw]. If the glasses or
 *   the SDK refuse it, streaming carries on without yaw.
 * - inputs (beta): touchpad / button events. For now only logged together with the stream's
 *   state and frame flow, to find a gesture that arrives reliably without pausing the stream
 *   (docs/step1-goal.md, section 4). If refused, streaming carries on without inputs.
 */
class Glasses(
    private val scope: CoroutineScope,
    private val frames: FrameStore,
    private val onYaw: (Double) -> Unit,
    /** A glasses gesture as a link event: "select", "capture", "button", "nav" (+ direction). */
    private val onInputEvent: (String, String?) -> Unit = { _, _ -> },
) {
  @Volatile var sessionState = "-"
  @Volatile var streamState = "-"
  @Volatile var motionState = "-"
  @Volatile var streamSize = "-"
  @Volatile var inputsState = "-"
  @Volatile var lastInput = "-"

  private var session: DeviceSession? = null
  private var camera: Camera? = null
  private var motion: Motion? = null
  private var inputs: Inputs? = null
  // Compressed frames received from the glasses, and when the last one came (for gaps).
  @Volatile private var framesIn = 0L
  @Volatile private var lastFrameMs = 0L
  private var decoder: HevcDecoder? = null
  private val jobs = mutableListOf<Job>()
  private val yaw = YawEstimator()
  private var quality = VideoQuality.HIGH
  private var fps = 24
  @Volatile private var wanted = false
  private var retryJob: Job? = null

  fun start(quality: VideoQuality, fps: Int) {
    this.quality = quality
    this.fps = fps
    wanted = true
    openSession()
  }

  /** While running, a session that fails or ends is retried every few seconds: the glasses
   * may be folded, out of range or still connecting to the phone. */
  private fun retryLater(why: String) {
    if (!wanted || retryJob?.isActive == true) return
    sessionState = "retrying: $why"
    retryJob = scope.launch {
      kotlinx.coroutines.delay(4000)
      if (wanted && session == null) openSession()
    }
  }

  private fun openSession() {
    if (session != null) return
    AppLog.i("opening a session with the glasses")
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
                if (inputs == null) attachInputs(s)
              } else if (st == DeviceSessionState.STOPPED && session === s) {
                endSession()
                retryLater("session stopped")
              }
            }
          }
          s.start() // fire-and-forget; progress arrives through s.state / s.errors
        }
        .onFailure { e, _ ->
          AppLog.e("could not create a session: ${e.description}")
          retryLater(e.description)
        }
  }

  /** Drops this session's capabilities (it has ended) but keeps the collectors' scope. */
  private fun endSession() {
    jobs.forEach { it.cancel() }
    jobs.clear()
    synchronized(this) {
      decoder?.stop()
      decoder = null
    }
    motion = null
    inputs = null
    camera = null
    session = null
    streamState = "-"
    motionState = "-"
    inputsState = "-"
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
              AppLog.i("stream $st @ ${clock()}")
            }
          }
          jobs += scope.launch { watchFrameGaps() }
          jobs += scope.launch {
            stream.errorStream.collect { e -> AppLog.e("stream: ${e.description}") }
          }
          stream.start().onFailure { e, _ -> AppLog.e("stream would not start: ${e.description}") }
          AppLog.i("camera added: $quality @ $fps fps (compressed HEVC)")
        }
        .onFailure { e, _ -> AppLog.e("camera would not attach: ${e.description}") }
  }

  private fun onFrame(frame: VideoFrame) {
    framesIn++
    lastFrameMs = System.currentTimeMillis()
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

  /** Touchpad + both buttons; Neural Band sources left out (none here). Adding Inputs starts
   * it (no start()); the SDK re-attaches it itself after a reconnect. */
  private fun attachInputs(s: DeviceSession) {
    val config = InputsConfiguration(
        sources = setOf(InputSource.CAPTOUCH, InputSource.CAPTURE_BUTTON, InputSource.ACTION_BUTTON),
        consumeBack = true,
    )
    s.addInputs(config)
        .onSuccess { inp ->
          inputs = inp
          jobs += scope.launch { inp.events.collect { onInput(it) } }
          jobs += scope.launch {
            inp.state.collect { st ->
              inputsState = st.toString()
              AppLog.i("inputs $st")
            }
          }
          jobs += scope.launch {
            inp.errors.collect { e ->
              if (e != null) {
                inputsState = "error: $e"
                AppLog.e("inputs: $e (${e.description})")
              }
            }
          }
        }
        .onFailure { e, _ ->
          inputsState = "unavailable"
          AppLog.e("inputs unavailable (streaming continues): ${e.description}")
        }
  }

  /** Logs the event with the stream's state and frame flow now, 1 s and 3 s later: a gesture
   * that pauses the stream shows up as frames stopping even if the state doesn't change. */
  private fun onInput(ev: InputEvent) {
    val what = when (ev) {
      is InputEvent.Select -> "SELECT"
      is InputEvent.Nav -> "NAV ${ev.direction}"
      is InputEvent.Capture -> "CAPTURE ${ev.pressType}"
      is InputEvent.Button -> "BUTTON ${ev.button}"
      is InputEvent.Back -> "BACK"
      is InputEvent.Drag -> "DRAG ${ev.action}"
    }
    when (ev) {
      is InputEvent.Select -> onInputEvent("select", null)
      is InputEvent.Capture -> onInputEvent("capture", ev.pressType.toString().lowercase())
      is InputEvent.Button -> onInputEvent("button", null)
      is InputEvent.Nav -> onInputEvent("nav", ev.direction.toString().lowercase())
      else -> {}
    }
    val at = clock()
    lastInput = "$what @ $at"
    val n0 = framesIn
    AppLog.i("input $what from ${ev.source} @ $at (glasses t=${ev.timestampMs} ms), " +
        "stream $streamState, frames in $n0")
    scope.launch {
      for (after in listOf(1000L, 3000L)) {
        delay(if (after == 1000L) 1000L else 2000L)
        AppLog.i("  +${after / 1000}s after $what: stream $streamState, " +
            "${framesIn - n0} frames since")
      }
    }
  }

  /** Logs when frames stop for over a second and when they come back. */
  private suspend fun watchFrameGaps() {
    var inGap = false
    while (true) {
      delay(250)
      val last = lastFrameMs
      if (last == 0L) continue
      val idle = System.currentTimeMillis() - last
      if (!inGap && idle > 1000) {
        inGap = true
        AppLog.i("frames stopped (none for ${idle} ms), stream $streamState @ ${clock()}")
      } else if (inGap && idle < 250) {
        inGap = false
        AppLog.i("frames resumed, stream $streamState @ ${clock()}")
      }
    }
  }

  private fun clock(): String =
      java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.US).format(java.util.Date())

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
    wanted = false
    retryJob?.cancel()
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
    inputs = null
    camera = null
    session = null
    sessionState = "-"
    streamState = "-"
    motionState = "-"
    inputsState = "-"
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
