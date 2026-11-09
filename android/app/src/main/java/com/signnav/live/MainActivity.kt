package com.signnav.live

import android.Manifest
import android.graphics.Typeface
import android.os.Bundle
import android.text.InputType
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts.RequestMultiplePermissions
import androidx.lifecycle.lifecycleScope
import com.meta.wearable.dat.camera.types.VideoQuality
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import kotlin.coroutines.resume
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine

/**
 * One screen: connect the glasses (registration via the Meta AI app), Start/Stop streaming,
 * a few settings, and live status for checking the link to the PC.
 */
class MainActivity : ComponentActivity() {
  companion object {
    private var sdkInitialized = false
  }

  private lateinit var status: TextView
  private lateinit var log: TextView
  private lateinit var startStop: Button
  private lateinit var quality: RadioGroup
  private lateinit var streamFps: EditText
  private lateinit var mjpegFps: EditText
  private lateinit var jpegQuality: EditText
  private var registration = "-"

  private val androidPermissions =
      registerForActivityResult(RequestMultiplePermissions()) { result ->
        // Bluetooth is what the SDK needs; notifications only show the "streaming" notice
        if (result[Manifest.permission.BLUETOOTH_CONNECT] == true) initSdk()
        else AppLog.e("Bluetooth permission is needed to reach the glasses")
        if (result[Manifest.permission.POST_NOTIFICATIONS] == false) {
          AppLog.i("notifications off: streaming still works, without the status notification")
        }
      }

  private var cameraContinuation: CancellableContinuation<PermissionStatus>? = null
  private val cameraPermission =
      registerForActivityResult(Wearables.RequestPermissionContract()) { result ->
        cameraContinuation?.resume(result.getOrDefault(PermissionStatus.Denied))
        cameraContinuation = null
      }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(buildUi())
    androidPermissions.launch(
        arrayOf(Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.POST_NOTIFICATIONS))
    lifecycleScope.launch {
      while (true) {
        refresh()
        delay(500)
      }
    }
  }

  private fun initSdk() {
    if (!sdkInitialized) {
      Wearables.initialize(this)
          .onSuccess { AppLog.i("DAT SDK initialized") }
          .onFailure { e, _ -> AppLog.e("DAT SDK failed to initialize: ${e.description}") }
      sdkInitialized = true
    }
    lifecycleScope.launch {
      Wearables.registrationState.collect { registration = it.toString() }
    }
  }

  private suspend fun ensureCameraPermission(): Boolean {
    var status: PermissionStatus = PermissionStatus.Denied
    Wearables.checkPermissionStatus(Permission.CAMERA)
        .onSuccess { status = it }
        .onFailure { e, _ -> AppLog.e("camera permission check failed: ${e.description}") }
    if (status == PermissionStatus.Granted) return true
    AppLog.i("asking for glasses camera permission (opens the Meta AI app)")
    status = suspendCancellableCoroutine<PermissionStatus> { cont ->
      cameraContinuation = cont
      cont.invokeOnCancellation { cameraContinuation = null }
      cameraPermission.launch(Permission.CAMERA)
    }
    return status == PermissionStatus.Granted
  }

  private fun onStartStop() {
    if (LiveSession.running) {
      LiveSession.stop(this)
      window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
      return
    }
    applySettings()
    lifecycleScope.launch {
      if (!ensureCameraPermission()) {
        AppLog.e("no camera permission for the glasses")
        return@launch
      }
      val q = if (quality.checkedRadioButtonId == 2) VideoQuality.MEDIUM else VideoQuality.HIGH
      val fps = streamFps.text.toString().toIntOrNull()?.takeIf { it in setOf(2, 7, 15, 24, 30) } ?: 24
      LiveSession.start(this@MainActivity, q, fps)
      window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
  }

  private fun applySettings() {
    LiveSession.frames.maxFps = mjpegFps.text.toString().toDoubleOrNull()?.coerceIn(1.0, 30.0) ?: 10.0
    LiveSession.frames.jpegQuality = jpegQuality.text.toString().toIntOrNull()?.coerceIn(30, 95) ?: 70
  }

  // ---- status

  private var lastTick = 0L
  private var lastDecoded = 0L
  private var lastEncoded = 0L

  private fun refresh() {
    val now = System.currentTimeMillis()
    val f = LiveSession.frames
    val dt = (now - lastTick) / 1000.0
    val decFps = if (lastTick > 0 && dt > 0) (f.decodedFrames - lastDecoded) / dt else 0.0
    val encFps = if (lastTick > 0 && dt > 0) (f.encodedFrames - lastEncoded) / dt else 0.0
    lastTick = now
    lastDecoded = f.decodedFrames
    lastEncoded = f.encodedFrames
    val sp = LiveSession.speech
    val lines = mutableListOf(
        "registration: $registration",
        "running: ${LiveSession.running}",
    )
    LiveSession.status().forEach { (k, v) -> lines += "$k: $v" }
    lines += "decoded: %.1f fps   sent: %.1f fps, %dx%d, %.0f KB/s".format(
        decFps, encFps, f.width, f.height, encFps * f.lastJpegBytes / 1024.0)
    lines += "yaw: " + (LiveSession.yawDeg?.let { "%+.0f deg".format(it) } ?: "-")
    lines += "speech: ${sp?.spoken ?: 0} said, ${sp?.dropped ?: 0} dropped (expired)"
    if (!sp?.lastSpoken.isNullOrEmpty()) lines += "last: \"${sp?.lastSpoken}\""
    lines += ""
    lines += "phone addresses (use on the PC):"
    LiveSession.addresses().forEach { lines += "  $it" }
    lines += "  http://<address>:${LiveSession.MJPEG_PORT}/frames.mjpg"
    status.text = lines.joinToString("\n")
    log.text = AppLog.recent(15).joinToString("\n")
    startStop.text = if (LiveSession.running) "Stop" else "Start"
  }

  // ---- layout (built in code: one screen, no resources to keep in sync)

  private fun buildUi(): ScrollView {
    val pad = (12 * resources.displayMetrics.density).toInt()
    val col = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(pad, pad, pad, pad)
    }
    fun button(text: String, onClick: () -> Unit) =
        Button(this).apply { this.text = text; setOnClickListener { onClick() } }.also { col.addView(it) }
    fun label(text: String) = TextView(this).apply { this.text = text }.also { col.addView(it) }
    fun number(hint: String, value: String) = EditText(this).apply {
      this.hint = hint
      setText(value)
      inputType = InputType.TYPE_CLASS_NUMBER
    }.also { col.addView(it) }

    button("Connect glasses (Meta AI app)") { Wearables.startRegistration(this) }
    startStop = button("Start") { onStartStop() }
    button("Test speech") {
      LiveSession.speech?.say("Sign navigation test. Hallway ahead on your left.")
          ?: AppLog.i("press Start first (speech starts with the servers)")
    }
    button("Reset yaw to 0") { LiveSession.resetYaw() }
    CheckBox(this).apply {
      text = "Invert yaw (if turning right makes it go down)"
      setOnCheckedChangeListener { _, on -> LiveSession.invertYaw = on }
    }.also { col.addView(it) }

    label("Glasses video quality")
    quality = RadioGroup(this).apply {
      orientation = RadioGroup.HORIZONTAL
      addView(RadioButton(context).apply { id = 1; text = "High 720x1280" })
      addView(RadioButton(context).apply { id = 2; text = "Medium 504x896" })
      check(1)
    }.also { col.addView(it) }
    label("Glasses stream fps (2, 7, 15, 24, 30)")
    streamFps = number("24", "24")
    label("Frames sent to the PC per second")
    mjpegFps = number("10", "10")
    label("JPEG quality (30-95)")
    jpegQuality = number("70", "70")
    button("Apply rate/quality (while running)") { applySettings() }

    status = TextView(this).apply {
      typeface = Typeface.MONOSPACE
      setPadding(0, pad, 0, pad)
    }.also { col.addView(it) }
    label("Log")
    log = TextView(this).apply {
      typeface = Typeface.MONOSPACE
      textSize = 11f
    }.also { col.addView(it) }
    return ScrollView(this).apply { addView(col, MATCH_PARENT, WRAP_CONTENT) }
  }
}
