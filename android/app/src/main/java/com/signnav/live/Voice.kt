package com.signnav.live

import android.content.Context
import android.content.Intent
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer

/**
 * Always-on voice commands (docs/step1-goal.md, section 4a): Android speech recognition in a
 * loop, each recognised phrase handed to [onText] (sent to the PC, which decides what it
 * means). Paused while the phone is speaking so it doesn't hear itself.
 *
 * Prefers the on-device recognizer (works offline, no network round trip). With
 * [glassesMic] the glasses' microphones are used over Bluetooth (SCO), which also routes
 * speech output through that link; otherwise the phone's microphone.
 *
 * SpeechRecognizer must be driven from the main thread: every call is posted there.
 */
class Voice(private val context: Context, private val onText: (String) -> Unit) {
  @Volatile var state = "off"
    private set
  @Volatile var lastHeard = "-"
    private set

  private val main = Handler(Looper.getMainLooper())
  private val audio = context.getSystemService(AudioManager::class.java)
  private var recognizer: SpeechRecognizer? = null
  private var wanted = false
  private var paused = false
  private var listening = false
  private var glassesMic = true

  fun start(glassesMic: Boolean) = main.post {
    this.glassesMic = glassesMic
    if (wanted) return@post
    if (!SpeechRecognizer.isRecognitionAvailable(context)) {
      state = "unavailable (no speech recognition on this phone)"
      AppLog.e("voice: no speech recognition service on this phone")
      return@post
    }
    wanted = true
    routeMic(glassesMic)
    listen()
  }

  fun stop() = main.post {
    wanted = false
    listening = false
    runCatching { recognizer?.cancel() }
    runCatching { recognizer?.destroy() }
    recognizer = null
    routeMic(false)
    state = "off"
  }

  /** While the phone speaks: stop listening (it would hear its own voice), then resume. */
  fun pause(on: Boolean) = main.post {
    paused = on
    if (on) {
      if (listening) runCatching { recognizer?.cancel() }
      listening = false
      if (wanted) state = "paused (speaking)"
    } else {
      main.postDelayed({ listen() }, 250)
    }
  }

  private fun routeMic(glasses: Boolean) {
    val am = audio ?: return
    if (glasses) {
      val bt = am.availableCommunicationDevices.firstOrNull {
        it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO
      }
      if (bt != null && am.setCommunicationDevice(bt)) {
        AppLog.i("voice: using the glasses' microphone (${bt.productName})")
      } else {
        AppLog.i("voice: glasses microphone not available; using the phone's")
      }
    } else {
      runCatching { am.clearCommunicationDevice() }
    }
  }

  private fun create(): SpeechRecognizer {
    val onDevice = SpeechRecognizer.isOnDeviceRecognitionAvailable(context)
    val r = if (onDevice) SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
        else SpeechRecognizer.createSpeechRecognizer(context)
    AppLog.i("voice: ${if (onDevice) "on-device" else "online"} recognizer")
    r.setRecognitionListener(listener)
    return r
  }

  private fun listen() {
    if (!wanted || paused || listening) return
    val r = recognizer ?: create().also { recognizer = it }
    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
      putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
      putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
      putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
      putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
      putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
      putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 1200L)
    }
    runCatching { r.startListening(intent) }
        .onSuccess { listening = true; state = "listening" }
        .onFailure { e ->
          AppLog.e("voice: could not start listening: ${e.message}")
          restart(1000, recreate = true)
        }
  }

  private fun restart(delayMs: Long, recreate: Boolean = false) {
    listening = false
    if (recreate) {
      runCatching { recognizer?.destroy() }
      recognizer = null
    }
    if (wanted) main.postDelayed({ listen() }, delayMs)
  }

  private val listener = object : RecognitionListener {
    override fun onResults(results: Bundle?) {
      val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
          ?.firstOrNull()?.trim().orEmpty()
      if (text.isNotEmpty()) {
        lastHeard = text
        AppLog.i("voice: heard \"$text\"")
        onText(text)
      }
      restart(100)
    }

    override fun onError(error: Int) {
      when (error) {
        // Silence or nothing understood: just listen again.
        SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> restart(100)
        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> restart(500, recreate = true)
        SpeechRecognizer.ERROR_CLIENT -> restart(300)   // also after our own cancel()
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> {
          wanted = false
          listening = false
          state = "no microphone permission"
          AppLog.e("voice: microphone permission missing (allow it in the app's settings)")
        }
        else -> {
          AppLog.d("voice: recognizer error $error; retrying")
          restart(1000, recreate = true)
        }
      }
    }

    override fun onReadyForSpeech(params: Bundle?) {}
    override fun onBeginningOfSpeech() {}
    override fun onRmsChanged(rmsdB: Float) {}
    override fun onBufferReceived(buffer: ByteArray?) {}
    override fun onEndOfSpeech() {}
    override fun onPartialResults(partialResults: Bundle?) {}
    override fun onEvent(eventType: Int, params: Bundle?) {}
  }
}
