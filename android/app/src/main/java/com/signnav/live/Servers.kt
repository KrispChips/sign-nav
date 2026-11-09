package com.signnav.live

import android.content.Context
import android.media.AudioAttributes
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.util.Locale
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

/** A listening socket with an accept thread; one handler thread per client. */
abstract class TcpServer(val port: Int, private val name: String) {
  @Volatile private var server: ServerSocket? = null
  val clients = CopyOnWriteArrayList<Socket>()

  fun start() {
    if (server != null) return
    val s = ServerSocket(port).apply { reuseAddress = true }
    server = s
    thread(name = "$name-accept", isDaemon = true) {
      while (!s.isClosed) {
        val c = try { s.accept() } catch (e: SocketException) { break }
        c.tcpNoDelay = true
        clients += c
        AppLog.i("$name: client ${c.inetAddress.hostAddress}")
        thread(name = "$name-client", isDaemon = true) {
          try {
            handle(c)
          } catch (e: Exception) {
            // client went away
          } finally {
            clients -= c
            runCatching { c.close() }
            AppLog.i("$name: client ${c.inetAddress.hostAddress} left")
          }
        }
      }
    }
  }

  fun stop() {
    runCatching { server?.close() }
    server = null
    clients.forEach { runCatching { it.close() } }
    clients.clear()
  }

  abstract fun handle(client: Socket)
}

/**
 * http://<phone>:8080/frames.mjpg  MJPEG (multipart/x-mixed-replace), OpenCV reads it directly
 * http://<phone>:8080/snapshot.jpg the newest frame
 * http://<phone>:8080/             a page showing the stream (check it in a browser)
 */
class MjpegServer(private val frames: FrameStore, port: Int = 8080) : TcpServer(port, "mjpeg") {
  override fun handle(client: Socket) {
    val reader = BufferedReader(InputStreamReader(client.getInputStream()))
    val request = reader.readLine() ?: return
    while (true) {                      // skip headers
      val line = reader.readLine() ?: break
      if (line.isEmpty()) break
    }
    val path = request.split(" ").getOrNull(1) ?: "/"
    val out = client.getOutputStream()
    when {
      path.startsWith("/frames.mjpg") -> stream(out)
      path.startsWith("/snapshot.jpg") -> {
        val jpeg = frames.latest() ?: frames.next(-1, 5000).first
        if (jpeg == null) return notFound(out, "no frame yet")
        out.write(("HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.size}\r\n" +
            "Cache-Control: no-cache\r\nConnection: close\r\n\r\n").toByteArray())
        out.write(jpeg)
        out.flush()
      }
      path == "/" -> {
        val html = "<html><body style='margin:0;background:#000'>" +
            "<img src='/frames.mjpg' style='height:100vh'></body></html>"
        out.write(("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: ${html.length}\r\n" +
            "Connection: close\r\n\r\n$html").toByteArray())
        out.flush()
      }
      else -> notFound(out, "not found")
    }
  }

  private fun stream(out: OutputStream) {
    out.write(("HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n" +
        "Cache-Control: no-cache\r\nConnection: close\r\n\r\n").toByteArray())
    out.flush()
    var seen = -1L
    while (true) {
      val (jpeg, seq) = frames.next(seen)
      if (jpeg == null || seq == seen) continue
      seen = seq
      out.write("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.size}\r\n\r\n".toByteArray())
      out.write(jpeg)
      out.write("\r\n".toByteArray())
      out.flush()
    }
  }

  private fun notFound(out: OutputStream, msg: String) {
    out.write(("HTTP/1.1 404 Not Found\r\nContent-Length: ${msg.length}\r\nConnection: close\r\n\r\n$msg")
        .toByteArray())
    out.flush()
  }
}

/**
 * tcp :9002, one JSON line per sentence from sign-nav: {"text", "kind": "sign"|"place", "t", "ttl_s"}.
 * Spoken in order with Android TTS (played as navigation guidance, so it goes to the glasses
 * when they're the phone's Bluetooth audio device), never interrupting. A sentence still
 * waiting in the queue `ttl_s` after it ARRIVED is dropped: a hallway warning said late is
 * worse than none. (Arrival time, not "t", so the phone's and PC's clocks needn't agree.)
 */
class SpeechServer(context: Context, port: Int = 9002) : TcpServer(port, "speech") {
  private data class Msg(val text: String, val kind: String, val arrivedMs: Long, val ttlMs: Long)

  private val queue = LinkedBlockingQueue<Msg>()
  @Volatile private var ready = false
  @Volatile var lastSpoken = ""
  @Volatile var spoken = 0
  @Volatile var dropped = 0
  private val tts: TextToSpeech = TextToSpeech(context.applicationContext) { status ->
    ready = status == TextToSpeech.SUCCESS
    AppLog.i(if (ready) "speech: TTS ready" else "speech: TTS failed to start ($status)")
  }

  init {
    tts.language = Locale.US
    tts.setAudioAttributes(
        AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build())
    thread(name = "speech-out", isDaemon = true) { speakLoop() }
  }

  override fun handle(client: Socket) {
    val reader = BufferedReader(InputStreamReader(client.getInputStream(), Charsets.UTF_8))
    while (true) {
      val line = reader.readLine() ?: return
      val msg = runCatching { JSONObject(line) }.getOrNull() ?: continue
      val text = msg.optString("text").trim()
      if (text.isEmpty()) continue
      val ttl = (msg.optDouble("ttl_s", 10.0) * 1000).toLong()
      queue.put(Msg(text, msg.optString("kind", "sign"), System.currentTimeMillis(), ttl))
    }
  }

  /** Also for testing from the phone's screen. */
  fun say(text: String) {
    queue.put(Msg(text, "test", System.currentTimeMillis(), 60_000))
  }

  private fun speakLoop() {
    while (true) {
      val m = queue.take()
      val waited = System.currentTimeMillis() - m.arrivedMs
      if (waited > m.ttlMs) {
        dropped++
        AppLog.i("speech: dropped (${waited / 1000.0}s old, ttl ${m.ttlMs / 1000.0}s): ${m.text}")
        continue
      }
      while (!ready) Thread.sleep(100)
      val id = "u${System.nanoTime()}"
      val done = CountDownLatch(1)
      tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) {}
        override fun onDone(utteranceId: String?) { if (utteranceId == id) done.countDown() }
        @Deprecated("Deprecated in Java")
        override fun onError(utteranceId: String?) { if (utteranceId == id) done.countDown() }
        override fun onError(utteranceId: String?, errorCode: Int) { if (utteranceId == id) done.countDown() }
      })
      tts.speak(m.text, TextToSpeech.QUEUE_ADD, null, id)
      lastSpoken = m.text
      spoken++
      AppLog.i("speech: ${m.kind}: ${m.text}")
      done.await(30, TimeUnit.SECONDS)
    }
  }

  fun shutdown() {
    stop()
    queue.clear()
    tts.stop()
    tts.shutdown()
  }
}

/**
 * tcp :9003: head yaw lines {"yaw_deg": <deg, + = turning right>, "t": <unix s>} to every
 * connected PC (sign-nav: --imu-from <phone>:9003).
 */
class YawServer(port: Int = 9003) : TcpServer(port, "yaw") {
  override fun handle(client: Socket) {
    // Nothing to read; keep the connection until the PC hangs up.
    val input = client.getInputStream()
    while (input.read() >= 0) { /* ignore */ }
  }

  fun publish(yawDeg: Double) {
    if (clients.isEmpty()) return
    val line = "{\"yaw_deg\":${"%.2f".format(Locale.US, yawDeg)},\"t\":${System.currentTimeMillis() / 1000.0}}\n"
        .toByteArray()
    for (c in clients) {
      try {
        c.getOutputStream().write(line)
      } catch (e: Exception) {
        clients -= c
        runCatching { c.close() }
      }
    }
  }
}
