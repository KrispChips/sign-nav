package com.signnav.live

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import java.io.ByteArrayOutputStream

/**
 * The newest frame as a JPEG, for the MJPEG server. Frames come from the decoder at the
 * stream's rate (e.g. 24 fps); only `maxFps` of them per second are converted, and a new
 * one replaces the old (the PC always wants the newest view, never a backlog).
 */
class FrameStore {
  @Volatile var maxFps = 10.0
  @Volatile var jpegQuality = 70

  private val lock = Object()
  private var jpeg: ByteArray? = null
  private var seq = 0L
  private var lastEncodeNs = 0L

  // stats for the status screen
  @Volatile var width = 0
  @Volatile var height = 0
  @Volatile var decodedFrames = 0L
  @Volatile var encodedFrames = 0L
  @Volatile var lastJpegBytes = 0

  /** Called on the decoder thread for every decoded frame. Must not keep [image]. */
  fun offer(image: Image) {
    decodedFrames++
    val now = System.nanoTime()
    if (now - lastEncodeNs < (1e9 / maxFps).toLong()) return
    lastEncodeNs = now
    val crop = image.cropRect
    val w = crop.width() and 1.inv()     // even sizes for 4:2:0
    val h = crop.height() and 1.inv()
    val nv21 = toNv21(image, crop, w, h)
    val out = ByteArrayOutputStream(w * h / 4)
    YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), jpegQuality, out)
    val bytes = out.toByteArray()
    width = w
    height = h
    lastJpegBytes = bytes.size
    encodedFrames++
    synchronized(lock) {
      jpeg = bytes
      seq++
      lock.notifyAll()
    }
  }

  /** Blocks until a frame newer than [seen] exists (or timeout). Returns (jpeg, seq). */
  fun next(seen: Long, timeoutMs: Long = 2000): Pair<ByteArray?, Long> {
    synchronized(lock) {
      val end = System.currentTimeMillis() + timeoutMs
      while (seq == seen) {
        val left = end - System.currentTimeMillis()
        if (left <= 0) break
        lock.wait(left)
      }
      return jpeg to seq
    }
  }

  fun latest(): ByteArray? = synchronized(lock) { jpeg }

  fun clear() {
    synchronized(lock) {
      jpeg = null
      lock.notifyAll()
    }
    decodedFrames = 0
    encodedFrames = 0
  }

  companion object {
    /** YUV_420_888 (any row/pixel strides) -> NV21 (Y plane, then interleaved V,U). */
    fun toNv21(image: Image, crop: Rect, w: Int, h: Int): ByteArray {
      val out = ByteArray(w * h * 3 / 2)
      val (yP, uP, vP) = image.planes
      val yBuf = yP.buffer
      val yRow = yP.rowStride
      val yPix = yP.pixelStride
      var o = 0
      for (r in 0 until h) {
        val base = (crop.top + r) * yRow + crop.left * yPix
        if (yPix == 1) {
          yBuf.position(base)
          yBuf.get(out, o, w)
          o += w
        } else {
          for (c in 0 until w) out[o++] = yBuf.get(base + c * yPix)
        }
      }
      val uBuf = uP.buffer
      val vBuf = vP.buffer
      val cRow = uP.rowStride
      val cPix = uP.pixelStride
      val vRow = vP.rowStride
      val vPix = vP.pixelStride
      for (r in 0 until h / 2) {
        val ur = (crop.top / 2 + r) * cRow + (crop.left / 2) * cPix
        val vr = (crop.top / 2 + r) * vRow + (crop.left / 2) * vPix
        for (c in 0 until w / 2) {
          out[o++] = vBuf.get(vr + c * vPix)
          out[o++] = uBuf.get(ur + c * cPix)
        }
      }
      return out
    }
  }
}
