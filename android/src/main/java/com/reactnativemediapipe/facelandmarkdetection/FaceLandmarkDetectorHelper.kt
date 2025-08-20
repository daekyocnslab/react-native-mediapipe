package com.reactnativemediapipe.facelandmarkdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.facebook.react.common.annotations.VisibleForTesting
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.mrousavy.camera.core.types.Orientation
import com.reactnativemediapipe.shared.orientationToDegrees
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class FaceLandmarkDetectorHelper(
    var minFaceDetectionConfidence: Float = DEFAULT_FACE_DETECTION_CONFIDENCE,
    var minFaceTrackingConfidence: Float = DEFAULT_FACE_TRACKING_CONFIDENCE,
    var minFacePresenceConfidence: Float = DEFAULT_FACE_PRESENCE_CONFIDENCE,
    var maxNumFaces: Int = DEFAULT_NUM_FACES,
    var currentDelegate: Int = DELEGATE_CPU,
    var currentModel: String,
    var runningMode: RunningMode = RunningMode.IMAGE,
    val context: Context,
    var faceLandmarkDetectorListener: DetectorListener? = null
) {

  private var faceLandmarker: FaceLandmarker? = null
  private var imageRotation = 0
  private var currentBitmap: Bitmap? = null

  // MediaPipe 랜드마크 인덱스 정의 (파이썬 코드와 동일)
  private val leftEyeIndices = listOf(33, 160, 158, 133, 153, 144)
  private val rightEyeIndices = listOf(362, 385, 387, 263, 373, 380)
  private val leftIrisIndices = listOf(468, 469, 470, 471)
  private val rightIrisIndices = listOf(473, 474, 475, 476)
  
  // MAR 계산을 위한 입술 랜드마크 인덱스
  private val mouthIndices = listOf(
    13, 14, 15, 16, 17, 18,  // 상단 입술 외곽
    61, 84, 17, 314, 405, 320, 307, 375, 308, 324, 318,  // 하단 입술 외곽
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324  // 입술 내부
  )

  init {
    setupFaceLandmarker()
  }

  fun clearFaceLandmarker() {
    faceLandmarkDetectorListener = null
    currentBitmap = null
    Handler(Looper.getMainLooper())
        .postDelayed(
            {
              faceLandmarker?.close()
              faceLandmarker = null
            },
            100
        )
  }

  private fun setupFaceLandmarker() {
    val baseOptionBuilder = BaseOptions.builder()

    when (currentDelegate) {
      DELEGATE_CPU -> {
        baseOptionBuilder.setDelegate(Delegate.CPU)
      }
      DELEGATE_GPU -> {
        baseOptionBuilder.setDelegate(Delegate.GPU)
      }
    }

    baseOptionBuilder.setModelAssetPath(currentModel)

    when (runningMode) {
      RunningMode.LIVE_STREAM -> {
        if (faceLandmarkDetectorListener == null) {
          throw IllegalStateException(
              "faceLandmarkDetectorListener must be set when runningMode is LIVE_STREAM."
          )
        }
      }
      else -> {
        // no-op
      }
    }

    try {
      val baseOptions = baseOptionBuilder.build()
      val optionsBuilder =
          FaceLandmarker.FaceLandmarkerOptions.builder()
              .setBaseOptions(baseOptions)
              .setMinFaceDetectionConfidence(minFaceDetectionConfidence)
              .setMinTrackingConfidence(minFaceTrackingConfidence)
              .setMinFacePresenceConfidence(minFacePresenceConfidence)
              .setNumFaces(maxNumFaces)
              .setOutputFaceBlendshapes(true)
              .setRunningMode(runningMode)

      if (runningMode == RunningMode.LIVE_STREAM) {
        optionsBuilder
            .setRunningMode(runningMode)
            .setResultListener(this::returnLivestreamResult)
            .setErrorListener(this::returnLivestreamError)
      }

      val options = optionsBuilder.build()
      faceLandmarker = FaceLandmarker.createFromOptions(context, options)
    } catch (e: IllegalStateException) {
      faceLandmarkDetectorListener?.onError(
          "Face Landmarker failed to initialize. See error logs for details"
      )
      Log.e(TAG, "MediaPipe failed to load the task with error: " + e.message)
    } catch (e: RuntimeException) {
      faceLandmarkDetectorListener?.onError(
          "Face Landmarker failed to initialize. See error logs for details",
          GPU_ERROR
      )
      Log.e(TAG, "Face Landmarker failed to load model with error: " + e.message)
    }
  }

  fun isClosed(): Boolean {
    return faceLandmarker == null
  }

  /**
   * EAR (눈 종횡비) 계산
   */
  private fun calculateEAR(eyeLandmarks: List<NormalizedLandmark>): Float {
    if (eyeLandmarks.size != 6) return 0f
    
    // 수직 거리 (눈 높이) 계산
    val verticalDist1 = sqrt(
      (eyeLandmarks[1].x() - eyeLandmarks[5].x()).pow(2) + 
      (eyeLandmarks[1].y() - eyeLandmarks[5].y()).pow(2)
    )
    val verticalDist2 = sqrt(
      (eyeLandmarks[2].x() - eyeLandmarks[4].x()).pow(2) + 
      (eyeLandmarks[2].y() - eyeLandmarks[4].y()).pow(2)
    )
    
    // 수평 거리 (눈 너비) 계산
    val horizontalDist = sqrt(
      (eyeLandmarks[0].x() - eyeLandmarks[3].x()).pow(2) + 
      (eyeLandmarks[0].y() - eyeLandmarks[3].y()).pow(2)
    )
    
    return if (horizontalDist > 0) {
      (verticalDist1 + verticalDist2) / (2.0f * horizontalDist)
    } else 0f
  }

  /**
   * MAR (입 종횡비) 계산
   */
  private fun calculateMAR(landmarks: List<NormalizedLandmark>): Float {
    // 입의 수직 거리
    val verticalDistances = listOf(
      // 윗입술과 아랫입술의 중심
      sqrt((landmarks[13].x() - landmarks[14].x()).pow(2) + (landmarks[13].y() - landmarks[14].y()).pow(2)),
      // 입의 왼쪽
      sqrt((landmarks[78].x() - landmarks[95].x()).pow(2) + (landmarks[78].y() - landmarks[95].y()).pow(2)),
      // 입의 오른쪽
      sqrt((landmarks[308].x() - landmarks[324].x()).pow(2) + (landmarks[308].y() - landmarks[324].y()).pow(2))
    )
    
    // 수평 거리 (입 너비)
    val horizontalDist = sqrt(
      (landmarks[61].x() - landmarks[291].x()).pow(2) + 
      (landmarks[61].y() - landmarks[291].y()).pow(2)
    )
    
    val avgVerticalDist = verticalDistances.average().toFloat()
    return if (horizontalDist > 0) avgVerticalDist / horizontalDist else 0f
  }

  /**
   * 시선 추정 (간단한 버전)
   */
  private fun estimateGaze(irisCenter: Pair<Float, Float>, eyeLeft: NormalizedLandmark, eyeRight: NormalizedLandmark): Float {
    val eyeCenterX = (eyeLeft.x() + eyeRight.x()) / 2f
    val gazeOffset = irisCenter.first - eyeCenterX
    val eyeWidth = abs(eyeRight.x() - eyeLeft.x())
    return if (eyeWidth > 0) gazeOffset / eyeWidth else 0f
  }

  /**
   * 랜드마크의 중심점 계산
   */
  private fun calculateCenter(landmarks: List<NormalizedLandmark>): Pair<Float, Float> {
    val centerX = landmarks.map { it.x() }.average().toFloat()
    val centerY = landmarks.map { it.y() }.average().toFloat()
    return Pair(centerX, centerY)
  }

  /**
   * 두 점 사이의 거리 계산
   */
  private fun calculateDistance(p1: Pair<Float, Float>, p2: Pair<Float, Float>): Float {
    return sqrt((p1.first - p2.first).pow(2) + (p1.second - p2.second).pow(2))
  }

  /**
   * 얼굴 특징을 계산하는 메인 함수
   */
  private fun calculateFaceFeatures(landmarks: List<NormalizedLandmark>, imageWidth: Int, imageHeight: Int): FaceFeatures {
    // 왼쪽 및 오른쪽 눈 랜드마크 추출
    val leftEyeLandmarks = leftEyeIndices.map { landmarks[it] }
    val rightEyeLandmarks = rightEyeIndices.map { landmarks[it] }
    val leftIrisLandmarks = leftIrisIndices.map { landmarks[it] }
    val rightIrisLandmarks = rightIrisIndices.map { landmarks[it] }
    
    // EAR 계산 (정규화된 좌표 사용)
    val leftEAR = calculateEAR(leftEyeLandmarks)
    val rightEAR = calculateEAR(rightEyeLandmarks)
    val avgEAR = (leftEAR + rightEAR) / 2f
    
    // MAR 계산 (정규화된 좌표 사용)
    val mar = calculateMAR(landmarks)
    
    // 눈과 홍채의 중심점 계산 (정규화된 좌표 사용)
    val leftEyeCenter = calculateCenter(leftEyeLandmarks)
    val rightEyeCenter = calculateCenter(rightEyeLandmarks)
    val leftIrisCenter = calculateCenter(leftIrisLandmarks)
    val rightIrisCenter = calculateCenter(rightIrisLandmarks)
    
    // 얼굴 중심점 계산 (정규화된 좌표 사용)
    val faceCenter = calculateCenter(landmarks)
    
    // 시선 추정 (정규화된 좌표 사용)
    val leftGaze = estimateGaze(leftIrisCenter, landmarks[33], landmarks[133])
    val rightGaze = estimateGaze(rightIrisCenter, landmarks[362], landmarks[263])
    
    // 눈 너비를 계산하기 위해 픽셀 좌표로 변환
    val leftEyePixelLeft = Pair(landmarks[33].x() * imageWidth, landmarks[33].y() * imageHeight)
    val leftEyePixelRight = Pair(landmarks[133].x() * imageWidth, landmarks[133].y() * imageHeight)
    val rightEyePixelLeft = Pair(landmarks[362].x() * imageWidth, landmarks[362].y() * imageHeight)
    val rightEyePixelRight = Pair(landmarks[263].x() * imageWidth, landmarks[263].y() * imageHeight)
    
    val leftEyeWidth = calculateDistance(leftEyePixelLeft, leftEyePixelRight)
    val rightEyeWidth = calculateDistance(rightEyePixelLeft, rightEyePixelRight)
    
    return FaceFeatures(
      earLeft = leftEAR,
      earRight = rightEAR,
      earAvg = avgEAR,
      gazeLeft = leftGaze,
      gazeRight = rightGaze,
      mar = mar,
      leftEyeX = leftEyeCenter.first * imageWidth,
      leftEyeY = leftEyeCenter.second * imageHeight,
      rightEyeX = rightEyeCenter.first * imageWidth,
      rightEyeY = rightEyeCenter.second * imageHeight,
      leftIrisX = leftIrisCenter.first * imageWidth,
      leftIrisY = leftIrisCenter.second * imageHeight,
      rightIrisX = rightIrisCenter.first * imageWidth,
      rightIrisY = rightIrisCenter.second * imageHeight,
      faceCenterX = faceCenter.first * imageWidth,
      faceCenterY = faceCenter.second * imageHeight,
      leftEyeWidth = leftEyeWidth,
      rightEyeWidth = rightEyeWidth
    )
  }

  fun detectImage(image: Bitmap): ResultBundle {
    if (runningMode != RunningMode.IMAGE) {
      throw IllegalArgumentException(
          "Attempting to call detectImage while not using RunningMode.IMAGE"
      )
    }

    val startTime = SystemClock.uptimeMillis()
    val mpImage = BitmapImageBuilder(image).build()

    return try {
      val landmarkResult = faceLandmarker?.detect(mpImage)
      val inferenceTimeMs = SystemClock.uptimeMillis() - startTime
      
      if (landmarkResult != null) {
        // 얼굴이 감지된 경우에만 특징 계산 (이미지 크기 정보 전달)
        var faceFeatures: FaceFeatures? = null
        if (landmarkResult.faceLandmarks().isNotEmpty()) {
          faceFeatures = calculateFaceFeatures(landmarkResult.faceLandmarks()[0], image.width, image.height)
          Log.d(TAG, "Face detected in detectImage - features calculated")
        } else {
          Log.d(TAG, "No face detected in detectImage")
        }
        
        ResultBundle(
            results = listOf(landmarkResult),
            inferenceTime = inferenceTimeMs,
            inputImageHeight = image.height,
            inputImageWidth = image.width,
            inputImageRotation = imageRotation,
            croppedFrame = null,      // IMAGE 모드에서는 null
            onnxInputData = null,     // IMAGE 모드에서는 null
            faceFeatures = faceFeatures // 얼굴이 감지되지 않으면 null
        )
      } else {
        // 감지에 실패하면 빈 결과와 함께 ResultBundle 반환
        Log.d(TAG, "Face detection returned null - returning empty ResultBundle")
        ResultBundle(
            results = emptyList(),
            inferenceTime = SystemClock.uptimeMillis() - startTime,
            inputImageHeight = image.height,
            inputImageWidth = image.width,
            inputImageRotation = imageRotation,
            croppedFrame = null,
            onnxInputData = null,
            faceFeatures = null
        )
      }
    } catch (e: Exception) {
      // 오류 발생 시 빈 결과와 함께 ResultBundle 반환
      Log.e(TAG, "Face detection failed: ${e.message}", e)
      ResultBundle(
          results = emptyList(),
          inferenceTime = SystemClock.uptimeMillis() - startTime,
          inputImageHeight = image.height,
          inputImageWidth = image.width,
          inputImageRotation = imageRotation,
          croppedFrame = null,
          onnxInputData = null,
          faceFeatures = null
      )
    }
  }

  // 기존 메서드 (하위 호환성 유지)
  fun detectLiveStream(mpImage: MPImage, orientation: Orientation) {
    detectLiveStream(mpImage, orientation, null)
  }

  // 비트맵을 함께 전달하는 새로운 메서드
  fun detectLiveStream(mpImage: MPImage, orientation: Orientation, sourceBitmap: Bitmap?) {
    if (runningMode != RunningMode.LIVE_STREAM) {
      throw IllegalArgumentException(
          "Attempting to call detectLiveStream while not using RunningMode.LIVE_STREAM"
      )
    }
    val frameTime = SystemClock.uptimeMillis()
    this.imageRotation = orientationToDegrees(Orientation.PORTRAIT)
    this.currentBitmap = sourceBitmap
    detectAsync(mpImage, frameTime, this.imageRotation)
  }

  @VisibleForTesting
  fun detectAsync(mpImage: MPImage, frameTime: Long, imageRotation: Int) {
    val imageProcessingOptions = ImageProcessingOptions.builder()
        .setRotationDegrees(imageRotation)
        .build()
    faceLandmarker?.detectAsync(mpImage, imageProcessingOptions, frameTime)
  }

  /**
   * 잘라낸 비트맵을 ONNX 모델 입력 형식(NHWC Float32Array)으로 변환
   */
  private fun convertBitmapToOnnxInput(croppedBitmap: Bitmap): FloatArray {
    val resizedBitmap = Bitmap.createScaledBitmap(croppedBitmap, 64, 64, true)
    
    val tensorData = FloatArray(1 * 64 * 64 * 3)
    val pixels = IntArray(64 * 64)
    resizedBitmap.getPixels(pixels, 0, 64, 0, 0, 64, 64)
    
    for (h in 0 until 64) {
      for (w in 0 until 64) {
        val pixelIndex = h * 64 + w
        val pixel = pixels[pixelIndex]
        
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        
        val baseIndex = h * 64 * 3 + w * 3
        
        tensorData[baseIndex] = (b / 255.0f) * 255.0f        // B
        tensorData[baseIndex + 1] = (g / 255.0f) * 255.0f    // G  
        tensorData[baseIndex + 2] = (r / 255.0f) * 255.0f    // R
      }
    }
    
    if (resizedBitmap != croppedBitmap) {
      resizedBitmap.recycle()
    }
    
    Log.d(TAG, "Converted bitmap to ONNX input: ${tensorData.size} floats")
    return tensorData
  }

  /**
   * FloatArray를 ByteArray로 변환
   */
  private fun floatArrayToByteArray(floatArray: FloatArray): ByteArray {
    val byteBuffer = ByteBuffer.allocate(floatArray.size * 4)
    byteBuffer.order(ByteOrder.nativeOrder())
    val floatBuffer = byteBuffer.asFloatBuffer()
    floatBuffer.put(floatArray)
    return byteBuffer.array()
  }

  private fun returnLivestreamResult(result: FaceLandmarkerResult, input: MPImage) {
    val finishTimeMs = SystemClock.uptimeMillis()
    val inferenceTime = finishTimeMs - result.timestampMs()

    var croppedFrameByteArray: ByteArray? = null
    var onnxInputData: ByteArray? = null
    var faceFeatures: FaceFeatures? = null
    
    // 얼굴이 감지된 경우에만 처리
    if (result.faceLandmarks().isNotEmpty()) {
      // 얼굴 특징 계산 (이미지 크기 정보 전달)
      faceFeatures = calculateFaceFeatures(result.faceLandmarks()[0], input.width, input.height)
      
      // currentBitmap이 있는 경우에만 자르기 작업 수행
      currentBitmap?.let { sourceBitmap ->
        try {
          // 얼굴 경계 상자 계산
          var minX = 1.0f
          var minY = 1.0f
          var maxX = 0.0f
          var maxY = 0.0f
          result.faceLandmarks()[0].forEach { landmark ->
              minX = minOf(minX, landmark.x())
              minY = minOf(minY, landmark.y())
              maxX = maxOf(maxX, landmark.x())
              maxY = maxOf(maxY, landmark.y())
          }

          // 여백 추가
          val margin = 0.1f
          minX = maxOf(0.0f, minX - margin)
          minY = maxOf(0.0f, minY - margin)
          maxX = minOf(1.0f, maxX + margin)
          maxY = minOf(1.0f, maxY + margin)

          val cropX = (minX * sourceBitmap.width).toInt()
          val cropY = (minY * sourceBitmap.height).toInt()
          val cropWidth = ((maxX - minX) * sourceBitmap.width).toInt()
          val cropHeight = ((maxY - minY) * sourceBitmap.height).toInt()

          if (cropWidth > 0 && cropHeight > 0 && 
              cropX >= 0 && cropY >= 0 && 
              cropX + cropWidth <= sourceBitmap.width && 
              cropY + cropHeight <= sourceBitmap.height) {
              
              val croppedBitmap = Bitmap.createBitmap(
                  sourceBitmap,
                  cropX,
                  cropY,
                  cropWidth,
                  cropHeight
              )

              val resizedBitmap = Bitmap.createScaledBitmap(
                  croppedBitmap, 192, 192, true
              )

              // Base64 이미지 생성
              val stream = ByteArrayOutputStream()
              resizedBitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream)
              croppedFrameByteArray = stream.toByteArray()
              
              // ONNX 입력 데이터 생성
              val onnxFloatData = convertBitmapToOnnxInput(resizedBitmap)
              onnxInputData = floatArrayToByteArray(onnxFloatData)
              
              Log.d(TAG, "Successfully processed face: JPEG=${croppedFrameByteArray?.size} bytes, ONNX=${onnxInputData?.size} bytes")
              
              croppedBitmap.recycle()
              resizedBitmap.recycle()
          } else {
            Log.w(TAG, "Invalid crop dimensions: x=$cropX, y=$cropY, w=$cropWidth, h=$cropHeight")
          }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to process frame: ${e.message}", e)
        }
      }
      
      Log.d(TAG, "Face detected - sending ResultBundle with data")
    } else {
      Log.d(TAG, "No face detected - sending ResultBundle with null data")
    }

    // 얼굴 감지 여부와 관계없이 항상 ResultBundle을 반환 (얼굴이 없으면 null 데이터와 함께)
    faceLandmarkDetectorListener?.onResults(
        ResultBundle(
            results = listOf(result),
            inferenceTime = inferenceTime,
            inputImageHeight = input.height,
            inputImageWidth = input.width,
            inputImageRotation = imageRotation,
            croppedFrame = croppedFrameByteArray,   // 얼굴이 없으면 null
            onnxInputData = onnxInputData,          // 얼굴이 없으면 null
            faceFeatures = faceFeatures             // 얼굴이 없으면 null
        )
    )
    
    Log.d(TAG, "Sent ResultBundle - face detected: ${result.faceLandmarks().isNotEmpty()}")
  }

  private fun returnLivestreamError(error: RuntimeException) {
    faceLandmarkDetectorListener?.onError(
        error.message ?: "An unknown error has occurred"
    )
  }

  companion object {
    const val TAG = "FaceLandmarkDetectorHelper"
    const val DELEGATE_CPU = 0
    const val DELEGATE_GPU = 1
    const val DEFAULT_FACE_DETECTION_CONFIDENCE = 0.5F
    const val DEFAULT_FACE_TRACKING_CONFIDENCE = 0.5F
    const val DEFAULT_FACE_PRESENCE_CONFIDENCE = 0.5F
    const val DEFAULT_NUM_FACES = 1
    const val OTHER_ERROR = 0
    const val GPU_ERROR = 1
  }

  /**
   * 얼굴 특징을 담는 데이터 클래스
   */
  data class FaceFeatures(
    val earLeft: Float,
    val earRight: Float,
    val earAvg: Float,
    val gazeLeft: Float,
    val gazeRight: Float,
    val mar: Float,
    val leftEyeX: Float,
    val leftEyeY: Float,
    val rightEyeX: Float,
    val rightEyeY: Float,
    val leftIrisX: Float,
    val leftIrisY: Float,
    val rightIrisX: Float,
    val rightIrisY: Float,
    val faceCenterX: Float,
    val faceCenterY: Float,
    val leftEyeWidth: Float,
    val rightEyeWidth: Float
  )

  data class ResultBundle(
      val results: List<FaceLandmarkerResult>,
      val inferenceTime: Long,
      val inputImageHeight: Int,
      val inputImageWidth: Int,
      val inputImageRotation: Int = 0,
      val croppedFrame: ByteArray? = null,
      val onnxInputData: ByteArray? = null,
      val faceFeatures: FaceFeatures? = null
  )

  interface DetectorListener {
    fun onError(error: String, errorCode: Int = OTHER_ERROR)
    fun onResults(resultBundle: ResultBundle)
    fun onEmpty() {}
  }
}
