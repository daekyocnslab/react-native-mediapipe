package com.reactnativemediapipe.facelandmarkdetection

import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.WritableNativeMap
import com.facebook.react.bridge.WritableMap
import com.facebook.react.bridge.WritableArray
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.reactnativemediapipe.shared.loadBitmapFromPath

object FaceLandmarkDetectorMap {
  internal val detectorMap = mutableMapOf<Int, FaceLandmarkDetectorHelper>()
}

class FaceLandmarkDetectionModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

  private var nextId = 22

  override fun getName(): String {
    return "FaceLandmarkDetection"
  }

  override fun getConstants(): MutableMap<String, Any>? {
    val knownLandmarks = WritableNativeMap()
    knownLandmarks.putArray(
        "lips",
        connectionSetToWritableArray(FaceLandmarker.FACE_LANDMARKS_LIPS)
    )
    knownLandmarks.putArray(
        "leftEye",
        connectionSetToWritableArray(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE)
    )
    knownLandmarks.putArray(
        "leftEyebrow",
        connectionSetToWritableArray(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE_BROW)
    )
    knownLandmarks.putArray(
        "leftIris",
        connectionSetToWritableArray(FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS)
    )
    knownLandmarks.putArray(
        "rightEye",
        connectionSetToWritableArray(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)
    )
    knownLandmarks.putArray(
        "rightEyebrow",
        connectionSetToWritableArray(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE_BROW)
    )
    knownLandmarks.putArray(
        "rightIris",
        connectionSetToWritableArray(FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS)
    )
    knownLandmarks.putArray(
        "faceOval",
        connectionSetToWritableArray(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL)
    )
    knownLandmarks.putArray(
        "connectors",
        connectionSetToWritableArray(FaceLandmarker.FACE_LANDMARKS_CONNECTORS)
    )
    knownLandmarks.putArray(
        "tesselation",
        connectionSetToWritableArray(FaceLandmarker.FACE_LANDMARKS_TESSELATION)
    )

    return hashMapOf("knownLandmarks" to knownLandmarks)
  }

  private class FaceLandmarkDetectorListener(
      private val module: FaceLandmarkDetectionModule,
      private val handle: Int
  ) : FaceLandmarkDetectorHelper.DetectorListener {
    override fun onError(error: String, errorCode: Int) {
      module.sendErrorEvent(handle, error, errorCode)
    }

    override fun onResults(resultBundle: FaceLandmarkDetectorHelper.ResultBundle) {
      module.sendResultsEvent(handle, resultBundle)
    }
  }

  @ReactMethod
  fun createDetector(
      numFaces: Int,
      minFaceDetectionConfidence: Float,
      minFacePresenceConfidence: Float,
      minTrackingConfidence: Float,
      model: String,
      delegate: Int,
      runningMode: Int,
      promise: Promise
  ) {
    val id = nextId++
    val helper =
        FaceLandmarkDetectorHelper(
            maxNumFaces = numFaces,
            minFaceDetectionConfidence = minFaceDetectionConfidence,
            minFacePresenceConfidence = minFacePresenceConfidence,
            minFaceTrackingConfidence = minTrackingConfidence,
            currentDelegate = delegate,
            currentModel = model,
            runningMode = enumValues<RunningMode>().first { it.ordinal == runningMode },
            context = reactApplicationContext.applicationContext,
            faceLandmarkDetectorListener = FaceLandmarkDetectorListener(this, id)
        )
    FaceLandmarkDetectorMap.detectorMap[id] = helper
    promise.resolve(id)
  }

  @ReactMethod
  fun releaseDetector(handle: Int, promise: Promise) {
    val entry = FaceLandmarkDetectorMap.detectorMap[handle]
    if (entry != null) {
      entry.clearFaceLandmarker()
      FaceLandmarkDetectorMap.detectorMap.remove(handle)
    }
    promise.resolve(true)
  }

  @ReactMethod
  fun detectOnImage(
      imagePath: String,
      numFaces: Int,
      minFaceDetectionConfidence: Float,
      minFacePresenceConfidence: Float,
      minTrackingConfidence: Float,
      model: String,
      delegate: Int,
      promise: Promise
  ) {
    try {
      val helper =
          FaceLandmarkDetectorHelper(
              maxNumFaces = numFaces,
              minFaceDetectionConfidence = minFaceDetectionConfidence,
              minFacePresenceConfidence = minFacePresenceConfidence,
              minFaceTrackingConfidence = minTrackingConfidence,
              currentDelegate = delegate,
              currentModel = model,
              runningMode = RunningMode.IMAGE,
              context = reactApplicationContext.applicationContext,
              faceLandmarkDetectorListener = FaceLandmarkDetectorListener(this, 0)
          )
      val bundle = helper.detectImage(loadBitmapFromPath(imagePath))
      val resultArgs = convertResultBundleToWritableMap(bundle)

      promise.resolve(resultArgs)
    } catch (e: Exception) {
      promise.reject(e)
    }
  }

  @ReactMethod
  fun detectOnVideo(
      videoPath: String,
      threshold: Float,
      maxResults: Int,
      delegate: Int,
      model: String,
      promise: Promise
  ) {
    promise.reject(UnsupportedOperationException("detectOnVideo not yet implemented"))
  }

  @ReactMethod
  fun addListener(eventName: String?) {
    /* Required for RN built-in Event Emitter Calls. */
  }

  @ReactMethod
  fun removeListeners(count: Int?) {
    /* Required for RN built-in Event Emitter Calls. */
  }

  private fun sendErrorEvent(handle: Int, message: String, code: Int) {
    val errorArgs =
        Arguments.makeNativeMap(mapOf("handle" to handle, "message" to message, "code" to code))

    reactApplicationContext
        .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
        .emit("onError", errorArgs)
  }

  private fun sendResultsEvent(handle: Int, bundle: FaceLandmarkDetectorHelper.ResultBundle) {
    val resultArgs = convertResultBundleToWritableMap(bundle)
    resultArgs.putInt("handle", handle)
    reactApplicationContext
        .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
        .emit("onResults", resultArgs)
  }

  private fun convertResultBundleToWritableMap(bundle: FaceLandmarkDetectorHelper.ResultBundle): WritableMap {
    android.util.Log.d(TAG, "=== convertResultBundleToWritableMap 시작 ===")
    val map = Arguments.createMap()
    
    // 기본 필드들
    map.putDouble("inferenceTime", bundle.inferenceTime.toDouble())
    map.putInt("inputImageHeight", bundle.inputImageHeight)
    map.putInt("inputImageWidth", bundle.inputImageWidth)
    map.putInt("inputImageRotation", bundle.inputImageRotation)
    
    // 기존 croppedFrame (Base64 이미지)
    bundle.croppedFrame?.let { croppedBytes ->
        val base64String = android.util.Base64.encodeToString(croppedBytes, android.util.Base64.DEFAULT)
        map.putString("croppedFrame", base64String)
        android.util.Log.d(TAG, "✅ Added croppedFrame: ${croppedBytes.size} bytes -> ${base64String.length} base64 chars")
    } ?: run {
        android.util.Log.w(TAG, "❌ croppedFrame is NULL")
    }
    
    // ONNX 입력 데이터
    bundle.onnxInputData?.let { onnxBytes ->
        val base64String = android.util.Base64.encodeToString(onnxBytes, android.util.Base64.DEFAULT)
        map.putString("onnxInputData", base64String)
        android.util.Log.d(TAG, "✅ Added onnxInputData: ${onnxBytes.size} bytes -> ${base64String.length} base64 chars")
    } ?: run {
        android.util.Log.w(TAG, "❌ onnxInputData is NULL")
    }
    
    // 얼굴 특징 데이터 추가
    bundle.faceFeatures?.let { features ->
        val featuresMap = Arguments.createMap()
        featuresMap.putDouble("earLeft", features.earLeft.toDouble())
        featuresMap.putDouble("earRight", features.earRight.toDouble())
        featuresMap.putDouble("earAvg", features.earAvg.toDouble())
        featuresMap.putDouble("gazeLeft", features.gazeLeft.toDouble())
        featuresMap.putDouble("gazeRight", features.gazeRight.toDouble())
        featuresMap.putDouble("mar", features.mar.toDouble())
        featuresMap.putDouble("leftEyeX", features.leftEyeX.toDouble())
        featuresMap.putDouble("leftEyeY", features.leftEyeY.toDouble())
        featuresMap.putDouble("rightEyeX", features.rightEyeX.toDouble())
        featuresMap.putDouble("rightEyeY", features.rightEyeY.toDouble())
        featuresMap.putDouble("leftIrisX", features.leftIrisX.toDouble())
        featuresMap.putDouble("leftIrisY", features.leftIrisY.toDouble())
        featuresMap.putDouble("rightIrisX", features.rightIrisX.toDouble())
        featuresMap.putDouble("rightIrisY", features.rightIrisY.toDouble())
        featuresMap.putDouble("faceCenterX", features.faceCenterX.toDouble())
        featuresMap.putDouble("faceCenterY", features.faceCenterY.toDouble())
        featuresMap.putDouble("leftEyeWidth", features.leftEyeWidth.toDouble())
        featuresMap.putDouble("rightEyeWidth", features.rightEyeWidth.toDouble())
        
        map.putMap("faceFeatures", featuresMap)
        android.util.Log.d(TAG, "✅ Added faceFeatures")
    } ?: run {
        android.util.Log.w(TAG, "❌ faceFeatures is NULL")
    }
    
    // results 배열 처리
    val resultsArray = Arguments.createArray()
    android.util.Log.d(TAG, "Processing ${bundle.results.size} results")
    bundle.results.forEach { result ->
        val resultMap = Arguments.createMap()
        
        // 랜드마크 처리
        val landmarksArray = Arguments.createArray()
        android.util.Log.d(TAG, "Processing ${result.faceLandmarks().size} face landmark groups")
        result.faceLandmarks().forEach { facelandmarks ->
            val faceArray = Arguments.createArray()
            facelandmarks.forEach { landmark ->
                val landmarkMap = Arguments.createMap()
                landmarkMap.putDouble("x", landmark.x().toDouble())
                landmarkMap.putDouble("y", landmark.y().toDouble())
                landmarkMap.putDouble("z", landmark.z().toDouble())
                faceArray.pushMap(landmarkMap)
            }
            landmarksArray.pushArray(faceArray)
        }
        resultMap.putArray("landmarks", landmarksArray)
        
        // 블렌드셰이프 처리 (있는 경우)
        val blendshapesArray = Arguments.createArray()
        if (result.faceBlendshapes().isPresent) {
            result.faceBlendshapes().get().forEach { faceBlendshapes ->
                val faceBlendshapesArray = Arguments.createArray()
                faceBlendshapes.forEach { blendshape ->
                    val blendshapeMap = Arguments.createMap()
                    blendshapeMap.putString("categoryName", blendshape.categoryName())
                    blendshapeMap.putDouble("score", blendshape.score().toDouble())
                    faceBlendshapesArray.pushMap(blendshapeMap)
                }
                blendshapesArray.pushArray(faceBlendshapesArray)
            }
        }
        resultMap.putArray("faceBlendshapes", blendshapesArray)
        
        // 기타 필드들
        resultMap.putArray("worldLandmarks", Arguments.createArray())
        resultMap.putArray("segmentationMask", Arguments.createArray())
        resultMap.putArray("facialTransformationMatrixes", Arguments.createArray())
        
        resultsArray.pushMap(resultMap)
    }
    map.putArray("results", resultsArray)
    
    android.util.Log.d(TAG, "=== convertResultBundleToWritableMap 완료 ===")
    return map
  }

  companion object {
    const val TAG = "FaceLandmarkDetectionModule"
  }
}