package com.reactnativemediapipe.posedetection

import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.WritableMap
import com.facebook.react.bridge.WritableNativeArray
import com.facebook.react.bridge.WritableNativeMap
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import com.reactnativemediapipe.shared.landmarkToWritableMap
import com.reactnativemediapipe.shared.normalizedLandmarkToWritableMap

object PoseMap {
  val POSE_LANDMARKS = mapOf(
    0 to "nose",
    15 to "left_wrist",
    16 to "right_wrist", 
    19 to "left_index",
    20 to "right_index"
  )
}

fun convertResultBundleToWritableMap(resultBundle: PoseDetectorHelper.ResultBundle): WritableMap {
  val map = Arguments.createMap()
  val resultsArray = Arguments.createArray()
  resultBundle.results.forEach { result ->
    resultsArray.pushMap(poseLandmarkerResultToWritableMap(result, resultBundle.inputImageWidth, resultBundle.inputImageHeight))
  }
  map.putArray("results", resultsArray)
  map.putInt("inputImageHeight", resultBundle.inputImageHeight)
  map.putInt("inputImageWidth", resultBundle.inputImageWidth)
  map.putDouble("inferenceTime", resultBundle.inferenceTime.toDouble())
  return map
}

fun poseLandmarkerResultToWritableMap(result: PoseLandmarkerResult, imageWidth: Int, imageHeight: Int): WritableMap {
  val resultMap = WritableNativeMap()
  val landmarksArray = WritableNativeArray()
  val worldLandmarksArray = WritableNativeArray()
  val selectedLandmarksArray = WritableNativeArray() // 추가: 선택된 landmark만 저장
  
  result.landmarks().forEach { landmarks ->
    val landmarkArray = WritableNativeArray()
    val selectedLandmarkMap = WritableNativeMap() // 추가: 선택된 landmark 맵
    
    landmarks.forEachIndexed { index, landmark ->
      landmarkArray.pushMap(normalizedLandmarkToWritableMap(landmark))
      
      // Python의 pose_map에 해당하는 landmark만 추출
      if (PoseMap.POSE_LANDMARKS.containsKey(index)) {
        val landmarkName = PoseMap.POSE_LANDMARKS[index]!!
        val landmarkData = WritableNativeMap()
        
        // visibility가 0.5 이상일 때만 좌표 저장 (Python 코드와 동일)
        if (landmark.visibility().isPresent && landmark.visibility().get() > 0.5f) {
          // normalized 좌표에 이미지 크기를 곱해서 실제 픽셀 좌표로 변환
          val actualX = landmark.x() * imageWidth
          val actualY = landmark.y() * imageHeight
          
          landmarkData.putDouble("x", actualX.toDouble())
          landmarkData.putDouble("y", actualY.toDouble())
          landmarkData.putDouble("visibility", landmark.visibility().get().toDouble())
        } else {
          // visibility가 0.5 이하일 때는 null 저장 (Python에서 np.nan과 동일)
          landmarkData.putNull("x")
          landmarkData.putNull("y")
          landmarkData.putDouble("visibility", landmark.visibility().orElse(0.0f).toDouble())
        }
        
        selectedLandmarkMap.putMap(landmarkName, landmarkData)
      }
    }
    
    landmarksArray.pushArray(landmarkArray)
    selectedLandmarksArray.pushMap(selectedLandmarkMap) // 선택된 landmark 추가
  }

  result.worldLandmarks().forEach { worldLandmarks ->
    val worldLandmarkArray = WritableNativeArray()
    worldLandmarks.forEach { it -> worldLandmarkArray.pushMap(landmarkToWritableMap(it)) }
    worldLandmarksArray.pushArray(worldLandmarkArray)
  }

  resultMap.putArray("landmarks", landmarksArray)
  resultMap.putArray("worldLandmarks", worldLandmarksArray)
  resultMap.putArray("segmentationMask", WritableNativeArray())
  resultMap.putArray("selectedLandmarks", selectedLandmarksArray) // 추가: 선택된 landmark 배열

  return resultMap
}